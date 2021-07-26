# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel

from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import (
    _TORCH_GREATER_EQUAL_1_7,
    _TORCH_GREATER_EQUAL_1_8,
    rank_zero_deprecation,
    rank_zero_warn,
)
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.distributed import (
    distributed_available,
    rank_zero_info,
    rank_zero_only,
    ReduceOp,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.seed import reset_seed

if _TORCH_GREATER_EQUAL_1_8:
    from pytorch_lightning.utilities.distributed import register_ddp_comm_hook

log = logging.getLogger(__name__)


class DDPSpawnPlugin(ParallelPlugin):
    """
    Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after
    training finishes.
    """

    distributed_backend = "ddp_spawn"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: Optional[int] = None,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: Optional[bool] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        **kwargs: Any,
    ):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment)
        if num_nodes is not None:
            rank_zero_deprecation(
                "Argument `num_nodes` in `DDPSpawnPlugin` is deprecated in v1.4, and will be removed in v1.6. "
                "Notice that it will be overriden by the trainer setting."
            )
        self._num_nodes = num_nodes or 1
        if sync_batchnorm is not None:
            rank_zero_deprecation(
                "Argument `sync_batchnorm` in `DDPSpawnPlugin` is deprecated in v1.4, and will be removed in v1.6. "
                "Notice that it will be overriden by the trainer setting."
            )
        self._sync_batchnorm = sync_batchnorm or False
        self._ddp_kwargs = kwargs
        self.dist = LightningDistributed()
        self.num_processes = len(parallel_devices) if parallel_devices is not None else 0
        self.mp_queue = None
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._local_rank = 0
        self.set_world_ranks()

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes
        self.set_world_ranks()

    @property
    def sync_batchnorm(self) -> bool:
        return self._sync_batchnorm

    @sync_batchnorm.setter
    def sync_batchnorm(self, sync_batchnorm: bool) -> None:
        self._sync_batchnorm = sync_batchnorm

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def __getstate__(self):
        """Makes this plugin pickleable without destroying the queue in the current process."""
        state = self.__dict__.copy()
        state["mp_queue"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self):
        return True

    def setup(self, model):
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        # pass in a state q
        smp = mp.get_context("spawn")
        self.mp_queue = smp.SimpleQueue()

    def set_world_ranks(self, process_idx: int = 0) -> None:
        self._local_rank = process_idx
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    @property
    def mp_spawn_kwargs(self):
        return {"args": (self.lightning_module.trainer, self.mp_queue), "nprocs": self.num_processes}

    def start_training(self, trainer):
        mp.spawn(self.new_process, **self.mp_spawn_kwargs)
        # reset optimizers, since main process is never used for training and thus does not have a valid optim state
        trainer.optimizers = []

    def start_evaluating(self, trainer):
        mp.spawn(self.new_process, **self.mp_spawn_kwargs)

    def start_predicting(self, trainer):
        mp.spawn(self.new_process, **self.mp_spawn_kwargs)

    def new_process(self, process_idx, trainer, mp_queue):
        self.mp_queue = mp_queue

        reset_seed()

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: we moved it to the trainer.fit after calling pre_dispatch
        #   ... need to double check that it is the correct place
        # self.trainer.call_setup_hook(self.model)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        # move the model to the correct device
        self.model_to_device()

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        self.configure_ddp()

        self.barrier()

        results = trainer.run_stage()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(results)

    def post_dispatch(self):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        last_path = self.mp_queue.get()
        self._results = self.mp_queue.get()
        # get the `callback_metrics` and set it to the trainer
        # only in case the user does not override it.
        self.lightning_module.get_from_queue(self.mp_queue)

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(best_path, last_path)

    def pre_configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        # todo: PyTorch 1.7.0 DDP introduces `self.reducer._rebuild_buckets()` breaking manual_optimization
        if (
            _TORCH_GREATER_EQUAL_1_7
            and not self.lightning_module.automatic_optimization
            and not self._ddp_kwargs.get("find_unused_parameters", False)
        ):
            rank_zero_warn(
                "From PyTorch 1.7.0, Lightning ``manual_optimization`` needs to set ``find_unused_parameters=True`` "
                "to properly work with DDP."
            )
            self._ddp_kwargs["find_unused_parameters"] = True

    def _register_ddp_hooks(self) -> None:
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if _TORCH_GREATER_EQUAL_1_8 and self.on_gpu and self._is_single_process_single_device:
            register_ddp_comm_hook(
                model=self._model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model), device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs
        )
        self._register_ddp_hooks()

    def init_ddp_connection(self, global_rank: Optional[int], world_size: Optional[int]) -> None:
        # TODO: this code is duplicated in DDP and DDPSpawn, make this a function
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                self.torch_distributed_backend, rank=global_rank, world_size=world_size
            )

            # on rank=0 let everyone know training is starting
            rank_zero_info(
                f"{'-' * 100}\n"
                f"distributed_backend={self.torch_distributed_backend}\n"
                f"All DDP processes registered. Starting ddp with {self.world_size} processes\n"
                f"{'-' * 100}\n"
            )

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        checkpoint_callback = self.lightning_module.trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = self.lightning_module.state_dict()

        if self.global_rank == 0 and self.mp_queue is not None:
            rank_zero_warn("cleaning up ddp environment...")

            # save the last weights
            last_path = None
            if (
                self.lightning_module.trainer.state.fn == TrainerFn.FITTING
                and best_model_path is not None
                and len(best_model_path) > 0
            ):
                last_path = re.sub(".ckpt", ".tmp_end.ckpt", best_model_path)
                atomic_save(self.on_save(state_dict), last_path)

            # todo, pass complete checkpoint as state dictionary
            self.mp_queue.put(best_model_path)
            self.mp_queue.put(last_path)
            self.mp_queue.put(results)
            self.lightning_module.add_to_queue(self.mp_queue)  # adds the `callback_metrics` to the queue

    def __recover_child_process_weights(self, best_path, last_path):
        # transfer back the best path to the trainer
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        if last_path is not None and self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            ckpt = pl_load(last_path, map_location=lambda storage, loc: storage)
            self.lightning_module.load_state_dict(ckpt)

    def barrier(self, *args, **kwargs) -> None:
        if not distributed_available():
            return
        if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not distributed_available():
            return obj
        return self.dist.broadcast(obj)

    def model_to_device(self):
        if self.root_device.type == "cuda":
            # set the device on the spawned subprocesses
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        """Run before precision plugin executes backward"""
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, closure_loss)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Union[ReduceOp, str] = "mean") -> torch.Tensor:
        """
        Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, torch.Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def post_training_step(self):
        if not self.lightning_module.automatic_optimization:
            self.model.require_backward_grad_sync = True

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register(
            "ddp_spawn_find_unused_parameters_false",
            cls,
            description="DDPSpawn Plugin with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
