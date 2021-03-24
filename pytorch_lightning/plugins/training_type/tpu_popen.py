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
import io
import sys
import os
import sys
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
import re
from typing import Any, Dict, Iterable, List, Optional, Union
import logging
import subprocess
from time import sleep
import numpy as np
import torch
import torch.distributed as torch_distrib
import torch.multiprocessing as mp
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.plugins.training_type.utils import on_colab_kaggle
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import _TPU_AVAILABLE, _HYDRA_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.distributed import rank_zero_only, ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.core.xla_model import rendezvous
    from torch_xla.distributed.parallel_loader import ParallelLoader
    import torch_xla.core.xla_env_vars as xenv
else:
    xm, xla_pl, xmp, ParallelLoader, rendezvous = [None] * 5


if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path


log = logging.getLogger(__name__)

class TPUPopenlugin(DDPPlugin):

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(
            parallel_devices, num_nodes=num_nodes, cluster_environment=None, sync_batchnorm=False, **kwargs
        )
        self.tpu_local_core_rank = 0

    @property
    def distributed_sampler_kwargs(self) -> dict:
        return dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    @property
    def is_distributed(self):
        return self.world_size != 1

    def setup_distributed(self):
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

    def setup_environment(self):
        # start the other scripts
        if os.environ.get("PL_IN_DDP_SUBPROCESS", "0") != "1":
            self._prepare_children_env()
            self._call_children_scripts()

        self._setup_xla()

        self.setup_distributed()

    def _prepare_children_env(self):
        _host_world_size = self.num_processes * self.num_nodes
        pf_cfg = xmp._pre_fork_setup(_host_world_size)
        os.environ["PL_DEV_KIND"] = str(pf_cfg.dev_kind)
        os.environ["PL_NUM_DEVICES"] = str(_host_world_size)
        os.environ["PL_TPU_AVAILABLE"] = "1"

    def _setup_xla(self) -> int:
        index = os.getenv("LOCAL_RANK", "0")
        num_processes = os.getenv("PL_NUM_DEVICES")
        dev_kind = os.getenv("PL_DEV_KIND")
        pf_cfg = xmp.PreForkConfig(dev_kind, num_processes)
        xmp._prepare_env_for_index(index, pf_cfg)

        print("device")
        device = xm.xla_device(index)
        print(device)
        print(index, os.environ)
        xmp._setup_replication()
        
        return int(index)

    def _call_children_scripts(self):

        # bookkeeping of spawned processes
        assert self.global_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        # when user is using hydra find the absolute path
        path_lib = os.path.abspath if not _HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception:
            full_path = os.path.abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.parallel_devices is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        if self.lightning_module.logger is not None:
            os.environ["PL_EXP_VERSION"] = str(self.lightning_module.logger.version)

        num_gpus = len(self.parallel_devices)

        self.interactive_ddp_procs = []

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
                    os_cwd = f'"{os.getcwd()}"'
                    command += [f'hydra.run.dir={os_cwd}', f'hydra.job.name=train_ddp_process_{local_rank}']
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def process_dataloader(self, dataloader: Union[Iterable, torch.utils.data.DataLoader]) -> ParallelLoader:
        device = xm.xla_device()
        dataloader = xla_pl.ParallelLoader(dataloader, [device])
        dataloader = dataloader.per_device_loader(device)
        return dataloader

    def configure_ddp(self) -> None:
        pass

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

    def set_world_ranks(self) -> None:
        print("SETTING WORLD RANKS")
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()
        self.global_rank = self.tpu_local_core_rank
        self.world_size = self.num_nodes * self.num_processes
        print("DONE set_world_ranks")

    def model_to_device(self) -> None:
        print("HERE")
        self._model.to(xm.xla_device())

    def barrier(self, name: Optional[str] = None) -> None:
        if torch_distrib.is_initialized():
            rendezvous(f"pl.Trainer.{name}")

    def save(self, state_dict: Dict, path: str) -> None:
        """
        Saving with ``xm.save`` can be unstable and miss the rendez-vous after ``torch.save``.
        The rendez-vous doesn't affect directly saving.
        We can ignore the ``RuntimeError`` to reduce friction with TPUs.
        """
        try:
            xm.save(state_dict, path)
        except RuntimeError as e:
            if "Failed to meet rendezvous" not in str(e):
                raise e

    def broadcast(self, obj: object, src: int = 0) -> object:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data).to(xm.xla_device(), dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def load_spawn_weights(self, original_model: LightningModule) -> LightningModule:
        """
        Load the temp weights saved in the process
        To recover the trained model from the ddp process we load the saved weights
        """

        loaded_model = original_model

        if self.is_global_zero:
            # load weights saved in ddp
            path = os.path.join(original_model.trainer.default_root_dir, "__temp_weight_distributed_end.ckpt")
            loaded_model = original_model.__class__.load_from_checkpoint(path)

            # copy loaded weights to old model
            original_model.load_state_dict(loaded_model.state_dict())

            # remove ddp weights
            os.remove(path)

        return loaded_model

    def save_spawn_weights(self, model: LightningModule) -> Optional[str]:
        """
        Dump a temporary checkpoint after ddp ends to get weights out of the process
        """
        if model.trainer.is_global_zero:
            path = os.path.join(model.trainer.default_root_dir, "__temp_weight_distributed_end.ckpt")
            model.trainer.save_checkpoint(path)
            return path

    def reduce_decision(self, decision: bool) -> bool:
        decision = torch.tensor(int(decision), device=self.device)
        decision = self.reduce(decision, "sum")
        decision = bool(decision == self.world_size)
        return decision

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, device=self.device)

        _invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        _invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if _invalid_reduce_op or _invalid_reduce_op_str:
            raise MisconfigurationException(
                "Currently, TPUSpawn TrainingTypePlugin only support `sum`, `mean`, `avg` reduce operation."
            )

        output = xm.mesh_reduce('reduce', output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def post_dispatch(self) -> None:
        pass

    def start_training(self, trainer) -> None:
        pass

    def start_evaluating(self, trainer) -> None:
        pass

    def start_predicting(self, trainer) -> None:
        pass

    def training_step(self, *args, **kwargs):
        return self.lightning_module.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.lightning_module.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.lightning_module.predict_step(*args, **kwargs)

    def save_checkpoint(self, filepath, weights_only: bool = False):
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        # dump states as a checkpoint dictionary object
        _checkpoint = self.lightning_module.trainer.checkpoint_connector.dump_checkpoint(weights_only)
        # Todo: TypeError: 'mappingproxy' object does not support item assignment
        self.save({k: v for k, v in _checkpoint.items() if k != "callbacks"}, filepath)
