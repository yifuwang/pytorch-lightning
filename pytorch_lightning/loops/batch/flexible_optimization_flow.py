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
from typing import Iterator, List, Optional, Tuple

import itertools
import pytorch_lightning as pl
import torch
import logging

from copy import copy
from collections import OrderedDict
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.batch_loop_common import (
    check_finite,
    check_training_step_output,
    process_training_step_output,
)


log = logging.getLogger(__name__)


class FlexibleOptimizationFlow:
    """
    Other than that, it's effectively manual optimization loop without tbppt support.
    TODO: doc
    Manual optimization only
    - no running loss support
    No graident accumulation support
    No tbptt support

    Contract: the training_step get access to dataloader_iter and can work with
    multiple batches at once. However, each time it's called it can optimize
    the model with one and only one batch.

    return the loss
    """

    def __init__(self, trainer_ref: "pl.Trainer", model_ref: "pl.LightningModule") -> None:
        if is_overridden("on_train_batch_start", model_ref):
            raise MisconfigurationException(
                "The model hook `on_train_batch_start` is not compatible with FlexibleOptimizationFlow."
            )
        if is_overridden("on_train_batch_end", model_ref):
            raise MisconfigurationException(
                "The model hook `on_train_batch_end` is not compatible with FlexibleOptimizationFlow."
            )
        if is_overridden("tbptt_split_batch", model_ref):
            raise MisconfigurationException(
                "The model hook `tbptt_split_batch` is not compatible with FlexibleOptimizationFlow."
            )
        if model_ref.automatic_optimization:
            raise MisconfigurationException("`automatic_optimization` is not support by FlexibleOptimizationFlow.")
        if trainer_ref.accumulate_grad_batches != 1:
            raise MisconfigurationException(
                "`accumulate_grad_batches` can only be 1 when using FlexibleOptimizationFlow."
            )

        self.trainer_ref = trainer_ref
        # Initialize placeholder `running_loss` and `optim_progress` to comply
        # with `FitLoop`'s expectation. These are not used for manual optimization.
        self.running_loss: TensorRunningAccum = TensorRunningAccum(window_length=1)
        self.optim_progress = OptimizationProgress()

    def num_active_optimizers(self, batch_idx: Optional[int] = None) -> int:
        """
        Returns the number of active optimizers.
        """
        return len(self.trainer_ref.optimizers)

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, torch.optim.Optimizer]]:
        """
        Returns the currently active optimizers.

        Returns:
            A list of tuples (opt_idx, optimizer) of currently active optimizers.
        """
        return list(enumerate(self.trainer_ref.optimizers))

    def run(self, dataloader_iter: Iterator) -> Optional[AttributeDict]:
        """
        Args:
            dataloader_iter: the iterator over the dataloader producing the new batch
        """
        dataloader_iter = itertools.starmap(
            lambda batch_idx, batch_with_is_last: batch_with_is_last[0], dataloader_iter
        )

        self.trainer_ref.logger_connector.on_batch_start()
        response = self.trainer_ref.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1)

        self.trainer_ref.fit_loop.epoch_loop.batch_progress.increment_started()

        # give the PL module a result for logging
        model_ref = self.trainer_ref.lightning_module

        with self.trainer_ref.profiler.profile("model_forward"):
            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer_ref.profiler.profile("training_step"):
                step_kwargs = OrderedDict([("dataloader_iter", dataloader_iter)])
                training_step_output = self.trainer_ref.accelerator.training_step(step_kwargs)
                self.trainer_ref.accelerator.post_training_step()

            training_step_output = self.trainer_ref.call_hook("training_step_end", training_step_output)
            check_training_step_output(self.trainer_ref, training_step_output)

            if training_step_output is None or "is_last" not in training_step_output:
                raise MisconfigurationException(
                    "When using `FlexibleOptimizationFlow`, `training_step` must return a dict containing `is_last` "
                    "which indicated whether there are more batches to be processed."
                )
            is_last = training_step_output["is_last"]
            training_step_output, _ = process_training_step_output(self.trainer_ref, training_step_output)

            if self.trainer_ref.terminate_on_nan:
                check_finite(self.trainer_ref, training_step_output.minimize)

        batch_outputs = [[] for _ in range(len(self.trainer_ref.optimizers))]

        batch_outputs[0].append(copy(training_step_output))
        return AttributeDict(signal=0, training_step_output=batch_outputs, is_last=is_last)

    def teardown(self) -> None:
        """
        No-op. Only defined to comply with FitLoop's expectation.
        """
        pass
