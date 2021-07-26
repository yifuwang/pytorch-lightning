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
from abc import ABC


class TrainingStepVariations(ABC):
    """
    Houses all variations of training steps
    """

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        self.training_step_called = True

        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        # calculate loss
        loss_train = self.loss(y, y_hat)
        return {"loss": loss_train}

    def training_step__multiple_dataloaders(self, batch, batch_idx, optimizer_idx=None):
        """Training step for multiple train loaders"""

        assert isinstance(batch, dict)
        assert len(batch) == 2

        assert "a_b" in batch and "c_d_e" in batch, batch.keys()
        assert isinstance(batch["a_b"], list) and len(batch["a_b"]) == 2
        assert isinstance(batch["c_d_e"], list) and len(batch["c_d_e"]) == 3

        # forward pass
        x, y = batch["a_b"][0]
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        return {"loss": loss_val}
