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
import os
import sys

import pytest
import torch

from tests.helpers.runif import RunIf

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from pytorch_lightning import LightningModule  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from tests.helpers.boring_model import BoringModel  # noqa: E402


# TODO(Borda): When multi-node tests are re-enabled (.github/workflows/ci_test-mnodes.yml)
# use an environment variable `PL_RUNNING_MULTINODE_TESTS` and set `RunIf(multinode=True)`
@pytest.mark.skip("Multi-node testing is currently disabled")
@RunIf(special=True)
def test_logging_sync_dist_true_ddp(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with CPU (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log("foo", torch.tensor(fake_result), on_step=False, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("bar", torch.tensor(fake_result), on_step=False, on_epoch=True)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        accelerator="ddp",
        gpus=1,
        num_nodes=2,
    )
    trainer.fit(model)

    assert trainer.logged_metrics["foo"] == fake_result
    assert trainer.logged_metrics["bar"] == fake_result


# TODO(Borda): When multi-node tests are re-enabled (.github/workflows/ci_test-mnodes.yml)
# use an environment variable `PL_RUNNING_MULTINODE_TESTS` and set `RunIf(multinode=True)`
@pytest.mark.skip("Multi-node testing is currently disabled")
@RunIf(special=True)
def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch)
            acc = acc + batch_idx
            self.log("a", acc, on_step=True, on_epoch=True)
            self.log("a2", 2)

            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch)
            acc = acc + batch_idx
            self.log("b", acc, on_step=True, on_epoch=True)
            self.training_step_called = True

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
        accelerator="ddp",
        gpus=1,
        num_nodes=2,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    assert set(trainer.logged_metrics) == {"a2", "a_step", "a_epoch", "b_step", "b_epoch", "epoch"}

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow b_step... it's silly to monitor val step metrics
    assert set(trainer.callback_metrics) == {"a", "a2", "b", "a_epoch", "b_epoch", "a_step"}
