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

import pytest
import torch
from torch import nn
from torchmetrics import Metric as TMetric

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import Metric as PLMetric
from pytorch_lightning.metrics import MetricCollection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


class SumMetric(TMetric):
    def __init__(self):
        super().__init__()
        self.add_state("x", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


class DiffMetric(PLMetric):
    def __init__(self):
        super().__init__()
        self.add_state("x", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.x -= x

    def compute(self):
        return self.x


def test_metric_lightning(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()

            return self.step(x)

        def training_epoch_end(self, outs):
            assert torch.allclose(self.sum, self.metric.compute())
            self.sum = 0.0
            self.metric.reset()

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)


def test_metric_lightning_log(tmpdir):
    """Test logging a metric object and that the metric state gets reset after each epoch."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric_step = SumMetric()
            self.metric_epoch = SumMetric()
            self.sum = 0.0
            self.total_sum = 0.0

        def on_epoch_start(self):
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric_step(x.sum())
            self.sum += x.sum()
            self.log("sum_step", self.metric_step, on_epoch=True, on_step=False)
            return {"loss": self.step(x), "data": x}

        def training_epoch_end(self, outs):
            total = torch.stack([o["data"] for o in outs]).sum()
            self.metric_epoch(total)
            self.log("sum_epoch", self.metric_epoch)
            self.total_sum = total

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(torch.tensor(logged["sum_step"]), model.sum)
    assert torch.allclose(torch.tensor(logged["sum_epoch"]), model.total_sum)


def test_scriptable(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            # the metric is not used in the module's `forward`
            # so the module should be exportable to TorchScript
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            self.log("sum", self.metric, on_epoch=True, on_step=False)
            return self.step(x)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)
    rand_input = torch.randn(10, 32)

    script_model = model.to_torchscript()

    # test that we can still do inference
    output = model(rand_input)
    script_output = script_model(rand_input)
    assert torch.allclose(output, script_output)


def test_metric_collection_lightning_log(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = MetricCollection([SumMetric(), DiffMetric()])
            self.sum = 0.0
            self.diff = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            metric_vals = self.metric(x.sum())
            self.sum += x.sum()
            self.diff -= x.sum()
            self.log_dict({f"{k}_step": v for k, v in metric_vals.items()})
            return self.step(x)

        def training_epoch_end(self, outputs):
            metric_vals = self.metric.compute()
            self.log_dict({f"{k}_epoch": v for k, v in metric_vals.items()})

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(torch.tensor(logged["SumMetric_epoch"]), model.sum)
    assert torch.allclose(torch.tensor(logged["DiffMetric_epoch"]), model.diff)


def test_log_metric_no_attributes_raises(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            metric = SumMetric()
            self.log("foo", metric)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    model = TestModel()
    with pytest.raises(MisconfigurationException, match="Could not find the `LightningModule` attribute"):
        trainer.fit(model)


def test_log_metric_dict(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metrics = nn.ModuleDict({"sum": SumMetric(), "diff": DiffMetric()})
            self.sum = 0.0
            self.diff = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metrics["sum"](x.sum())
            self.metrics["diff"](x.sum())
            self.sum += x.sum()
            self.diff -= x.sum()
            self.log_dict({f"{k}_step": v for k, v in self.metrics.items()})
            return self.step(x)

        def training_epoch_end(self, outputs):
            self.metrics["sum"].compute()
            self.metrics["diff"].compute()
            self.log_dict({f"{k}_epoch": v for k, v in self.metrics.items()})

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(torch.tensor(logged["sum_epoch"]), model.sum)
    assert torch.allclose(torch.tensor(logged["diff_epoch"]), model.diff)
