from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


def test_unsupported_precision_plugins():
    """Test error messages are raised for unsupported precision plugins with CPU."""
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")), precision_plugin=MixedPrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match=r"AMP \+ CPU is not supported"):
        accelerator.setup(trainer=trainer, model=model)


@pytest.mark.parametrize("delay_dispatch", [True, False])
def test_plugin_setup_optimizers_in_pre_dispatch(tmpdir, delay_dispatch):
    """
    Test when using a custom training type plugin that delays setup optimizers,
    we do not call setup optimizers till ``pre_dispatch``.
    """

    class TestModel(BoringModel):
        def on_fit_start(self):
            if delay_dispatch:
                # Ensure we haven't setup optimizers if we've delayed dispatch
                assert len(self.trainer.optimizers) == 0
            else:
                assert len(self.trainer.optimizers) > 0

        def on_fit_end(self):
            assert len(self.trainer.optimizers) > 0

    class CustomPlugin(SingleDevicePlugin):
        @property
        def setup_optimizers_in_pre_dispatch(self) -> bool:
            return delay_dispatch

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, plugins=CustomPlugin(device=torch.device("cpu")))
    trainer.fit(model)


def test_accelerator_on_reset_dataloader_hooks(tmpdir):
    """
    Ensure data-loader hooks are called using an Accelerator.
    """

    class CustomAccelerator(CPUAccelerator):
        train_count: int = 0
        val_count: int = 0
        test_count: int = 0
        predict_count: int = 0

        def on_reset_train_dataloader(self, dataloader):
            self.train_count += 1
            assert self.lightning_module.trainer.training
            return super().on_reset_train_dataloader(dataloader)

        def on_reset_val_dataloader(self, dataloader):
            self.val_count += 1
            assert self.lightning_module.trainer.training or self.lightning_module.trainer.validating
            return super().on_reset_val_dataloader(dataloader)

        def on_reset_test_dataloader(self, dataloader):
            self.test_count += 1
            assert self.lightning_module.trainer.testing
            return super().on_reset_test_dataloader(dataloader)

        def on_reset_predict_dataloader(self, dataloader):
            self.predict_count += 1
            assert self.lightning_module.trainer.predicting
            return super().on_reset_predict_dataloader(dataloader)

    model = BoringModel()
    accelerator = CustomAccelerator(PrecisionPlugin(), SingleDevicePlugin(device=torch.device("cpu")))
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator=accelerator)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, dataloaders=model.test_dataloader())
    # assert that all loader hooks were called
    assert accelerator.train_count == 1
    assert accelerator.val_count == 1  # only called once during the entire session
    assert accelerator.test_count == 1
    assert accelerator.predict_count == 1

    accelerator = CustomAccelerator(PrecisionPlugin(), SingleDevicePlugin(device=torch.device("cpu")))
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator=accelerator)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
    # assert val/test/predict loader hooks were called
    assert accelerator.val_count == 1
    assert accelerator.test_count == 1
    assert accelerator.predict_count == 1


def test_plugin_on_reset_dataloader_hooks(tmpdir):
    """
    Ensure data-loader hooks are called using a Plugin.
    """

    class CustomPlugin(SingleDevicePlugin):
        train_count: int = 0
        val_count: int = 0
        test_count: int = 0
        predict_count: int = 0

        def on_reset_train_dataloader(self, dataloader):
            self.train_count += 1
            assert self.lightning_module.trainer.training
            return super().on_reset_train_dataloader(dataloader)

        def on_reset_val_dataloader(self, dataloader):
            self.val_count += 1
            assert self.lightning_module.trainer.training or self.lightning_module.trainer.validating
            return super().on_reset_val_dataloader(dataloader)

        def on_reset_test_dataloader(self, dataloader):
            self.test_count += 1
            assert self.lightning_module.trainer.testing
            return super().on_reset_test_dataloader(dataloader)

        def on_reset_predict_dataloader(self, dataloader):
            self.predict_count += 1
            assert self.lightning_module.trainer.predicting
            return super().on_reset_predict_dataloader(dataloader)

    plugin = CustomPlugin(device=torch.device("cpu"))
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, plugins=plugin)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, dataloaders=model.test_dataloader())
    # assert that all loader hooks were called
    assert plugin.train_count == 1
    assert plugin.val_count == 1  # only called once during the entire session
    assert plugin.test_count == 1
    assert plugin.predict_count == 1
    plugin = CustomPlugin(device=torch.device("cpu"))
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, plugins=plugin)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
    # assert val/test/predict loader hooks were called
    assert plugin.val_count == 1
    assert plugin.test_count == 1
    assert plugin.predict_count == 1
