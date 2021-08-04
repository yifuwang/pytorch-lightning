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
""" Test deprecated functionality which will be removed in v1.7.0 """

import pytest

from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringModel


def test_v1_7_0_deprecated_lightning_module_summarize(tmpdir):
    from pytorch_lightning.core.lightning import warning_cache

    model = BoringModel()
    model.summarize(max_depth=1)
    assert any("The `LightningModule.summarize` method is deprecated in v1.5" in w for w in warning_cache)
    warning_cache.clear()


def test_v1_7_0_moved_model_summary_and_layer_summary(tmpdir):
    _soft_unimport_module("pytorch_lightning.core.memory")
    with pytest.deprecated_call(match="to `pytorch_lightning.utilities.model_summary` since v1.5"):
        from pytorch_lightning.core.memory import LayerSummary, ModelSummary  # noqa: F401


def test_v1_7_0_moved_get_memory_profile_and_get_gpu_memory_map(tmpdir):
    _soft_unimport_module("pytorch_lightning.core.memory")
    with pytest.deprecated_call(match="to `pytorch_lightning.utilities.memory` since v1.5"):
        from pytorch_lightning.core.memory import get_gpu_memory_map, get_memory_profile  # noqa: F401


def test_v1_7_0_deprecated_model_size():
    model = BoringModel()
    with pytest.deprecated_call(
        match="LightningModule.model_size` property was deprecated in v1.5 and will be removed in v1.7"
    ):
        _ = model.model_size
