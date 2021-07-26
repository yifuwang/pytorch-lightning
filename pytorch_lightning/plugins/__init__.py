from pytorch_lightning.plugins.base_plugin import Plugin  # noqa: F401
from pytorch_lightning.plugins.plugins_registry import (  # noqa: F401
    call_training_type_register_plugins,
    TrainingTypePluginsRegistry,
)
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.deepspeed_precision import DeepSpeedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.fully_sharded_native_amp import (  # noqa: F401
    FullyShardedNativeMixedPrecisionPlugin,
)
from pytorch_lightning.plugins.precision.ipu_precision import IPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu_bfloat import TPUHalfPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp2 import DDP2Plugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.fully_sharded import DDPFullyShardedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ipu import IPUPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.sharded import DDPShardedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin  # noqa: F401

__all__ = [
    "ApexMixedPrecisionPlugin",
    "DataParallelPlugin",
    "DDP2Plugin",
    "DDPPlugin",
    "DDPSpawnPlugin",
    "DDPFullyShardedPlugin",
    "DeepSpeedPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "HorovodPlugin",
    "IPUPlugin",
    "IPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "SingleDevicePlugin",
    "SingleTPUPlugin",
    "TPUHalfPrecisionPlugin",
    "TPUSpawnPlugin",
    "TrainingTypePlugin",
    "ParallelPlugin",
    "Plugin",
    "DDPShardedPlugin",
    "DDPSpawnShardedPlugin",
]

from pathlib import Path

FILE_ROOT = Path(__file__).parent
TRAINING_TYPE_BASE_MODULE = "pytorch_lightning.plugins.training_type"

call_training_type_register_plugins(FILE_ROOT, TRAINING_TYPE_BASE_MODULE)
