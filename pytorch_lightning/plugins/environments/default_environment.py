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
from typing import Optional

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities.distributed import find_free_network_port


class DefaultEnvironment(ClusterEnvironment):
    """
    A default environment for a single node or free cluster (not managed).

    The master process must be launched by the user and Lightning will spawn new
    worker processes for distributed training, either in a single node or across multiple nodes.

    If the master address and port are not provided, the default environment will choose them
    automatically. It is recommended to use this default environment for single-node distributed
    training as it provides the most convenient way to launch the training script.
    """

    def __init__(self):
        super().__init__()

    def spawns_children(self) -> bool:
        return False

    def master_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    def master_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", find_free_network_port()))

    def world_size(self) -> Optional[int]:
        return None

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        group_rank = os.environ.get("GROUP_RANK", 0)
        return int(os.environ.get("NODE_RANK", group_rank))
