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
from collections import OrderedDict

import torch


class TestStepVariations(ABC):
    """
    Houses all variations of test steps
    """

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        self.test_step_called = True

        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        test_acc = test_acc.type_as(x)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc})
            return output
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            output = OrderedDict(
                {"test_loss": loss_test, "test_acc": test_acc, "test_dic": dict(test_loss_a=loss_test)}
            )
            return output

    def test_step__multiple_dataloaders(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        test_acc = test_acc.type_as(x)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc})
            return output
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            output = OrderedDict(
                {"test_loss": loss_test, "test_acc": test_acc, "test_dic": dict(test_loss_a=loss_test)}
            )
            return output
        if batch_idx % 5 == 0:
            output = OrderedDict({f"test_loss_{dataloader_idx}": loss_test, f"test_acc_{dataloader_idx}": test_acc})
            return output
