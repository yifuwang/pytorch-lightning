import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything


class Model(pl.LightningModule):

    def __init__(self, channel=3):
        super(Model, self).__init__()
        self.automatic_optimization = False
        self.nn = nn.Sequential(*[nn.Conv2d(channel, channel, 1) for _ in range(32)])

    def forward(self, input):
        z = self.nn(input)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=1e-4)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        z = self(batch[0])
        loss = torch.nn.functional.mse_loss(z, batch[0])
        opt = self.optimizers()
        self.manual_backward(loss, opt)
        opt.step()
        opt.zero_grad()
        print("after zero grad")


if __name__ == '__main__':
    train_ds = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=32,
        num_workers=8,
        drop_last=True)

    trainer_kwargs = {
        'gpus': 2,
        'accelerator': 'dp',
    }

    seed_everything(42)
    model = Model()
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dl)
