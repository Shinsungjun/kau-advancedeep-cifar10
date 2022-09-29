import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger
import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from lightningmodel import LitModel
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
def main():
    transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    train_set, val_set = random_split(dataset, [45000, 5000])
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    wandb_logger = WandbLogger(project='advance_deep', name = '2022707002_SSJ_cifar10', job_type='train', entity="holyjoon")

    checkpoint_callback = ModelCheckpoint(monitor="acc/val", mode="max", save_last=True)

    model = LitModel((3, 32, 32), num_classes = 10)

    trainer = pl.Trainer(max_epochs=50,
                     progress_bar_refresh_rate=20, 
                     gpus=2, 
                     logger=wandb_logger,
                     checkpoint_callback=checkpoint_callback)

    trainer.fit(model, DataLoader(train_set, batch_size = batch_size), DataLoader(val_set, batch_size = batch_size))
    trainer.test(model, DataLoader(test_set))

if __name__ == '__main__':
    main()
