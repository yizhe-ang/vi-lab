from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST, SVHN


class MNIST_SVHN(Dataset):
    def __init__(self, data_dir, train=True, download=False, transform=None):
        super().__init__()

        self.mnist = MNIST(data_dir, train=train, download=download, transform=transform)
        split = "train" if train else "test"
        self.svhn = SVHN(data_dir, split=split, download=download, transform=transform)

        self.indices_mnist = torch.load(Path(data_dir) / f"{split}-ms-mnist-idx.pt")
        self.indices_svhn = torch.load(Path(data_dir) / f"{split}-ms-svhn-idx.pt")

        assert len(self.indices_mnist) == len(
            self.indices_svhn
        ), "Expected indices to be same size but are {:d} and {:d}".format(
            len(self.indices_mnist), len(self.indices_svhn)
        )

    def __len__(self):
        return len(self.indices_mnist)

    def __getitem__(self, idx):
        idx1, idx2 = self.indices_mnist[idx], self.indices_svhn[idx]
        assert self.mnist[idx1][1] == self.svhn[idx2][1], "Something evil has happened!"
        return self.mnist[idx1][0], self.svhn[idx2][0], self.svhn[idx2][1]


class MNIST_SVHNDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: int = 5000,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.transform = transform_lib.ToTensor()

    def prepare_data(self):
        # Download
        MNIST_SVHN(self.data_dir, train=True, download=True)
        MNIST_SVHN(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = MNIST_SVHN(
                self.data_dir, train=True, transform=self.transform
            )
            self.val_set = MNIST_SVHN(
                self.data_dir, train=False, transform=self.transform
            )

            # Infer dimension of dataset
            self.dims = (
                tuple(self.train_set[0][0].shape),
                tuple(self.train_set[0][1].shape),
            )

        if stage == "test" or stage is None:
            # FIXME No separate val / test dataset
            self.test_set = self.val_set

            # Infer dimension of dataset
            self.dims = (
                tuple(self.test_set[0][0].shape),
                tuple(self.test_set[0][1].shape),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
