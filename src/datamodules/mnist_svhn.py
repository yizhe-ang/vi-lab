from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST, SVHN


class MNIST_SVHN(Dataset):
    def __init__(
        self,
        data_dir,
        train=True,
        download=False,
        transform=None,
        paired_prop=1.0,
        resize=False,
    ):
        super().__init__()
        self.resize = transform_lib.Resize(32) if resize else None

        self.mnist = MNIST(
            data_dir, train=train, download=download, transform=transform
        )
        split = "train" if train else "test"
        self.svhn = SVHN(data_dir, split=split, download=download, transform=transform)

        self.indices_mnist = torch.load(Path(data_dir) / f"{split}-ms-mnist-idx.pt")
        self.indices_svhn = torch.load(Path(data_dir) / f"{split}-ms-svhn-idx.pt")

        assert len(self.indices_mnist) == len(
            self.indices_svhn
        ), "Expected indices to be same size but are {:d} and {:d}".format(
            len(self.indices_mnist), len(self.indices_svhn)
        )

        self.dataset_len = len(self.indices_mnist)

        # Create boolean tensor of data points that are to be paired
        self.paired = torch.rand(self.dataset_len) <= paired_prop

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx1, idx2 = self.indices_mnist[idx], self.indices_svhn[idx]
        assert self.mnist[idx1][1] == self.svhn[idx2][1], "Something evil has happened!"

        mnist = self.mnist[idx1][0]
        if self.resize:
            mnist = self.resize(mnist)

        svhn = self.svhn[idx2][0]
        label = self.svhn[idx2][1]

        # Whether this data point is to be paired
        paired = self.paired[idx]

        return {"data": [mnist, svhn], "label": label, "paired": paired}


class MNIST_SVHN_DataModule(LightningDataModule):
    """Paired MNIST - SVHN multimodal dataset.

    Training set size: 1,682,040
    Test set size: 300,000
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: int = 50_000,
        num_workers: int = 16,
        seed: int = 42,
        paired_prop=1.0,
        resize=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.paired_prop = paired_prop
        self.resize = resize

        # Number of class labels for each modality
        self.n_classes = 10
        self.transform = transform_lib.ToTensor()

    def prepare_data(self):
        # Download
        MNIST_SVHN(self.data_dir, train=True, download=True, resize=self.resize)
        MNIST_SVHN(self.data_dir, train=False, download=True, resize=self.resize)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = MNIST_SVHN(
                self.data_dir,
                train=True,
                transform=self.transform,
                paired_prop=self.paired_prop,
                resize=self.resize,
            )
            self.train_set, self.val_set = random_split(
                dataset,
                [len(dataset) - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Infer dimension of dataset
            self.dims = [tuple(modality.shape) for modality in self.val_set[0]["data"]]

        if stage == "test" or stage is None:
            self.test_set = MNIST_SVHN(
                self.data_dir, train=False, transform=self.transform, resize=self.resize
            )

            # Infer dimension of dataset
            self.dims = [tuple(modality.shape) for modality in self.test_set[0]["data"]]

        self.likelihood_weights = (
            np.prod(self.dims[1]) / np.prod(self.dims[0]),
            1.0,
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
