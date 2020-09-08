from torchvision.datasets import MNIST, SVHN
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule


class MNIST_SVHN(Dataset):
    def __init__(self, dir, train=True, transform=None):
        super().__init__()
        # files = ['train-ms-mnist-idx.pt', 'train-ms-svhn-idx.pt',
        #          'test-ms-mnist-idx.pt', 'test-ms-svhn-idx.pt']
        # if not all([exists(join(dir, f)) for f in files]):
        #     dataset_utils.generate_indices_mnist_svhn(dir=dir)

        self.mnist = MNIST(dir, train=train, download=True, transform=transform)
        split = 'train' if train else 'test'
        self.svhn = SVHN(dir, split=split, download=True, transform=transform)

        self.indices_mnist = torch.load(join(dir, '{}-ms-mnist-idx.pt'.format(split)))
        self.indices_svhn = torch.load(join(dir, '{}-ms-svhn-idx.pt'.format(split)))

        assert len(self.indices_mnist) == len(self.indices_svhn),\
            'Expected indices to be same size but are {:d} and {:d}'.format(len(self.indices_mnist), len(self.indices_svhn))

    def __len__(self):
        return len(self.indices_mnist)

    def __getitem__(self, idx):
        idx1, idx2 = self.indices_mnist[idx], self.indices_svhn[idx]
        assert self.mnist[idx1][1] == self.svhn[idx2][1], 'Something evil has happened!'
        return self.mnist[idx1][0], self.svhn[idx2][0], self.svhn[idx2][1]


class MNIST_SVHNDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: int = 5000,
        num_workers: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

        self.transform =
