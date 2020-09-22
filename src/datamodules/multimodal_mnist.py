"""From https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/mnist_datamodule.py
"""
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class MultimodalMNIST(MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        return dict(data=[img, target])


class MultimodalMNISTDataModule(LightningDataModule):

    name = "mnist"

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        val_split: int = 10_000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """
        .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
            :width: 400
            :alt: MNIST
        Specs:
            - 10 classes (1 per digit)
            - Each image is (1 x 28 x 28)
        Standard MNIST, train, val, test splits and transforms
        Transforms::
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])
        Example::
            from pl_bolts.datamodules import MNISTDataModule
            dm = MNISTDataModule('.')
            model = LitModel()
            Trainer().fit(model, dm)
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        MultimodalMNIST(
            self.data_dir, train=True, download=True, transform=transform_lib.ToTensor()
        )
        MultimodalMNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transform_lib.ToTensor(),
        )

    def setup(self, stage=None):
        # Set dimensions of dataset
        self.dims = [(1, 28, 28), (1,)]

        # Set likelihood ratio for ELBO
        self.likelihood_weights = [1.0, 50.0]

        if stage == "fit" or stage is None:
            dataset = MultimodalMNIST(
                self.data_dir,
                train=True,
                download=False,
                transform=self._default_transforms(),
            )

            train_length = len(dataset)

            self.train_set, self.val_set = random_split(
                dataset,
                [train_length - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.test_set = MultimodalMNIST(
                self.data_dir,
                train=False,
                download=False,
                transform=self._default_transforms(),
            )

    def train_dataloader(self, transforms=None):
        """
        MNIST train set removes a subset to use for validation
        Args:
            transforms: custom transforms
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self, transforms=None):
        """
        MNIST val set uses a subset of the training set for validation
        Args:
            transforms: custom transforms
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self, transforms=None):
        """
        MNIST test set uses the test split
        Args:
            transforms: custom transforms
        """
        return DataLoader(
            self.test_set,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def _default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [
                    transform_lib.ToTensor(),
                    transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms
