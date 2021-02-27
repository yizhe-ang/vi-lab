import json
from pathlib import Path

import pandas as pd
import PIL.Image as Image
import torch
from pytorch_lightning import LightningDataModule
from src.datamodules.utils import one_hot_encode
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transform_celeba(crop_size_img=148, img_size=64):
    # load training data
    offset_height = (218 - crop_size_img) // 2
    offset_width = (178 - crop_size_img) // 2
    crop = lambda x: x[
        :,
        offset_height : offset_height + crop_size_img,
        offset_width : offset_width + crop_size_img,
    ]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize(size=(img_size, img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    return transform


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images

    Train size: 162_770
    Val size: 19_867
    """

    def __init__(
        self,
        data_dir,
        partition=0,
        len_sequence=256,
        random_text_ordering=False,
        random_text_startindex=True,
        paired_prop=1.0,
    ):
        self.len_sequence = len_sequence

        dir_dataset_base = Path(data_dir) / "CelebA"

        filename_text = dir_dataset_base / (
            "list_attr_text_"
            + str(len_sequence).zfill(3)
            + "_"
            + str(random_text_ordering)
            + "_"
            + str(random_text_startindex)
            + "_celeba.csv"
        )
        filename_partition = dir_dataset_base / "list_eval_partition.csv"
        filename_attributes = dir_dataset_base / "list_attr_celeba.csv"

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition)
        df_attributes = pd.read_csv(filename_attributes)

        self.img_dir = dir_dataset_base / "img_align_celeba"
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes
        self.partition_path = filename_partition

        alphabet_path = dir_dataset_base / "alphabet.json"
        with open(alphabet_path) as alphabet_file:
            self.alphabet = str("".join(json.load(alphabet_file)))

        self.img_names = df_text.loc[df_partition["partition"] == partition][
            "image_id"
        ].values
        self.attributes = df_attributes.loc[df_partition["partition"] == partition]
        self.labels = df_attributes.loc[df_partition["partition"] == partition].values
        # atm, i am just using blond_hair as labels
        self.y = df_text.loc[df_partition["partition"] == partition]["text"].values

        self.transform = get_transform_celeba()

        self.dataset_len = self.y.shape[0]

        # Create boolean tensor of data points that are to be paired
        self.paired = torch.rand(self.dataset_len) <= paired_prop

    def __getitem__(self, index):
        img = Image.open(self.img_dir / self.img_names[index])

        if self.transform is not None:
            img = self.transform(img)
        text_str = one_hot_encode(self.len_sequence, self.alphabet, self.y[index])
        label = torch.from_numpy((self.labels[index, 1:] > 0).astype(int)).float()

        # Whether this data point is to be paired
        paired = self.paired[index]

        # img: [B, 3, 64, 64]
        # text_str: [B, len_sequence, len(alphabet)]
        # label: [B, n_attributes: 40]
        return {"data": [img, text_str], "label": label, "paired": paired}

    def __len__(self):
        return self.dataset_len

    def get_text_str(self, index):
        return self.y[index]


class CelebaDataModule(LightningDataModule):
    """Paired CelebA - Text multimodal dataset.

    Train size: 162_770
    Val size: 19_867
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        # val_split: int = 50_000,
        num_workers: int = 16,
        seed: int = 42,
        paired_prop=1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.paired_prop = paired_prop

        # FIXME Hardcode
        self.dims = [(3, 64, 64), (256, 71)]
        # img_size * img_size / len_sequence
        self.likelihood_weights = (1.0, 64 * 64 / 256)
        self.n_classes = 40

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = CelebaDataset(
                self.data_dir,
                partition=0,
                paired_prop=self.paired_prop,
            )
            self.val_set = CelebaDataset(
                self.data_dir,
                partition=1,
            )

        if stage == "test" or stage is None:
            self.test_set = self.val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
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
