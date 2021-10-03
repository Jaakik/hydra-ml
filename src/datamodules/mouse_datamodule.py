from typing import Optional, Tuple
from .datasets.mouse_dataset import MouseDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class MouseDataModule(LightningDataModule):
    """
    LightningDataModule for Caltech Mouse Social Interactions (CalMS21) Dataset.

    this DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_train_dir: str = "/Users/marouanejaakik/Desktop/git-explore/hydra-ml/data/mouse/train/train_features.npy",
            data_test_dir: str = "/Users/marouanejaakik/Desktop/git-explore/hydra-ml/data/mouse/train/train_features.npy",
            ann_dir: str = "/Users/marouanejaakik/Desktop/git-explore/hydra-ml/data/mouse/annotation /train.npy",
            train_val_split: Tuple[int, int] = (55, 15),
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.ann_dir = ann_dir
        self.data_train_dir = data_train_dir
        self.data_test_dir = data_test_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 4

    def prepare_data(self):
        # add the script used to directly download the dataset from aicrowd platform
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_test = MouseDataset(self.data_test_dir)
        train_set = MouseDataset(self.data_train_dir, self.ann_dir)
        self.data_train, self.data_val = random_split(train_set, self.train_val_split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
