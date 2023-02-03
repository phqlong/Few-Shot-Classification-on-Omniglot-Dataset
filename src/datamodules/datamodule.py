import hydra
import omegaconf
from omegaconf import DictConfig

import random
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from torchmeta.datasets import Omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from torchvision.transforms import transforms
from torchmeta.transforms import ClassSplitter, Categorical, Rotation

from src.datamodules.components import CombinationRandomSampler

class MetaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        kshot: int = 5,
        nway: int = 5,
        batch_size: int = 64,
        num_batches: DictConfig = DictConfig({"train": 10000, "val": 1000, "test": 1000}),
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.Resize(size=28),
             transforms.ToTensor(), 
             transforms.Normalize((0.1307,), (0.3081,))]
        )

        # target transformations
        self.target_transform = Categorical()

        # class augmentations
        self.class_augmentations = [] #[Rotation(angle=a) for a in [90, 180, 270]]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return self.hparams.nway

    def prepare_data(self):
        Omniglot(root=self.hparams.data_dir, num_classes_per_task=self.hparams.nway, meta_split="train", download=True)
        Omniglot(root=self.hparams.data_dir, num_classes_per_task=self.hparams.nway, meta_split="val", download=True)
        Omniglot(root=self.hparams.data_dir, num_classes_per_task=self.hparams.nway, meta_split="test", download=True)

    def setup_dataset(self, meta_split="train"):
        dataset = Omniglot(
            root=self.hparams.data_dir,
            num_classes_per_task=self.hparams.nway,
            transform=self.transforms,
            target_transform=self.target_transform,
            class_augmentations=self.class_augmentations,
            meta_split=meta_split, 
            download=False)
        dataset.seed(42)
        dataset = ClassSplitter(
            dataset,
            shuffle=True,
            random_state_seed=42,
            num_support_per_class=self.hparams.kshot,
            num_query_per_class=self.hparams.kshot)
        return dataset

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if stage is None or stage == "fit":
            self.data_train = self.setup_dataset("train")
            # self.data_train = Subset(data_train, random.sample(
            #     population=range(len(data_train)), 
            #     k=self.hparams.num_batches.train))

            self.data_val = self.setup_dataset("val")
            # self.data_val = Subset(data_val, random.sample(
            #     population=range(len(data_val)),
            #     k=self.hparams.num_batches.val))
        if stage == "test":
            self.data_test = self.setup_dataset("test")
            # self.data_test = Subset(data_test, random.sample(
            #     population=range(len(data_test)),
            #     k=self.hparams.num_batches.test))

    def train_dataloader(self):
        return BatchMetaDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=CombinationRandomSampler(self.data_train, num_samples=self.hparams.num_batches.train),
            shuffle=False,
        )

    def val_dataloader(self):
        return BatchMetaDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=CombinationRandomSampler(self.data_val, num_samples=self.hparams.num_batches.val),
            shuffle=False,
        )

    def test_dataloader(self):
        return BatchMetaDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=CombinationRandomSampler(
                self.data_test, num_samples=self.hparams.num_batches.test),
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "default.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
