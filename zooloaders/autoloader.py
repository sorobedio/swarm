import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# from zoodatasets.basedatasets import ZooDataset
from zoodatasets.chunkdatasets import ZooDataset
# from zoodatasets.layerdatasets import ZooDataset

class ZooDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, scale, topk, normalize):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.topk = topk
        self.normalize = normalize
        self.scale=scale

    def prepare_data(self):
        pass
        # datasets.CIFAR10(self.data_root, train=True, download=True)
        # datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(root=self.data_dir, split='train', scale=self.scale, topk=self.topk)
            self.valset = ZooDataset(root=self.data_dir, split='val',scale=self.scale)

        if stage == "test":
            self.testset = ZooDataset(root=self.data_dir, split='test', scale=self.scale)

        if stage == "predict":
            pass
            # pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        pass
        # return DataLoader(
        #     self.cifar10_predict,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )

