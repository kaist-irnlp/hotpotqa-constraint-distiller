"""
This file defines the core research contribution   
"""
import os
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl

import logging
from multiprocessing import cpu_count
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trec2019.utils.dataset import TRECTripleBERTDataset
from trec2019.utils.encoder import BertEncoder

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class SparseNet(pl.LightningModule):
    def __init__(self, hparams):
        super(SparseNet, self).__init__()
        self.hparams = hparams
        self._load_dataset()

    def _load_dataset(self):
        data_dir = Path(self.hparams.data_dir)
        self._train_dataset = TRECTripleBERTDataset(data_dir / "train.parquet")
        self._val_dataset = TRECTripleBERTDataset(data_dir / "valid.parquet")
        self._test_dataset = TRECTripleBERTDataset(data_dir / "test.parquet")

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def _get_dataloader(self, dataset, test=False):
        dist_sampler = DistributedSampler(dataset) if self.use_ddp else None
        batch_size = self.hparams.batch_size if not test else 100000
        num_workers = int(cpu_count() / 2)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    @pl.data_loader
    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset)

    @pl.data_loader
    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    @pl.data_loader
    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset, test=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--learning_rate", default=0.02, type=float)
        parser.add_argument("--batch_size", default=32, type=int)

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=2, type=int)

        return parser

