"""
This file defines the core research contribution   
"""
import os
import torch
from torch import optim
from torch import nn
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
from trec2019.sparse.sparsenet.helper import *

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class SparseNet(pl.LightningModule):
    def __init__(self, hparams):
        super(SparseNet, self).__init__()
        self.hparams = hparams
        self._encoded = None
        self._load_dataset()
        # network
        self._validate_network_params()
        self._init_network()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_sdr(x)
        x = self.fc(x)

        if self.training:
            batch_size = x.shape[0]
            self.learning_iterations += batch_size

        return x

    def _init_network(self):
        self.learning_iterations = 0
        self.flatten = Flatten()

        # Linear layers only (from original code)
        input_features = self.hparams.input_size
        output_size = self.hparams.output_size
        n = self.hparams.n
        k = self.hparams.k
        normalize_weights = self.hparams.normalize_weights
        weight_sparsity = self.hparams.weight_sparsity
        use_batch_norm = self.hparams.use_batch_norm
        dropout = self.hparams.dropout
        k_inference_factor = self.hparams.k_inference_factor
        boost_strength = self.hparams.boost_strength
        boost_strength_factor = self.hparams.boost_strength_factor

        self.linear_sdr = nn.Sequential()
        for i in range(len(n)):
            if n[i] != 0:
                linear = nn.Linear(input_features)
                if 0 < weight_sparsity[i] < 1:
                    linear = SparseWeights(linear, weightSparsity=weight_sparsity[i])
                    if normalize_weights:
                        linear.apply(normalizeSparseWeights)
                self.linear_sdr.add_module(f"linear_sdr{i+1}", linear)

                if use_batch_norm:
                    self.linear_sdr.add_module(
                        f"linear_sdr{i+1}_bn", nn.BatchNorm1d(n[i], affine=False)
                    )

                if dropout > 0.0:
                    self.linear_sdr.add_module(
                        f"linear_sdr{i+1}_dropout", nn.Dropout(dropout)
                    )

                if 0 < k[i] < n[i]:
                    kwinner = KWinners(
                        n=n[i],
                        k=k[i],
                        kInferenceFactor=k_inference_factor,
                        boostStrength=boost_strength,
                        boostStrengthFactor=boost_strength_factor,
                    )
                    self.linear_sdr.add_module(f"linear_sdr{i+1}_kwinner", kwinner)
                else:
                    self.linear_sdr.add_module(f"linear_sdr{i+1}_relu", nn.ReLU())
                # Feed this layer output into next layer input
                input_features = n[i]

        # Add one fully connected layer after all hidden layers
        self.fc = nn.Linear(input_features, output_size)

    def on_epoch_end(self):
        self.apply(updateBoostStrength)
        self.apply(rezeroWeights)

    def get_encoded(self):
        return self._encoded

    def get_learning_iterations(self):
        return self.learning_iterations

    def maxEntropy(self):
        entropy = 0
        for module in self.modules():
            if module == self:
                continue
            if hasattr(module, "maxEntropy"):
                entropy += module.maxEntropy()
        return entropy

    def entropy(self):
        entropy = 0
        for module in self.modules():
            if module == self:
                continue
            if hasattr(module, "entropy"):
                entropy += module.entropy()
        return entropy

    def pruneWeights(self, minWeight):
        """
        Prune all the weights whose absolute magnitude is less than minWeight
        :param minWeight: min weight to prune. If zero then no pruning
        :type minWeight: float
        """
        if minWeight == 0.0:
            return

        # Collect all weights
        weights = [v for k, v in self.named_parameters() if "weight" in k]
        for w in weights:
            # Filter weights above threshold
            mask = torch.ge(torch.abs(w.data), minWeight)
            # Zero other weights
            w.data.mul_(mask.type(torch.float32))

    def pruneDutycycles(self, threshold=0.0):
        """
        Prune all the units with dutycycles whose absolute magnitude is less than
        the given threshold
        :param threshold: min threshold to prune. If less than zero then no pruning
        :type threshold: float
        """
        if threshold < 0.0:
            return

        # Collect all layers with 'dutyCycle'
        for m in self.modules():
            if m == self:
                continue
            if hasattr(m, "pruneDutycycles"):
                m.pruneDutycycles(threshold)

    def _validate_network_params(self):
        hparams = self.hparams
        if type(hparams.n) is not list:
            hparams.n = [hparams.n]
        if type(hparams.k) is not list:
            hparams.k = [hparams.k] * len(hparams.n)
        assert len(hparams.n) == len(hparams.k)
        for i in range(len(hparams.n)):
            assert hparams.k[i] <= hparams.n[i]
        if type(hparams.weight_sparsity) is not list:
            hparams.weight_sparsity = [hparams.weight_sparsity] * len(hparams.n)
        assert len(hparams.n) == len(hparams.weight_sparsity)
        for i in range(len(hparams.weight_sparsity)):
            assert hparams.weight_sparsity[i] >= 0

    def _load_dataset(self):
        data_dir = Path(self.hparams.data_dir)
        self._train_dataset = TRECTripleBERTDataset(data_dir / "train.parquet")
        self._val_dataset = TRECTripleBERTDataset(data_dir / "valid.parquet")
        self._test_dataset = TRECTripleBERTDataset(data_dir / "test.parquet")

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

