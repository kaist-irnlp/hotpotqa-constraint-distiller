import os
import sys
import gc
import gzip
import json
import logging
import zarr
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import cpu_count

import torch
from torch import optim
import torch_optimizer
from torch import nn
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, SubwordVocab
import pytorch_lightning as pl

from trec2019.utils.dataset import *
from trec2019.utils.dense import *
from trec2019.utils.noise import *
from trec2019.model.module.topk import *


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
_root_dir = str(Path(__file__).parent.absolute())


class SSAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._dset_cls = News20EmbeddingDataset

        # layers
        self._init_layers()
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)

    def _init_layers(self):
        input_size = self.hparams.input_size
        n = self.hparams.n
        k = self.hparams.k
        dropout = self.hparams.dropout

        # encoder
        encoder_weights = []
        self.encoder = nn.Sequential()
        self.encoder.add_module("noise", GaussianNoise())
        for i in range(len(n)):
            fan_in = input_size if (i == 0) else n[i - 1]
            fan_out = n[i]
            linear = nn.Linear(fan_in, fan_out)
            encoder_weights.append(linear.weight)
            self.encoder.add_module(f"enc_linear_{i}", linear)
            self.encoder.add_module(
                f"enc_batch_norm_{i}", nn.BatchNorm1d(n[i], affine=False)
            )
            self.encoder.add_module(f"enc_dropout_{i}", nn.Dropout(dropout))
            self.encoder.add_module(f"enc_relu_{i}", nn.ReLU())
            self.encoder.add_module(f"enc_topk_{i}", BatchTopK(k[i]))

        # decoder
        self.decoder = nn.Sequential()
        for i in range(len(n)):
            enc_linear = encoder_weights[-(i + 1)]
            fan_out, fan_in = enc_linear.shape
            self.encoder.add_module(f"dec_linear_{i}", nn.Linear(fan_in, fan_out))
            self.encoder.add_module(f"dec_relu_{i}", nn.ReLU())

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_step_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_step_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch_optimizer.RAdam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # return [optimizer], [scheduler]
        return optimizer

    def _init_dataset(self):
        data_dir = Path(self.hparams.data_dir)
        self._train_dataset = self._dset_cls(str(data_dir / "train.zarr"))
        self._val_dataset = self._dset_cls(str(data_dir / "val.zarr"))
        self._test_dataset = self._dset_cls(str(data_dir / "test.zarr"))

    def _get_dataloader(self, dataset, test=False):
        # dist_sampler = DistributedSampler(dataset) if self.use_ddp else None
        batch_size = self.hparams.batch_size if not test else (2 ** 13)
        num_workers = int(cpu_count() / 2) or 1
        # num_workers = 0
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset, test=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # add more arguments
        parser.add_argument("--input_size", type=int, required=True)
        parser.add_argument("--dropout", default=0.2, type=float)

        return parser
