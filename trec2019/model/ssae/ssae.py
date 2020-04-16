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
        ## add noise
        self.encoder.add_module("noise", GaussianNoise())
        ## add encoder modules
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
            enc_weight = encoder_weights[-(i + 1)]
            fan_in = enc_weight.shape[1]
            fan_out = enc_weight.shape[0] if (i == (len(n) - 1)) else input_size
            linear = nn.Linear(fan_in, fan_out)
            linear.weight.data = enc_weight.transpose(0, 1)
            self.encoder.add_module(f"dec_linear_{i}", linear)
            self.encoder.add_module(f"dec_relu_{i}", nn.ReLU())

        # out
        self.out = nn.Sequential()

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        sparse_x = self.encoder(x)
        recover_x = self.decoder(sparse_x)
        out_x = self.out(sparse_x)

        return sparse_x, recover_x, out_x

    def training_step(self, batch, batch_idx):
        text, target = batch["data"], batch["target"]

        # forward
        sparse_x, recover_x, out_x = self.forward(text)

        return sparse_x, recover_x, out_x, target

    def training_step_end(self, outputs):
        # aggregate (for dp or ddp mode)
        sparse_x, recover_x, out_x, target = outputs

        # loss
        loss_total, loss_task, loss_recovery = self.loss(
            sparse_x, recover_x, out_x, target
        )

        # logging
        tqdm_dict = {
            "train_loss": loss_total,
            "loss_task": loss_task,
            "loss_recovery": loss_recovery,
        }
        log_dict = {
            "train_losses": tqdm_dict,
        }
        return {"loss": loss_total, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        text, target = batch["data"], batch["target"]

        # forward
        sparse_x, recover_x, out_x = self.forward(text)

        return sparse_x, recover_x, out_x, target

    def validation_step_end(self, outputs):
        # aggregate (for dp or ddp mode)
        sparse_x, recover_x, out_x, target = outputs

        # loss
        loss_total, loss_task, loss_recovery = self.loss(
            sparse_x, recover_x, out_x, target
        )

        # logging
        tqdm_dict = {
            "val_loss": loss_total,
            "loss_task": loss_task,
            "loss_recovery": loss_recovery,
        }
        log_dict = {
            "val_losses": tqdm_dict,
        }
        return {"val_loss": loss_total, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([out["val_loss"] for out in outputs]).mean()

        tqdm_dict = {"val_loss": avg_val_loss}

        results = {
            "val_loss": avg_val_loss,
            "progress_bar": tqdm_dict,
            "log": {"val_loss": avg_val_loss},
        }

        return results

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch_optimizer.RAdam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]
        # return optimizer

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
