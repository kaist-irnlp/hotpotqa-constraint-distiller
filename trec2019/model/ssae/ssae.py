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
from trec2019.model.ssae.module.topk import *


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
        self._noise_cls = GaussianNoise

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
        self.encoder.add_module("noise", self._noise_cls())
        ## add encoder modules
        for i in range(len(n)):
            fan_in = input_size if (i == 0) else n[i - 1]
            fan_out = n[i]
            linear = nn.Linear(fan_in, fan_out)
            self.encoder.add_module(f"enc_linear_{i}", linear)
            self.encoder.add_module(
                f"enc_batch_norm_{i}", nn.BatchNorm1d(n[i], affine=False)
            )
            if dropout:
                self.encoder.add_module(f"enc_dropout_{i}", nn.Dropout(dropout))
            self.encoder.add_module(f"enc_activation_{i}", nn.ReLU())
            self.encoder.add_module(f"enc_topk_{i}", BatchTopK(k[i]))
            ## for weight sharing
            ## (https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc)
            encoder_weights.append(linear.weight)

        # decoder
        self.decoder = nn.Sequential()
        for i in range(len(n)):
            enc_weight = encoder_weights[-(i + 1)]
            fan_in, fan_out = enc_weight.shape
            linear = nn.Linear(fan_in, fan_out)
            linear.weight.data = enc_weight.transpose(0, 1)  # weight sharing
            self.decoder.add_module(f"dec_linear_{i}", linear)
            # self.decoder.add_module(f"dec_activation_{i}", nn.ELU())

        # # out
        fan_in, fan_out = (
            # input_size,
            n[-1],
            self.hparams.output_size,
        )  # (input_size, output_size)
        self.out = nn.Linear(fan_in, fan_out)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        sparse_x = self.encoder(x)
        recover_x = self.decoder(sparse_x)
        out_x = self.out(sparse_x)

        # return sparse_x, recover_x, out_x
        return {"sparse": sparse_x, "recover": recover_x, "out": out_x}

    def loss_recovery(self, input, target):
        return F.mse_loss(input, target)
        # return F.l1_loss(input, target)

    def loss_classify(self, input, target):
        # input.shape() == (minibatch, C)
        return F.cross_entropy(input, target)

    def loss(self, outputs):
        # extract features
        x, target, sparse_x, recover_x, out_x = (
            outputs["x"],
            outputs["target"],
            outputs["sparse"],
            outputs["recover"],
            outputs["out"],
        )

        # task loss
        loss_task = self.loss_classify(out_x, target)

        # autoencoder loss (recovery)
        loss_recovery = self.loss_recovery(recover_x, x)

        # total loss
        loss_total = loss_recovery + loss_task

        return {
            "total": loss_total,
            "task": loss_task,
            "recovery": loss_recovery,
        }

    def training_step(self, batch, batch_idx):
        x, target = batch["data"], batch["target"]

        # forward
        features = self.forward(x)

        return {"x": x, "target": target, **features}

    def training_step_end(self, outputs):
        # aggregated loss (for dp or ddp mode)
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "train_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recovery": losses["recovery"],
        }
        log_dict = {
            "train_losses": tqdm_dict,
        }
        return {"loss": losses["total"], "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        x, target = batch["data"], batch["target"]

        # forward
        features = self.forward(x)

        return {"x": x, "target": target, **features}

    def validation_step_end(self, outputs):
        # aggregate (for dp or ddp mode)
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "val_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recovery": losses["recovery"],
        }
        log_dict = {
            "val_losses": tqdm_dict,
        }
        return {"val_loss": losses["total"], "progress_bar": tqdm_dict, "log": log_dict}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([out["val_loss"] for out in outputs]).mean()

        tqdm_dict = {"val_loss": avg_val_loss}

        results = {
            "val_loss": avg_val_loss,
            "progress_bar": tqdm_dict,
            "log": {"val_loss": avg_val_loss},
        }

        return results

    def test_step(self, batch, batch_idx):
        x, target = batch["data"], batch["target"]

        # forward
        features = self.forward(x)

        return {"x": x, "target": target, **features}

    def test_step_end(self, outputs):
        # aggregate (for dp or ddp mode)
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "test_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recovery": losses["recovery"],
        }
        log_dict = {
            "test_losses": tqdm_dict,
        }
        return {
            "test_loss": losses["total"],
            "progress_bar": tqdm_dict,
            "log": log_dict,
        }

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([out["test_loss"] for out in outputs]).mean()

        tqdm_dict = {"test_loss": avg_test_loss}

        results = {
            "test_loss": avg_test_loss,
            "progress_bar": tqdm_dict,
            "log": {"test_loss": avg_test_loss},
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

    def prepare_data(self):
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
        parser.add_argument("--n", type=int, nargs="+", required=True)
        parser.add_argument("--k", type=float, nargs="+", required=True)
        parser.add_argument("--output_size", "-out", type=int, required=True)
        parser.add_argument("--dropout", default=None, type=float)

        return parser
