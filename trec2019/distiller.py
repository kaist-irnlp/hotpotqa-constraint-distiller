"""
This file defines the core research contribution
"""
import gc
import logging
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_optimizer
import zarr
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning.core.decorators import auto_move_data
from test_tube import HyperOptArgumentParser
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from trec2019.model import WTAModel
from trec2019.model.wta.nupic import *
from trec2019.task import ClassificationTask, RankingTask
from trec2019.utils.dataset import *
from trec2019.utils.losses import SupConLoss, TripletLoss
from trec2019.utils.noise import *

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class Distiller(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # dataset
        self._init_datasets()

        # layers
        self._init_layers()

    def _init_layers(self):
        # encoder
        self._enc = WTAModel(self.hparams)

        # discriminator
        ## define
        in_dim = self._enc.output_size * 2  # encoder output will be concatenated.
        h_dims = self.hparams.disc.hidden
        out_dim = self.hparams.disc.out
        weight_sparsity = self.hparams.disc.weight_sparsity
        ## build
        self._disc = nn.Sequential()
        for i in range(len(h_dims)):
            linear = nn.Linear(in_dim, h_dims[i])
            # use sparse weights
            if 0 < weight_sparsity < 1:
                linear = SparseWeights(linear, sparsity=weight_sparsity)
                linear.apply(normalize_sparse_weights)
            # add modules
            self._disc.add_module(f"disc_linear_{i+1}", linear)
            self._disc.add_module(
                f"disc_bn_{i+1}", nn.BatchNorm1d(h_dims[i], affine=False)
            )
            self._disc.add_module(f"disc_selu_{i+1}", nn.SELU())
            # prepare next connection
            in_dim = h_dims[i]
        ## add out layer
        self._disc.add_module(f"disc_out_linear", nn.Linear(in_dim, out_dim))
        self._disc.add_module(f"disc_out_bn", nn.BatchNorm1d(out_dim, affine=False))
        self._disc.add_module(f"disc_out_selu", nn.SELU())

    # dataset
    def _init_datasets(self):
        self._train_dataset = self._init_dataset("train")
        self._val_dataset = self._init_dataset("val")
        self._test_dataset = self._init_dataset("test")

    def _init_dataset(self, dset_type):
        data_path = Path(self.hparams.dataset.path) / f"{dset_type}.zarr"
        data_cls = self.hparams.dataset.cls
        emb_path = self.hparams.dataset.emb_path
        on_memory = self.hparams.dataset.on_memory

        data_cls = {
            "tr": TripleEmbeddingDataset,
            "emb": EmbeddingDataset,
            "emb-lbl": EmbeddingLabelDataset,
        }[data_cls]

        return data_cls(data_path, emb_path, on_memory=on_memory,)

    def _get_dataloader(self, dataset, shuffle=False):
        num_workers = int(cpu_count() / 2)
        batch_size = self.hparams.train.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset)

    def loss_rank(self, outputs):
        q, pos, neg = outputs["enc_query"], outputs["enc_pos"], outputs["enc_neg"]
        sim_p = F.cosine_similarity(q, pos)
        sim_n = F.cosine_similarity(q, neg)
        delta = torch.mean(sim_n - sim_p)
        margin = 1.0
        return max(delta + margin, 0)

    def loss_disc(self, outputs):
        out_pos, target_pos = outputs["out_pos"], outputs["target_pos"]
        out_neg, target_neg = outputs["out_neg"], outputs["target_neg"]
        loss_pos, loss_neg = (
            F.cross_entropy(out_pos, target_pos),
            F.cross_entropy(out_neg, target_neg),
        )
        return loss_pos + loss_neg

    def loss(self, outputs):
        losses = {}

        # L1: contrastive loss between pos/neg
        losses["rank"] = self.loss_rank(outputs)

        # L2: disc loss
        losses["disc"] = self.loss_disc(outputs)

        # L1 + L2
        losses["total"] = losses["rank"] + losses["disc"]

        return losses

    def encode(self, data):
        # return self._enc(data)
        return F.normalize(self._enc(data), dim=1)

    def disc(self, q, d):
        t_max = F.normalize(torch.max(q, d), dim=1)
        t_dot = F.normalize(q * d, dim=1)
        if self.hparams.disc.use_binary:
            t_max = (t_max > 0).type_as(q)
            t_dot = (t_dot > 0).type_as(q)
        t = torch.cat([t_max, t_dot], dim=1)
        # t = torch.cat([q, d], dim=1)
        return self._disc(t)

    @auto_move_data
    def forward(self, batch):
        # output features
        outputs = batch.copy()

        # forward encoder
        outputs["enc_query"] = self.encode(outputs["query"])
        outputs["enc_pos"] = self.encode(outputs["pos"])
        outputs["enc_neg"] = self.encode(outputs["neg"])

        # forward disc
        outputs["out_pos"] = self.disc(outputs["enc_query"], outputs["enc_pos"])
        outputs["out_neg"] = self.disc(outputs["enc_query"], outputs["enc_neg"])

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self.loss(outputs)
        # logging
        result = pl.TrainResult(minimize=losses["total"])
        result.log(
            "train_loss", losses["total"], prog_bar=True, logger=True, sync_dist=True
        )
        result.log(
            "train_loss_rank",
            losses["rank"],
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        result.log(
            "train_loss_disc",
            losses["disc"],
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self.loss(outputs)
        # logging
        result = pl.EvalResult(checkpoint_on=losses["total"],)
        result.log_dict(
            {
                "val_loss": losses["total"],
                "val_loss_rank": losses["rank"],
                "val_loss_disc": losses["disc"],
            }
        )
        # result.val_loss = losses["total"]

        return result

    # def validation_epoch_end(self, outputs):
    #     avg_loss = outputs.val_loss.mean()
    #     result = pl.EvalResult()
    #     result.log('val_loss', avg_loss)
    #     return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys(
            {
                "val_loss": "test_loss",
                "val_loss_rank": "test_loss_rank",
                "val_loss_disc": "test_loss_disc",
            }
        )
        return result

    def _log_kwinner(self, m):
        if isinstance(m, KWinnersBase):
            # entropy
            duty_cycles = m.duty_cycle.cpu()
            _, entropy = binary_entropy(duty_cycles)
            self.logger.experiment.log_metric(f"entropy", entropy)

            # duty cycle
            fig = plot_duty_cycle(duty_cycles)
            self.logger.experiment.log_image(f"duty_cycles", fig)
            plt.close(fig)

            # boost strength
            boost_strength = m._cached_boost_strength
            self.logger.experiment.log_metric(f"boost_strength", boost_strength)

            # boost factors
            # boost_factors = m.boost_factors
            # if boost_factors is not None:
            #     fig = plot_boost_factors(boost_factors)
            #     self.logger.experiment.log_image(f"boost_factors", fig)
            #     plt.close(fig)

    def _log_network_states(self):
        # duty cycles & entropy
        self.apply(self._log_kwinner)

    # sparsity boosting weight adjustment, etc.
    def on_epoch_end(self):
        self._enc.on_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.train.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer
        # return [optimizer], [scheduler]
