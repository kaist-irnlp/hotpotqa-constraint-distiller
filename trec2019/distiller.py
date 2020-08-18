"""
This file defines the core research contribution
"""
import os
import sys
import gc
from argparse import ArgumentParser
import logging
from multiprocessing import cpu_count
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch import optim
import torch_optimizer
from torch import nn
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from test_tube import HyperOptArgumentParser
from omegaconf import ListConfig
from omegaconf import OmegaConf

import zarr
import numpy as np
import pandas as pd
from pprint import pprint

# project specific
from trec2019.utils.dataset import *
from trec2019.utils.noise import *
from trec2019.model import WTAModel
from trec2019.model.wta.nupic import *
from trec2019.task import ClassificationTask, RankingTask
from trec2019.utils.losses import SupConLoss, TripletLoss

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
        self.encoder = WTAModel(self.hparams)

        # discriminator
        ## define
        in_dim = self.encoder.output_size * 2  # encoder output will be concatenated.
        h_dims = self.hparams.discriminator.hidden
        out_dim = self.hparams.discriminator.out
        weight_sparsity = self.hparams.discriminator.weight_sparsity
        ## build
        self.out = nn.Sequential()
        for i in range(len(h_dims)):
            linear = nn.Linear(in_dim, h_dims[i])
            # use sparse weights
            if 0 < weight_sparsity < 1:
                linear = SparseWeights(linear, sparsity=weight_sparsity)
                linear.apply(normalize_sparse_weights)
            # add modules
            self.out.add_module(f"disc_linear_{i+1}", linear)
            self.layers.add_module(
                f"disc_bn_{i+1}", nn.BatchNorm1d(h_dims[i], affine=False)
            )
            self.out.add_module(f"disc_selu_{i+1}", nn.SELU())
            # prepare next connection
            in_dim = h_dims[i]
        ## add out layer
        self.out.add_module(f"disc_out_linear", nn.Linear(in_dim, out_dim))
        self.out.add_module(f"disc_out_selu", nn.SELU())

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
        num_workers = int(cpu_count() / 3)
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
        margin = 1.0
        return max(sim_n - sim_p + margin, 0)

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

        # L2: discriminator loss
        losses["disc"] = self.loss_disc(outputs)

        # L1 + L2
        losses["total"] = losses["rank"] + losses["disc"]

        return losses

    def forward_encoder(self, data):
        # return self.encoder(data)
        return F.normalize(self.encoder(data), dim=-1)

    def forward_out(self, q, d):
        # return self.task(data)
        # return F.normalize(self.task(data), dim=-1)
        t_max = F.normalize(torch.max(q, d), dim=-1)
        t_dot = F.normalize(q * d, dim=-1)
        if self.hparams.discriminator.use_binary:
            t_max = t_max > 0
            t_dot = t_dot > 0
        t = torch.cat([t_max, t_dot])
        return self.out(t)

    @auto_move_data
    def forward(self, batch):
        # output features
        outputs = batch.clone()

        # forward encoder
        outputs["enc_query"] = self.forward_encoder(outputs["query"])
        outputs["enc_pos"] = self.forward_encoder(outputs["pos"])
        outputs["enc_neg"] = self.forward_encoder(outputs["neg"])

        # forward discriminator
        outputs["out_pos"] = self.forward_out(outputs["enc_query"], outputs["enc_pos"])
        outputs["out_neg"] = self.forward_out(outputs["enc_query"], outputs["enc_neg"])

        return outputs

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging losses
        tqdm_dict = {
            "loss": losses["total"],
            "loss_rank": losses["rank"],
            "loss_disc": losses["disc"],
        }

        return {
            "loss": tqdm_dict["loss"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "val_loss": losses["total"],
            "val_loss_rank": losses["rank"],
            "val_loss_disc": losses["disc"],
        }

        return {
            **tqdm_dict,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_epoch_end(self, outputs):
        # network states
        self._log_network_states()

        # losses
        tqdm_dict = {}
        for k in outputs[0].keys():
            if "loss" in k:
                tqdm_dict[f"avg_{k}"] = torch.stack([out[k] for out in outputs]).mean()

        results = {
            "val_loss": tqdm_dict["avg_val_loss"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

        return results

    def test_step(self, batch, batch_idx):
        return self.forward(batch)

    def test_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "test_loss": losses["total"],
            "test_loss_rank": losses["rank"],
            "test_loss_disc": losses["disc"],
        }

        return {
            **tqdm_dict,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def test_epoch_end(self, outputs):
        # network states
        self._log_network_states()

        # losses
        tqdm_dict = {}
        for k in outputs[0].keys():
            if "loss" in k:
                tqdm_dict[f"avg_{k}"] = torch.stack([out[k] for out in outputs]).mean()

        results = {
            "avg_test_loss": tqdm_dict["avg_test_loss"],
            "log": tqdm_dict,
        }

        return results

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
        self.sparse.on_epoch_end()

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.train.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

