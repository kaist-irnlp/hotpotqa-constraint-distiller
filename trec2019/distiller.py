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
from trec2019.model.disc.disc import DiscModel

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
from pytorch_lightning.metrics import functional as FM

from trec2019.model import WTAModel, DiscModel
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

        # layers
        self._init_layers()

    def _use_disc(self):
        return self.hparams.disc.use_disc

    def _init_layers(self):
        # encoder
        self._enc = WTAModel(self.hparams)

        # discriminator
        if self._use_disc():
            self._disc = DiscModel(self.hparams)

    def loss_rank(self, outputs):
        q, pos, neg = outputs["enc_query"], outputs["enc_pos"], outputs["enc_neg"]
        sim_p = F.cosine_similarity(q, pos)
        sim_n = F.cosine_similarity(q, neg)
        delta = torch.mean(sim_n - sim_p)
        margin = self.hparams.model.rank_margin
        return torch.max(delta + margin, torch.zeros(1))

    def loss_disc(self, outputs):
        out_pos, target_pos = outputs["out_pos"], outputs["target_pos"]
        out_neg, target_neg = outputs["out_neg"], outputs["target_neg"]
        loss_pos, loss_neg = (
            F.cross_entropy(out_pos, target_pos),
            F.cross_entropy(out_neg, target_neg),
        )
        return (loss_pos + loss_neg) / 2

    def acc_disc(self, outputs):
        # TODO: acc = FM.accuracy(y_hat, y)
        # https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#lightningmodule-for-production
        out_pos, target_pos = outputs["out_pos"], outputs["target_pos"]
        out_neg, target_neg = outputs["out_neg"], outputs["target_neg"]
        num_classes = self.hparams.disc.out
        acc_pos, acc_neg = (
            FM.accuracy(out_pos.argmax(dim=-1), target_pos, num_classes=num_classes),
            FM.accuracy(out_neg.argmax(dim=-1), target_neg, num_classes=num_classes),
        )
        return (acc_pos + acc_neg) / 2

    @auto_move_data
    def loss(self, outputs):
        losses = {"total": torch.zeros(1, device=self.device)}

        # L1: contrastive loss between pos/neg
        losses["rank"] = self.loss_rank(outputs)
        losses["total"] += losses["rank"]

        if self._use_disc():
            # L2: disc loss
            losses["disc"] = self.loss_disc(outputs)
            # Acc
            losses["acc_disc"] = self.acc_disc(outputs)
            # add to total
            losses["total"] += losses["disc"]

        return losses

    def encode(self, data):
        return self._enc(data)
        # return F.normalize(self._enc(data), dim=1)

    def disc(self, q, d):
        # TODO: consider L2-distance?
        t_dot = q * d
        t_dist = F.pairwise_distance(q, d).unsqueeze(1)
        # if self.hparams.disc.use_maxpool:
        #     t_max = F.normalize(torch.max(q, d), dim=1)
        #     t = torch.cat([t_max, t_dot], dim=1)
        # else:
        #     t = t_dot
        t = torch.cat([t_dot, t_dist], dim=1)
        return self._disc(t)

    @auto_move_data
    def forward(self, batch):
        raise NotImplemented()

    @auto_move_data
    def shared_step(self, batch):
        # output features
        outputs = batch.copy()

        # forward encoder
        outputs["enc_query"] = self.encode(outputs["query"])
        outputs["enc_pos"] = self.encode(outputs["pos"])
        outputs["enc_neg"] = self.encode(outputs["neg"])

        # forward disc
        if self._use_disc():
            outputs["out_pos"] = self.disc(outputs["enc_query"], outputs["enc_pos"])
            outputs["out_neg"] = self.disc(outputs["enc_query"], outputs["enc_neg"])

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        losses = self.loss(outputs)
        # TODO: DEBUG
        # assert isinstance(losses["total"], torch.Tensor), f'{losses["total"]}, {type(losses["total"])} <- torch.Tensor'
        # logging
        result = pl.TrainResult(minimize=losses["total"])
        result.log_dict(
            {"train_loss": losses["total"], "train_loss_rank": losses["rank"],},
            on_step=True,
            on_epoch=False,
        )
        if self._use_disc():
            result.log_dict(
                {
                    "train_loss_disc": losses["disc"],
                    "train_acc_disc": losses["acc_disc"],
                },
                on_step=True,
                on_epoch=False,
            )
        return result

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        losses = self.loss(outputs)
        # for epoch_end
        result = pl.EvalResult(
            checkpoint_on=losses["total"], early_stop_on=losses["total"]
        )
        result.log_dict(
            {"val_loss": losses["total"], "val_loss_rank": losses["rank"],},
            on_step=False,
            on_epoch=True,
        )
        if self._use_disc():
            result.log_dict(
                {"val_loss_disc": losses["disc"], "val_acc_disc": losses["acc_disc"],},
                on_step=True,
                on_epoch=False,
            )
        return result

    # def validation_epoch_end(self, val_step_outputs):
    #     return val_step_outputs

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys(
            {
                "val_loss": "test_loss",
                "val_loss_rank": "test_loss_rank",
                "val_loss_disc": "test_loss_disc",
                "val_acc_disc": "test_acc_disc",
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
        self._log_network_states()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.train.learning_rate,)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # reduce every epoch (default)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            "monitor": "val_checkpoint_on",
        }
        return [optimizer], [scheduler]
