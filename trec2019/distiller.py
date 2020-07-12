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
from trec2019.model import SparseNetModel, WTAModel
from trec2019.task import ClassificationTask, RankingTask
from trec2019.utils.losses import SupConLoss, TripletLoss

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
_root_dir = str(Path(__file__).parent.absolute())


class Distiller(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # dataset
        self._init_datasets()

        # layers
        self._init_layers()

    def _init_layers(self):
        self._init_sparse_layer()
        self._init_task_layer()
        self._init_recover_layer()

    def _get_sparse_cls(self):
        name = self.hparams.model.name
        if name == "sparsenet":
            return SparseNetModel
        elif name == "wta":
            return WTAModel
        elif name == "kate":
            raise NotImplementedError()
        else:
            raise ValueError("Unknown sparse model")

    def _init_sparse_layer(self):
        sparse_cls = self._get_sparse_cls()
        self.sparse = sparse_cls(self.hparams)

    def _init_task_layer(self):
        if self.hparams.loss.use_task_loss:
            dim_in = self.sparse.output_size
            feat_dim = 128
            self.task = nn.Sequential(
                nn.Linear(dim_in, feat_dim), nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            )
            self.loss_task = SupConLoss()
        else:
            self.task = None
            self.loss_task = None

    def _init_recover_layer(self):
        if self.hparams.loss.use_recovery_loss:
            input_size = self.sparse.output_size
            output_size = self.hparams.model.input_size
            self.recover = nn.Linear(input_size, output_size)
        else:
            self.recover = None

    # dataset
    def _init_dataset(self, dset_type):
        data_path = Path(self.hparams.dataset.path) / f"{dset_type}.zarr"
        data_cls = self.hparams.dataset.cls
        emb_path = self.hparams.dataset.emb_path
        on_memory = self.hparams.dataset.on_memory
        noise_ratio = self.hparams.noise.ratio

        data_cls = {
            "tr": TripleEmbeddingDataset,
            "emb": EmbeddingDataset,
            "emb-lbl": EmbeddingLabelDataset,
        }[data_cls]

        return data_cls(
            data_path, emb_path, noise_ratio=noise_ratio, on_memory=on_memory,
        )

    def _init_datasets(self):
        self._train_dataset = self._init_dataset("train")
        self._val_dataset = self._init_dataset("val")

    def _get_dataloader(self, dataset, shuffle=False):
        batch_size = self.hparams.train.batch_size if self.training else 2 ** 13
        num_workers = 4
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def _use_recovery_loss(self):
        return self.hparams.loss.use_recovery_loss

    def _use_task_loss(self):
        return self.hparams.loss.use_task_loss

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def loss(self, outputs):
        losses = {}
        losses["total"] = 0.0

        # task loss
        if self._use_task_loss():
            losses["task"] = self.loss_task(outputs["task"], outputs["target"])
            losses["total"] += losses["task"]

        # recover loss
        if self._use_recovery_loss():
            orig_data = outputs["data"][:, 0, :]
            recv_data_list = outputs["recover"].unbind(dim=1)
            losses["recover"] = torch.mean(
                torch.stack([F.mse_loss(rt, orig_data) for rt in recv_data_list], dim=0)
            )
            # ratio = self.hparams.loss.recovery_loss_ratio
            # losses["recover"] = (
            #     F.mse_loss(outputs["recover"], outputs["orig_data"]) * ratio
            # )
            losses["total"] += losses["recover"]

        return losses

    def forward_sparse(self, x):
        sparse_tensors = [self.sparse(t) for t in x.unbind(dim=1)]
        return torch.stack(sparse_tensors, dim=1)

    def forward_task(self, x):
        # return self.task(x)
        return F.normalize(self.task(x), dim=-1)

    @auto_move_data
    def forward(self, batch):
        # output features
        outputs = batch.copy()
        outputs["data"] = outputs["data"].permute(0, 2, 1)

        # normalize
        outputs["data"] = F.normalize(outputs["data"], dim=-1)

        # forward sparse
        outputs["sparse"] = self.forward_sparse(outputs["data"])

        # forward task (optional)
        if self._use_task_loss():
            outputs["task"] = self.forward_task(outputs["sparse"])

        # forward recover (optional)
        if self._use_recovery_loss():
            outputs["recover"] = self.recover(outputs[f"sparse"])

        return outputs

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging losses
        tqdm_dict = {
            "train_loss": losses["total"],
        }
        for aux_loss in ["task", "recover"]:
            if aux_loss in losses:
                tqdm_dict[f"train_loss_{aux_loss}"] = losses[aux_loss]

        return {
            "loss": tqdm_dict["train_loss"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step_end(self, outputs):
        # aggregate
        # outputs = {}
        # for k in batch_parts_outputs[0].keys():
        #     outputs[k] = torch.cat([part[k] for part in batch_parts_outputs], dim=1)

        # loss
        losses = self.loss(outputs)

        # logging
        # tqdm_dict = {
        #     "val_loss": losses["total"],
        #     "val_loss_task": losses["task"],
        #     "val_loss_recover": losses["recover"],
        # }
        tqdm_dict = {
            "val_loss": losses["total"],
        }
        for aux_loss in ["task", "recover"]:
            if aux_loss in losses:
                tqdm_dict[f"val_loss_{aux_loss}"] = losses[aux_loss]
        return {
            **tqdm_dict,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_epoch_end(self, outputs):
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

    # sparsity boosting weight adjustment, etc.
    def on_epoch_end(self):
        self.sparse.on_epoch_end()

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.train.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    # def test_dataloader(self):
    #     return self._get_dataloader(self._test_dataset)
