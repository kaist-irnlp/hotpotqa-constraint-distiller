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
from test_tube import HyperOptArgumentParser
from omegaconf import ListConfig
from omegaconf import OmegaConf

import zarr
import numpy as np
import pandas as pd

# project specific
from trec2019.utils.dataset import *
from trec2019.utils.noise import *
from trec2019.model import SparseNetModel, WTAModel
from trec2019.utils.dataset import *
from trec2019.task import ClassificationTask, RankingTask

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
_root_dir = str(Path(__file__).parent.absolute())


class Distiller(pl.LightningModule):
    def __init__(self, hparams, data_path=None):
        super().__init__()
        self.hparams = hparams

        # dataset
        data_path = data_path or self.hparams.dataset.path
        self._init_dataset(data_path)

        # layers
        self._init_layers()

    def _get_data_cls(self):
        tp = self.hparams.dataset.type
        if tp == "emb":
            return EmbeddingDataset
        elif tp == "emb-lbl":
            return EmbeddingLabelDataset
        else:
            raise ValueError("Unkonwn dataset")

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

    def _get_task_cls(self):
        tp = self.hparams.task.type
        if tp == "classify":
            return ClassificationTask
        elif tp == "ranking":
            return RankingTask
        elif tp is None:
            return None
        else:
            raise ValueError("Unknown task")

    # layers
    def _init_layers(self):
        # self._init_noise_layer()
        self.sparse_cls = self._get_sparse_cls()
        self.task_cls = self._get_task_cls()
        self._init_sparse_layer()
        self._init_task_layer()
        self._init_recover_layer()

    # def _init_noise_layer(self):
    #     self.noise = GaussianNoise()

    def _init_task_layer(self):
        if self.hparams.loss.use_task_loss:
            self.task = self.task_cls(self.hparams)
        else:
            self.task = None

    def _init_sparse_layer(self):
        self.sparse = self.sparse_cls(self.hparams)

    def _init_recover_layer(self):
        input_size = self.sparse.output_size
        output_size = self.hparams.model.input_size
        if self.hparams.loss.use_recovery_loss:
            self.recover = nn.Linear(input_size, output_size)
        else:
            self.recover = None

    # Losses
    def loss_recovery(self, input, target):
        loss = F.l1_loss(input, target)
        # if self.hparams.loss.use_cosine_loss:
        #     loss = (1 - F.cosine_similarity(input, target)).mean()
        # else:
        #     loss = F.l1_loss(input, target)
        return loss

    def loss(self, outputs):
        target = outputs["target"].long() if ("target" in outputs) else None

        # autoencoder loss * lambda
        if self.recover:
            loss_recovery = (
                self.loss_recovery(outputs["recover"], outputs["orig_x"])
                * self.hparams.loss.recovery_loss_ratio
            )
        else:
            loss_recovery = torch.zeros((1,)).type_as(outputs["sparse"])

        # task loss
        if self.task:
            loss_task = self.task.loss(outputs["task"], target)
        else:
            loss_task = torch.zeros((1,)).type_as(outputs["sparse"])

        return {
            "total": loss_task + loss_recovery,
            "task": loss_task,
            "recovery": loss_recovery,
        }

    def forward(self, x, target):
        # noise
        # noise_x = self.noise(x)

        # sparse
        sparse_x = self.sparse(x)

        # 1. recover
        if self.recover is not None:
            recover_x = self.recover(sparse_x)
        else:
            recover_x = torch.zeros_like(x)

        # 2. out
        if self.task is not None:
            task_x = self.task(sparse_x)
        else:
            task_x = torch.zeros_like(x)

        features = {
            "x": x,
            "sparse": sparse_x,
            "recover": recover_x,
            "task": task_x,
        }
        if target is not None:
            features["target"] = target
        return features

    def _forward_step(self, batch, batch_idx, is_eval=False):
        data, orig_data = batch["data"], batch["orig_data"]
        # missing_target_vals = -torch.ones((data.size()[0],)).type_as(data).long()
        target = batch.get("target", None)

        # forward
        features = self.forward(data, target)
        return {**features, "orig_x": orig_data}

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def training_step_end(self, outputs):
        # loss
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
        return {"loss": losses["total"], "progress_bar": tqdm_dict, "log": tqdm_dict}

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, is_eval=True)

    def validation_step_end(self, outputs):
        # loss
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
        return {
            "val_loss": losses["total"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([out["val_loss"] for out in outputs]).mean()

        # val_loss_mean = 0
        # for output in outputs:
        #     val_loss_mean += output["val_loss"]
        # val_loss_mean /= len(outputs)
        tqdm_dict = {"val_loss": avg_val_loss}

        results = {
            "val_loss": avg_val_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

        return results

    ###
    def test_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, is_eval=True)

    def test_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "test_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recovery": losses["recovery"],
        }
        return {
            "test_loss": losses["total"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([out["test_loss"] for out in outputs]).mean()

        tqdm_dict = {"test_loss": avg_test_loss}

        results = {
            "test_loss": avg_test_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

        return results

    # sparsity boosting weight adjustment, etc.
    def on_epoch_end(self):
        self.sparse.on_epoch_end()

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch_optimizer.RAdam(
            self.parameters(), lr=self.hparams.train.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    # dataset
    def _init_dataset(self, data_path):
        self.data_path = Path(data_path)
        self.data_cls = self._get_data_cls()
        arr_path = self.hparams.dataset.arr_path
        noise = self.hparams.noise.type
        noise_ratio = self.hparams.noise.ratio

        self._train_dataset = self.data_cls(
            str(self.data_path / "train.zarr"),
            arr_path,
            noise=noise,
            noise_ratio=noise_ratio,
        )
        self._val_dataset = self.data_cls(
            str(self.data_path / "val.zarr"),
            arr_path,
            noise=noise,
            noise_ratio=noise_ratio,
        )
        self._test_dataset = self.data_cls(
            str(self.data_path / "test.zarr"),
            arr_path,
            noise=noise,
            noise_ratio=noise_ratio,
        )

    def _get_dataloader(self, dataset, shuffle=False):
        batch_size = self.hparams.train.batch_size if self.training else 2 ** 13
        num_workers = int(cpu_count() / 2) or 1
        pin_memory = True
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset)
