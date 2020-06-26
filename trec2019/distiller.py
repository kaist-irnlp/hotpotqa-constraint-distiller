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
from trec2019.task import ClassificationTask, RankingTask
from trec2019.utils.losses import SupConLoss, TripletLoss

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
_root_dir = str(Path(__file__).parent.absolute())

# class SupConResNet(nn.Module):
#     """backbone + projection head"""
#     def __init__(self, name='resnet50', head='mlp', feat_dim=128):
#         super(SupConResNet, self).__init__()
#         model_fun, dim_in = model_dict[name]
#         self.encoder = model_fun()
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim)
#             )
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))

#     def forward(self, x):
#         feat = self.encoder(x)
#         feat = F.normalize(self.head(feat), dim=1)
#         return feat


class Distiller(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # dataset
        self._init_dataset()

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

    def _init_task_layer(self):
        if self.hparams.loss.use_task_loss:
            dim_in = self.sparse.output_size
            feat_dim = 128
            self.task = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
            self._loss_task = TripletLoss()
        else:
            self.task = None

    def _init_sparse_layer(self):
        sparse_cls = self._get_sparse_cls()
        self.sparse = sparse_cls(self.hparams)

    def _init_recover_layer(self):
        if self.hparams.loss.use_recovery_loss:
            input_size = self.sparse.output_size
            output_size = self.hparams.model.input_size
            self.recover = nn.Linear(input_size, output_size)
        else:
            self.recover = None

    # TODO: 구현 필요
    def loss_task(self, outputs, margin=0.0):
        q, pos, neg = outputs["task_q"], outputs["task_pos"], outputs["task_neg"]
        return self._loss_task(q, pos, neg)
        # sim_p = torch.sum(q * pos, axis=1)
        # sim_n = torch.sum(q * neg, axis=1)
        # delta = sim_n - sim_p
        # return torch.sum(F.relu(margin + delta))

    def loss_recover(self, outputs):
        loss = 0.0
        ratio = self.hparams.loss.recovery_loss_ratio
        for e in ["q", "pos", "neg"]:
            loss += F.mse_loss(outputs[f"orig_{e}"], outputs[f"recover_{e}"])
        return loss * ratio

    def loss(self, outputs):
        # recover loss
        if self.recover is not None:
            loss_recover = self.loss_recover(outputs)
        else:
            loss_recover = 0.0

        # task loss
        if self.task is not None:
            loss_task = self.loss_task(outputs)
        else:
            loss_task = 0.0

        return {
            "total": loss_task + loss_recover,
            "task": loss_task,
            "recover": loss_recover,
        }

    def forward_sparse(self, x):
        return F.normalize(self.sparse(x), dim=1)

    def forward_task(self, x):
        return F.normalize(self.task(x), dim=1)

    def forward(self, batch):
        # elements to train
        trainable = ["q", "pos", "neg"]

        # output features (start with orig_* data)
        features = {k: v for (k, v) in batch.items() if "orig_" in k}

        # forward sparse
        for e in trainable:
            features[f"sparse_{e}"] = self.forward_sparse(batch[e])

        # forward task
        if self.task is not None:
            for e in trainable:
                features[f"task_{e}"] = self.forward_task(features[f"sparse_{e}"])

        # forward recover
        if self.recover is not None:
            for e in trainable:
                features[f"recover_{e}"] = self.recover(features[f"sparse_{e}"])

        return features

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "train_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recover": losses["recover"],
        }
        log_dict = {
            "train_losses": tqdm_dict,
        }
        return {"loss": losses["total"], "progress_bar": tqdm_dict, "log": tqdm_dict}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([out["train_loss"] for out in outputs]).mean()

        # val_loss_mean = 0
        # for output in outputs:
        #     val_loss_mean += output["val_loss"]
        # val_loss_mean /= len(outputs)
        tqdm_dict = {"train_loss": avg_train_loss}

        results = {
            "train_loss": avg_train_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

        return results

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step_end(self, outputs):
        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "val_loss": losses["total"],
            "loss_task": losses["task"],
            "loss_recover": losses["recover"],
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
    # def test_step(self, batch, batch_idx):
    #     return self.forward(batch)

    # def test_step_end(self, outputs):
    #     # loss
    #     losses = self.loss(outputs)

    #     # logging
    #     tqdm_dict = {
    #         "test_loss": losses["total"],
    #         "loss_task": losses["task"],
    #         "loss_recover": losses["recover"],
    #     }
    #     return {
    #         "test_loss": losses["total"],
    #         "progress_bar": tqdm_dict,
    #         "log": tqdm_dict,
    #     }

    # def test_epoch_end(self, outputs):
    #     avg_test_loss = torch.stack([out["test_loss"] for out in outputs]).mean()

    #     tqdm_dict = {"test_loss": avg_test_loss}

    #     results = {
    #         "test_loss": avg_test_loss,
    #         "progress_bar": tqdm_dict,
    #         "log": tqdm_dict,
    #     }

    #     return results

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
    def _init_dataset(self):
        data_path = self.hparams.dataset.path
        emb_path = self.hparams.dataset.emb_path
        noise = self.hparams.noise.type
        noise_ratio = self.hparams.noise.ratio

        self._train_dataset = TripleEmbeddingDataset(
            data_path, emb_path, "train", noise=noise, noise_ratio=noise_ratio
        )
        self._val_dataset = TripleEmbeddingDataset(
            data_path, emb_path, "val", noise=noise, noise_ratio=noise_ratio,
        )
        # self._test_dataset = TripleEmbeddingDataset(
        #     data_path, emb_path, noise=noise, noise_ratio=noise_ratio,
        # )

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

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    # def test_dataloader(self):
    #     return self._get_dataloader(self._test_dataset)
