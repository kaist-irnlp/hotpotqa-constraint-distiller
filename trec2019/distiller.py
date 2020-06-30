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

    def loss_task(self, outputs, margin=0.0):
        q, pos, neg = outputs["task_q"], outputs["task_pos"], outputs["task_neg"]
        return self._loss_task(q, pos, neg, size_average=False)
        # sim_p = torch.sum(q * pos, axis=1)
        # sim_n = torch.sum(q * neg, axis=1)
        # delta = sim_n - sim_p
        # return torch.sum(F.relu(margin + delta))

    def loss_recover(self, outputs):
        loss = 0.0
        ratio = self.hparams.loss.recovery_loss_ratio
        fields = ["q", "pos", "neg"]
        for e in fields:
            loss += F.mse_loss(outputs[f"orig_{e}"], outputs[f"recover_{e}"])
        return (loss / len(fields)) * ratio

    def loss(self, outputs):
        # recover loss
        if self.recover is not None:
            loss_recover = self.loss_recover(outputs)
        else:
            loss_recover = torch.zeros(1)

        # task loss
        if self.task is not None:
            loss_task = self.loss_task(outputs)
        else:
            loss_task = torch.zeros(1)

        return {
            "total": loss_task + loss_recover,
            "task": loss_task,
            "recover": loss_recover,
        }

    def forward_sparse(self, x):
        return F.normalize(self.sparse(x), dim=1)

    def forward_task(self, x):
        return F.normalize(self.task(x), dim=1)

    @auto_move_data
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
        # aggregate
        # outputs = {}
        # for k in batch_parts_outputs[0].keys():
        #     outputs[k] = torch.cat([part[k] for part in batch_parts_outputs], dim=1)

        # loss
        losses = self.loss(outputs)

        # logging
        tqdm_dict = {
            "train_loss": losses["total"],
            "train_loss_task": losses["task"],
            "train_loss_recover": losses["recover"],
        }
        return {
            "loss": tqdm_dict["train_loss"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    # def training_epoch_end(self, outputs):
    #     pprint(outputs)
    #     avg_train_loss = torch.stack([out["loss"] for out in outputs]).mean()

    #     # val_loss_mean = 0
    #     # for output in outputs:
    #     #     val_loss_mean += output["val_loss"]
    #     # val_loss_mean /= len(outputs)
    #     tqdm_dict = {"avg_train_loss": avg_train_loss}

    #     results = {
    #         "train_loss": avg_train_loss,
    #         "progress_bar": tqdm_dict,
    #         "log": tqdm_dict,
    #     }

    #     return results

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
        tqdm_dict = {
            "val_loss": losses["total"],
            "val_loss_task": losses["task"],
            "val_loss_recover": losses["recover"],
        }
        return {
            "val_loss": tqdm_dict["val_loss"],
            "val_loss_task": tqdm_dict["val_loss_task"],
            "val_loss_recover": tqdm_dict["val_loss_recover"],
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        avg_val_loss_task = torch.stack(
            [out["val_loss_task"] for out in outputs]
        ).mean()
        avg_val_loss_recover = torch.stack(
            [out["val_loss_recover"] for out in outputs]
        ).mean()

        # val_loss_mean = 0
        # for output in outputs:
        #     val_loss_mean += output["val_loss"]
        # val_loss_mean /= len(outputs)
        tqdm_dict = {
            "val_loss": avg_val_loss,
            "val_loss_task": avg_val_loss_task,
            "val_loss_recover": avg_val_loss_recover,
        }

        results = {
            "val_loss": avg_val_loss,
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
    def _init_dataset(self, dset_type):
        data_path = self.hparams.dataset.path
        data_cls = self.hparams.dataset.cls
        emb_path = self.hparams.dataset.emb_path
        on_memory = self.hparams.dataset.on_memory
        noise = self.hparams.noise.type
        noise_ratio = self.hparams.noise.ratio

        data_cls = {
            "tr": TripleEmbeddingDataset,
            "emb": EmbeddingDataset,
            "emb-lbl": EmbeddingLabelDataset,
        }[data_cls]
        return data_cls(
            data_path,
            emb_path,
            dset_type,
            noise=noise,
            noise_ratio=noise_ratio,
            on_memory=on_memory,
        )

    def _init_datasets(self):
        data_path = self.hparams.dataset.path
        data_cls = self.hparams.dataset.cls
        emb_path = self.hparams.dataset.emb_path
        noise = self.hparams.noise.type
        noise_ratio = self.hparams.noise.ratio

        self._train_dataset = self._init_dataset("train")
        self._val_dataset = self._init_dataset("val")

    def _get_dataloader(self, dataset, shuffle=False):
        batch_size = self.hparams.train.batch_size if self.training else 2 ** 13
        num_workers = 2
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
