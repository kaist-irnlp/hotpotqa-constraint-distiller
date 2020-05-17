"""
This file defines the core research contribution   
"""
import os
import sys
import torch
import gc

from torch import optim
import torch_optimizer
from torch import nn
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import gzip

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import spacy
from nltk.util import ngrams
from textblob import TextBlob
import json

import logging
import zarr
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
from pathlib import Path
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, SubwordVocab

# from transformers import BertModel
# from transformers import BertTokenizer
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from omegaconf import OmegaConf
from argparse import Namespace

from trec2019.model.sparsenet.helper import *
from collections import OrderedDict
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
from trec2019.utils.dataset import *

# from trec2019.utils.dense import *
from trec2019.utils.noise import *

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
_root_dir = str(Path(__file__).parent.absolute())


class SparseNet(pl.LightningModule):
    def __init__(self, hparams):
        super(SparseNet, self).__init__()
        self.hparams = hparams

        # dataset type
        self._dset_cls = EmbeddingLabelDataset

        # network
        self._init_dataset()
        self._init_layers()

    def _init_dataset(self):
        data_path = Path(self.hparams.dataset.path)
        arr_path = self.hparams.dataset.arr_path
        self._train_dataset = self._dset_cls(str(data_path / "train.zarr"), arr_path)
        self._val_dataset = self._dset_cls(str(data_path / "val.zarr"), arr_path)
        self._test_dataset = self._dset_cls(str(data_path / "test.zarr"), arr_path)

    def _init_layers(self):
        # self._init_dense_layer()
        self._init_noise_layer()
        self._init_sparse_layer()
        self._init_out_layer()
        self._init_recover_layer()

    def _init_noise_layer(self):
        self.noise = GaussianNoise()

    def _init_out_layer(self):
        D_in = self.sparse.output_size
        H = self.sparse.output_size
        D_out = self.hparams.model.output_size
        if (D_out is not None) and (self.hparams.loss.use_task_loss):
            self.out = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))
        else:
            self.out = None

    def _init_recover_layer(self):
        orig_size = self.hparams.model.input_size
        if self.hparams.loss.use_recovery_loss:
            self.recover = nn.Linear(self.sparse.output_size, orig_size)
        else:
            self.recover = None

    def _init_sparse_layer(self):
        # self.hparams.input_size = (
        #     self.dense.get_dim()
        #     if (self.dense is not None)
        #     else self._train_dataset.get_dim()
        # )  # TODO: is it safe to do this automatically?
        self.sparse = SparseNetModel(self.hparams)

    # def _init_dense_layer(self):
    #     dense_model = self.hparams.dense or None
    #     if dense_model is None:
    #         self.tokenizer = None
    #         self.dense = None
    #     elif dense_model == "bow":
    #         vocab = get_bow_vocab()
    #         # vocab.vectors = F.normalize(vocab.vectors, p=2, dim=1)
    #         self.tokenizer = BowTokenizer(vocab)
    #         self.dense = BowEmbedding(vocab)
    #     elif dense_model == "bert":
    #         weights = "bert-base-uncased"
    #         self.tokenizer = BertTokenizer(weights)
    #         self.dense = BertEmbedding(weights)
    #     else:
    #         raise ValueError(f"Unknown dense model: {dense_model}")

    # Task Loss: Ranking
    def distance(self, x1, x2):
        # TODO: 고민 필요
        # return torch.pow(a - b, 2).sum(1).sqrt()
        # return F.cosine_similarity(a, b)
        return torch.norm(x1 - x2, dim=1)

    def loss_triplet(self, q, pos, neg):
        distance_p = self.distance(q, pos)
        distance_n = self.distance(q, neg)
        # Should be distance_n > distance_p, so mark all as 1 (not -1)
        return F.margin_ranking_loss(
            distance_n, distance_p, torch.ones_like(distance_p), margin=1.0
        )

    # Task Loss: Classification
    def loss_classify(self, input, target):
        # input.shape() == (minibatch, C)
        return F.cross_entropy(input, target)

    # Autoencoder Loss (For generalizability)
    def loss_recovery(self, input, target):
        return F.mse_loss(input, target)
        # return F.l1_loss(input, target)

    def loss(self, outputs):
        target = outputs["target"].type(torch.long)

        # autoencoder loss * lambda
        loss_recovery = (
            self.loss_recovery(outputs["recover"], outputs["x"])
            * self.hparams.loss.recovery_loss_ratio
        )

        # task loss
        if self.hparams.loss.use_task_loss:
            loss_task = self.loss_classify(outputs["out"], target)
        else:
            loss_task = torch.zeros((1,)).type_as(loss_recovery)

        return {
            "total": loss_task + loss_recovery,
            "task": loss_task,
            "recovery": loss_recovery,
        }

    def forward(self, x):
        # dense
        # if self.dense is not None:
        #     dense_x = self.forward_dense(x)
        # else:
        #     dense_x = x

        # noise
        noise_x = self.noise(x)

        # sparse
        sparse_x = self.sparse(noise_x)

        # 1. recover
        if self.recover is not None:
            recover_x = self.recover(sparse_x)
        else:
            recover_x = x

        # 2. out
        if self.out is not None:
            out_x = self.out(sparse_x)
        else:
            out_x = torch.zeros_like(x)

        features = {"x": x, "sparse": sparse_x, "recover": recover_x, "out": out_x}
        return features

    def training_step(self, batch, batch_idx):
        text, target = batch["data"], batch["target"]

        # forward
        features = self.forward(text)

        return {**features, "target": target}

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
        text, target = batch["data"], batch["target"]

        # forward
        features = self.forward(text)

        return {**features, "target": target}

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
        text, target = batch["data"], batch["target"]

        # forward
        features = self.forward(text)

        return {**features, "target": target}

    def test_step_end(self, outputs):
        # loss
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

    ###

    def on_epoch_end(self):
        self.apply(updateBoostStrength)
        self.apply(rezeroWeights)

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch_optimizer.RAdam(
            self.parameters(), lr=self.hparams.train.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def _get_dataloader(self, dataset, test=False):
        # dist_sampler = DistributedSampler(dataset) if self.use_ddp else None
        batch_size = self.hparams.train.batch_size if not test else 2 ** 13
        num_workers = int(cpu_count() / 2) or 1
        pin_memory = True
        # num_workers = 0
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset, test=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser])
        # parser.add_argument("--dense", type=str, choices=["bow", "bert"], default=None)
        # parser.add_argument(
        #     "--fine_tune", "-ft", action="store_true", help="Fine-tune dense models"
        # )
        parser.add_argument("--n", type=int, nargs="+", required=True)
        parser.add_argument("--k", type=int, nargs="+", required=True)
        parser.add_argument("--input_size", type=int, required=True)
        parser.add_argument("--output_size", type=int, required=True)
        parser.add_argument("--k_inference_factor", default=1.5, type=float)
        parser.add_argument("--weight_sparsity", default=0.3, type=float)
        parser.add_argument("--boost_strength", default=1.5, type=float)
        parser.add_argument(
            "--boost_strength_factor", default=0.85, type=float,
        )
        parser.add_argument("--dropout", default=0.2, type=float)
        parser.add_argument("--use_batch_norm", default=True, type=bool)
        parser.add_argument(
            "--use_recovery_loss", dest="use_recovery_loss", action="store_true"
        )
        parser.add_argument(
            "--no_task_loss", dest="use_task_loss", action="store_false"
        )
        parser.add_argument(
            "--no_normalize_weights", dest="normalize_weights", action="store_false"
        )
        parser.add_argument(
            "--recovery_loss_ratio",
            default=1.0,
            type=float,
            help="Ratio for recovery loss",
        )

        return parser


class SparseNetModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init
        self._preprocess_params()
        self._init_layers()

    def forward(self, x):
        return self.layers(x)

    def _preprocess_params(self):
        hparams = self.hparams

        # to make compatible with pytorch-lightning model loading
        if type(hparams.model.n) is str:
            hparams.model.n = eval(hparams.model.n)
        if type(hparams.model.k) is str:
            hparams.model.k = eval(hparams.model.k)
        if type(hparams.model.weight_sparsity) is str:
            hparams.model.weight_sparsity = eval(hparams.model.weight_sparsity)

        # validate & clean
        if not type(hparams.model.n) in (list, ListConfig):
            hparams.model.n = [hparams.model.n]
            hparams.model.n = [int(n) for n in hparams.model.n]
        if not type(hparams.model.k) in (list, ListConfig):
            hparams.model.k = [hparams.model.k] * len(hparams.model.n)
            hparams.model.k = [int(k) for k in hparams.model.k]
        assert len(hparams.model.n) == len(hparams.model.k)
        for i in range(len(hparams.model.n)):
            assert hparams.model.k[i] <= hparams.model.n[i]
        if not type(hparams.model.weight_sparsity) in (list, ListConfig):
            hparams.model.weight_sparsity = [hparams.model.weight_sparsity] * len(
                hparams.model.n
            )
            hparams.model.weight_sparsity = [
                float(w) for w in hparams.model.weight_sparsity
            ]
        assert len(hparams.model.n) == len(hparams.model.weight_sparsity)
        for i in range(len(hparams.model.weight_sparsity)):
            assert hparams.model.weight_sparsity[i] >= 0

        # DEBUG
        print(vars(hparams))

        # save cleaned values
        self.hparams = hparams

    def _init_layers(self):
        # extract params
        hparams = self.hparams
        input_size = hparams.model.input_size
        n = hparams.model.n
        k = hparams.model.k
        normalize_weights = self.hparams.model.normalize_weights
        use_batch_norm = self.hparams.model.use_batch_norm
        dropout = self.hparams.model.dropout
        weight_sparsity = self.weightSparsity = hparams.model.weight_sparsity
        k_inference_factor = self.kInferenceFactor = hparams.model.k_inference_factor
        boost_strength = self.boostStrength = hparams.model.boost_strength
        boost_strength_factor = (
            self.boostStrengthFactor
        ) = hparams.model.boost_strength_factor

        # define network
        # TODO: May consider weight sharing (https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc)
        # Also see (https://pytorch.org/blog/pytorch-0_4_0-migration-guide/)
        self.layers = nn.Sequential()
        for i in range(len(n)):
            if n[i] != 0:
                linear = nn.Linear(input_size, n[i])
                if 0 < weight_sparsity[i] < 1:
                    linear = SparseWeights(linear, weightSparsity=weight_sparsity[i])
                    # if normalize_weights:
                    if normalize_weights:
                        linear.apply(normalizeSparseWeights)
                self.layers.add_module(f"sparse_{i+1}", linear)

                if use_batch_norm:
                    self.layers.add_module(
                        f"sparse_{i+1}_bn", nn.BatchNorm1d(n[i], affine=False)
                    )

                if dropout > 0.0:
                    self.layers.add_module(f"sparse_{i+1}_dropout", nn.Dropout(dropout))

                if 0 < k[i] < n[i]:
                    kwinner = KWinners(
                        n=n[i],
                        k=k[i],
                        kInferenceFactor=k_inference_factor,
                        boostStrength=boost_strength,
                        boostStrengthFactor=boost_strength_factor,
                    )
                    self.layers.add_module(f"sparse_{i+1}_kwinner", kwinner)
                else:
                    self.layers.add_module(f"sparse_{i+1}_relu", nn.ReLU())
                # Feed this layer output into next layer input
                input_size = n[i]

        self.output_size = input_size

    def maxEntropy(self):
        entropy = 0
        for module in self.modules():
            if module == self:
                continue
            if hasattr(module, "maxEntropy"):
                entropy += module.maxEntropy()
        return entropy

    def entropy(self):
        entropy = 0
        for module in self.modules():
            if module == self:
                continue
            if hasattr(module, "entropy"):
                entropy += module.entropy()
        return entropy

    def pruneWeights(self, minWeight):
        """
        Prune all the weights whose absolute magnitude is less than minWeight
        :param minWeight: min weight to prune. If zero then no pruning
        :type minWeight: float
        """
        if minWeight == 0.0:
            return

        # Collect all weights
        weights = [v for k, v in self.named_parameters() if "weight" in k]
        for w in weights:
            # Filter weights above threshold
            mask = torch.ge(torch.abs(w.data), minWeight)
            # Zero other weights
            w.data.mul_(mask.type(torch.float32))

    def pruneDutycycles(self, threshold=0.0):
        """
        Prune all the units with dutycycles whose absolute magnitude is less than
        the given threshold
        :param threshold: min threshold to prune. If less than zero then no pruning
        :type threshold: float
        """
        if threshold < 0.0:
            return

        # Collect all layers with 'dutyCycle'
        for m in self.modules():
            if m == self:
                continue
            if hasattr(m, "pruneDutycycles"):
                m.pruneDutycycles(threshold)
