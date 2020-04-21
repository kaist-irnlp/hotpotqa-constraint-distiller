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
        self._dset_cls = News20EmbeddingDataset

        # network
        self._init_dataset()
        self._init_layers()

    def _init_dataset(self):
        data_dir = Path(self.hparams.data_dir)
        self._train_dataset = self._dset_cls(str(data_dir / "train.zarr"))
        self._val_dataset = self._dset_cls(str(data_dir / "val.zarr"))
        self._test_dataset = self._dset_cls(str(data_dir / "test.zarr"))

    def _init_layers(self):
        # self._init_dense_layer()
        self._init_noise_layer()
        self._init_sparse_layer()
        self._init_out_layer()
        self._init_recover_layer()

    def _init_noise_layer(self):
        self.noise = GaussianNoise()

    def _init_out_layer(self):
        final_output_size = self.hparams.output_size
        if final_output_size and (final_output_size > 0):
            self.out = nn.Sequential(
                nn.Linear(self.sparse.output_size, final_output_size)
            )
        else:
            self.out = None

    def _init_recover_layer(self):
        orig_size = self.hparams.input_size
        if self.hparams.use_recovery_loss:
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

    def distance(self, x1, x2):
        # TODO: 고민 필요
        # return torch.pow(a - b, 2).sum(1).sqrt()
        # return F.cosine_similarity(a, b)
        return torch.norm(x1 - x2, dim=1)

    def loss_recovery(self, input, target):
        return F.mse_loss(input, target)
        # return F.l1_loss(input, target)

    def loss_triplet(self, q, pos, neg):
        distance_p = self.distance(q, pos)
        distance_n = self.distance(q, neg)
        # Should be distance_n > distance_p, so mark all as 1 (not -1)
        return F.margin_ranking_loss(
            distance_n, distance_p, torch.ones_like(distance_p), margin=1.0
        )

    def loss_classify(self, input, target):
        # input.shape() == (minibatch, C)
        return F.cross_entropy(input, target)

    def loss(self, outputs):
        # task loss
        loss_task = self.loss_classify(
            outputs["out"], outputs["target"].type(torch.long)
        )

        # recovery loss
        loss_recovery = self.loss_recovery(outputs["recover"], outputs["x"])

        return {
            "total": loss_task + (self.hparams.ratio_recovery_loss * loss_recovery),
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
        # noise_x = x

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
            out_x = sparse_x

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
        return {"loss": losses["total"], "progress_bar": tqdm_dict, "log": log_dict}

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
        return {"val_loss": losses["total"], "progress_bar": tqdm_dict, "log": log_dict}

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
            "log": {"val_loss": avg_val_loss},
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
            self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def _get_dataloader(self, dataset, test=False):
        # dist_sampler = DistributedSampler(dataset) if self.use_ddp else None
        batch_size = self.hparams.batch_size if not test else 2 ** 13
        num_workers = int(cpu_count() / 2) or 1
        pin_memory = False
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
            "--no_recovery_loss", dest="use_recovery_loss", action="store_false"
        )
        parser.add_argument(
            "--no_normalize_weights", dest="normalize_weights", action="store_false"
        )
        parser.add_argument(
            "--ratio_recovery_loss",
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
        if type(hparams.n) is str:
            hparams.n = eval(hparams.n)
        if type(hparams.k) is str:
            hparams.k = eval(hparams.k)
        if type(hparams.weight_sparsity) is str:
            hparams.weight_sparsity = eval(hparams.weight_sparsity)

        # validate & clean
        if type(hparams.n) is not list:
            hparams.n = [hparams.n]
            hparams.n = [int(n) for n in hparams.n]
        if type(hparams.k) is not list:
            hparams.k = [hparams.k] * len(hparams.n)
            hparams.k = [int(k) for k in hparams.k]
        assert len(hparams.n) == len(hparams.k)
        for i in range(len(hparams.n)):
            assert hparams.k[i] <= hparams.n[i]
        if type(hparams.weight_sparsity) is not list:
            hparams.weight_sparsity = [hparams.weight_sparsity] * len(hparams.n)
            hparams.weight_sparsity = [float(w) for w in hparams.weight_sparsity]
        assert len(hparams.n) == len(hparams.weight_sparsity)
        for i in range(len(hparams.weight_sparsity)):
            assert hparams.weight_sparsity[i] >= 0

        # DEBUG
        print(vars(hparams))

        # save cleaned values
        self.hparams = hparams

    def _init_layers(self):
        # extract params
        hparams = self.hparams
        input_size = hparams.input_size
        n = hparams.n
        k = hparams.k
        normalize_weights = self.hparams.normalize_weights
        use_batch_norm = self.hparams.use_batch_norm
        dropout = self.hparams.dropout
        weight_sparsity = self.weightSparsity = hparams.weight_sparsity
        k_inference_factor = self.kInferenceFactor = hparams.k_inference_factor
        boost_strength = self.boostStrength = hparams.boost_strength
        boost_strength_factor = self.boostStrengthFactor = hparams.boost_strength_factor

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
