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


class SparseNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init
        self._preprocess_params()
        self._init_layers()

    def forward(self, x):
        return self.layers(x)

    def on_epoch_end(self):
        self.apply(updateBoostStrength)
        self.apply(rezeroWeights)

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
