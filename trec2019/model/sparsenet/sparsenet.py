"""
This file defines the core research contribution   
"""
import os
import sys
import torch
import gc
from torch import optim
from torch import nn
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import spacy
from nltk.util import ngrams
from textblob import TextBlob

import logging
import zarr
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel
from transformers import BertTokenizer
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd

from trec2019.utils.dataset import *
from trec2019.model.sparsenet.helper import *
from collections import OrderedDict
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
from trec2019.utils.dense import *

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
root_dir = str(Path(__file__).parent.absolute())


class SparseNet(pl.LightningModule):
    def __init__(self, hparams):
        super(SparseNet, self).__init__()
        self.hparams = hparams
        self.encoded = None

        # network
        self._init_dense()
        self._preprocess_sparse_params()
        self._init_sparse()

    def prepare_data(self):
        data_path = self.hparams.data_path
        train, val, test = self.split_train_val_test()
        self._train_dataset = TripleDataset(data_path, train, self.tokenizer)
        self._val_dataset = TripleDataset(data_path, val, self.tokenizer)
        self._test_dataset = TripleDataset(data_path, test, self.tokenizer)

    def _get_bow_vocab(self):
        VOCAB_PATH = Path(root_dir) / "../../vocab/vocab.parquet"
        VECTORS = "fasttext.en.300d"
        MIN_FREQ = 10
        MAX_SIZE = 100000
        vocab_counts = Counter(
            {row.word: row.count for row in pd.read_parquet(VOCAB_PATH).itertuples()}
        )
        return Vocab(
            vocab_counts,
            vectors=VECTORS,
            min_freq=MIN_FREQ,
            max_size=MAX_SIZE, 
            unk_init=torch.Tensor.normal_,
        )

    def _init_dense(self):
        # init vocab
        if self.hparams.dense == "bow":
            vocab = self._get_bow_vocab()
            self.tokenizer = BowTokenizer(vocab)
            self.dense = BowEmbedding(vocab)
        elif self.hparams.dense == "bert":
            weights = "bert-base-uncased"
            self.tokenizer = BertTokenizer(weights)
            self.dense = BertEmbedding(weights)

    def distance(self, a, b):
        # return torch.pow(a - b, 2).sum(1).sqrt()
        return F.cosine_similarity(a, b)

    def loss_recovery(self, input, target):
        return F.mse_loss(input, target)

    # def loss_triplet(self, delta):
    #     return torch.log1p(torch.sum(torch.exp(delta)))

    def loss(self, out):
        (
            dense_query,
            dense_doc_pos,
            dense_doc_neg,
            sparse_query,
            sparse_doc_pos,
            sparse_doc_neg,
            recovered_query,
            recovered_doc_pos,
            recovered_doc_neg,
        ) = out

        # triplet loss
        distance_p = self.distance(sparse_query, sparse_doc_pos)
        distance_n = self.distance(sparse_query, sparse_doc_neg)
        # Should be distance_n > distance_p, so mark all as 1 (not -1)
        loss_triplet_val = F.margin_ranking_loss(
            distance_n, distance_p, torch.ones_like(distance_p)
        )
        # delta = distance_n - distance_p
        # loss_triplet_val = self.loss_triplet(delta)

        # recovery loss
        loss_recovery_val = (
            self.loss_recovery(recovered_query, dense_query)
            + self.loss_recovery(recovered_doc_pos, dense_doc_pos)
            + self.loss_recovery(recovered_doc_neg, dense_doc_neg)
        )

        # loss = triplet
        return loss_triplet_val + loss_recovery_val
        # return loss_triplet_val

    def forward(self, query, doc_pos, doc_neg):
        # dense
        with torch.no_grad():
            dense_query, dense_doc_pos, dense_doc_neg = (
                self.dense(query),
                self.dense(doc_pos),
                self.dense(doc_neg),
            )

        # sparse
        sparse_query, sparse_doc_pos, sparse_doc_neg = (
            self.sparse(dense_query),
            self.sparse(dense_doc_pos),
            self.sparse(dense_doc_neg),
        )

        # recover
        recovered_query, recovered_doc_pos, recovered_doc_neg = (
            self.recover(sparse_query),
            self.recover(sparse_doc_pos),
            self.recover(sparse_doc_neg),
        )

        if self.training:
            # batch_size = batch.shape[0]
            batch_size = len(query)
            self.learning_iterations += batch_size

        return (
            dense_query,
            dense_doc_pos,
            dense_doc_neg,
            sparse_query,
            sparse_doc_pos,
            sparse_doc_neg,
            recovered_query,
            recovered_doc_pos,
            recovered_doc_neg,
        )

    def training_step(self, batch, batch_idx):
        query, doc_pos, doc_neg = batch["query"], batch["doc_pos"], batch["doc_neg"]

        # infer
        out = self.forward(query, doc_pos, doc_neg)

        return {"out": out}

    def training_step_end(self, outputs):
        # aggregate (dp or ddp)
        out = outputs["out"]

        # loss
        loss_val = self.loss(out)

        # logging
        tqdm_dict = {"train_loss": loss_val}
        log_dict = {"losses": tqdm_dict}
        return {"loss": loss_val, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        query, doc_pos, doc_neg = batch["query"], batch["doc_pos"], batch["doc_neg"]

        # infer
        out = self.forward(query, doc_pos, doc_neg)

        return {"out": out}

    def validation_step_end(self, outputs):
        # aggregate (dp or ddp)
        out = outputs["out"]

        # loss
        loss_val = self.loss(out)

        # logging
        tqdm_dict = {"val_loss": loss_val}
        log_dict = {"val_losses": tqdm_dict}
        return {"val_loss": loss_val, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([out["val_loss"] for out in outputs]).mean()

        # val_loss_mean = 0
        # for output in outputs:
        #     val_loss_mean += output["val_loss"]
        # val_loss_mean /= len(outputs)
        tqdm_dict = {"val_loss": avg_val_loss}

        results = {"progress_bar": tqdm_dict, "log": {"avg_val_loss": avg_val_loss}}

        return results

    def _init_sparse(self):
        self.learning_iterations = 0
        # self.flatten = Flatten()

        # Linear layers only (from original code)
        input_features = self.input_dim
        output_size = self.input_dim
        n = self.n
        k = self.k
        normalize_weights = self.hparams.normalize_weights
        weight_sparsity = self.weightSparsity
        use_batch_norm = self.hparams.use_batch_norm
        dropout = self.hparams.dropout
        k_inference_factor = self.kInferenceFactor
        boost_strength = self.boostStrength
        boost_strength_factor = self.boostStrengthFactor

        self.sparse = nn.Sequential()
        for i in range(len(n)):
            if n[i] != 0:
                linear = nn.Linear(input_features, n[i])
                if 0 < weight_sparsity[i] < 1:
                    linear = SparseWeights(linear, weightSparsity=weight_sparsity[i])
                    if normalize_weights:
                        linear.apply(normalizeSparseWeights)
                self.sparse.add_module(f"sparse_{i+1}", linear)
                # Weight sharing (https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc)
                # Also see (https://pytorch.org/blog/pytorch-0_4_0-migration-guide/)
                # class TiedAutoEncoderOffTheShelf(nn.Module):
                #     def __init__(self, inp, out, weight):
                #         super().__init__()
                #         self.encoder = nn.Linear(inp, out, bias=False)
                #         self.decoder = nn.Linear(out, inp, bias=False)

                #         # tie the weights
                #         self.encoder.weight.data = weight.clone()
                #         self.decoder.weight.data = self.encoder.weight.data.transpose(0,1)

                #     def forward(self, input):
                #         encoded_feats = self.encoder(input)
                #         reconstructed_output = self.decoder(encoded_feats)
                #         return encoded_feats, reconstructed_output

                # class MixedAppraochTiedAutoEncoder(nn.Module):
                #     def __init__(self, inp, out, weight):
                #         super().__init__()
                #         self.encoder = nn.Linear(inp, out, bias=False)
                #         self.encoder.weight.data = weight.clone()

                #     def forward(self, input):
                #         encoded_feats = self.encoder(input)
                #         reconstructed_output = F.linear(encoded_feats, self.encoder.weight.t())
                #         return encoded_feats, reconstructed_output

                if use_batch_norm:
                    self.sparse.add_module(
                        f"sparse_{i+1}_bn", nn.BatchNorm1d(n[i], affine=False)
                    )

                if dropout > 0.0:
                    self.sparse.add_module(f"sparse_{i+1}_dropout", nn.Dropout(dropout))

                if 0 < k[i] < n[i]:
                    kwinner = KWinners(
                        n=n[i],
                        k=k[i],
                        kInferenceFactor=k_inference_factor,
                        boostStrength=boost_strength,
                        boostStrengthFactor=boost_strength_factor,
                    )
                    self.sparse.add_module(f"sparse_{i+1}_kwinner", kwinner)
                else:
                    self.sparse.add_module(f"sparse_{i+1}_relu", nn.ReLU())
                # Feed this layer output into next layer input
                input_features = n[i]

        # Add one fully connected layer after all hidden layers
        self.recover = nn.Linear(input_features, output_size)

        # if useSoftmax:
        #     self.softmax = nn.LogSoftmax(dim=1)
        # else:
        #     self.softmax = None

    def on_epoch_end(self):
        self.apply(updateBoostStrength)
        self.apply(rezeroWeights)

    def get_encoded(self):
        return self.encoded

    def get_learning_iterations(self):
        return self.learning_iterations

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

    def _preprocess_sparse_params(self):
        hparams = self.hparams
        if type(hparams.n) is not list:
            hparams.n = [hparams.n]
        if type(hparams.k) is not list:
            hparams.k = [hparams.k] * len(hparams.n)
        assert len(hparams.n) == len(hparams.k)
        for i in range(len(hparams.n)):
            assert hparams.k[i] <= hparams.n[i]
        if type(hparams.weight_sparsity) is not list:
            hparams.weight_sparsity = [hparams.weight_sparsity] * len(hparams.n)
        assert len(hparams.n) == len(hparams.weight_sparsity)
        for i in range(len(hparams.weight_sparsity)):
            assert hparams.weight_sparsity[i] >= 0

        # DEBUG
        print(vars(hparams))

        # assign
        self.input_dim = self.dense.get_dim()
        self.k = hparams.k
        self.kInferenceFactor = hparams.k_inference_factor
        self.n = hparams.n
        self.weightSparsity = (
            hparams.weight_sparsity
        )  # Pct of weights that are non-zero
        self.boostStrengthFactor = hparams.boost_strength_factor
        self.boostStrength = hparams.boost_strength
        self.learning_iterations = 0

    def split_train_val_test(self):
        data_path = self.hparams.data_path
        dataset = zarr.open(data_path, "r")
        indices = np.array(range(len(dataset)))
        train, val_test = train_test_split(indices, test_size=0.2)
        val, test = train_test_split(val_test, test_size=0.5)
        return train, val, test

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def _get_dataloader(self, dataset, test=False):
        # dist_sampler = DistributedSampler(dataset) if self.use_ddp else None
        batch_size = self.hparams.batch_size if not test else 100000
        num_workers = int(cpu_count() / 2) or 1
        # num_workers = 0
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
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
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--k_inference_factor", default=1.5, type=float)
        parser.add_argument("--weight_sparsity", default=0.3, type=float)
        parser.add_argument("--boost_strength", default=1.5, type=float)
        parser.add_argument(
            "--boost_strength_factor", default=0.85, type=float,
        )
        parser.add_argument("--dropout", default=0.0, type=float)
        parser.add_argument("--use_batch_norm", default=True, type=bool)
        parser.add_argument("--normalize_weights", default=False, type=bool)

        return parser
