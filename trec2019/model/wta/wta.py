from torch import nn
import torch
import math
import torch.nn.functional as F
import abc
from trec2019.model.wta.nupic import *


class WTAModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()

    def forward(self, x):
        features = self.layers(x)
        return F.normalize(features, dim=-1)

    def on_epoch_end(self):
        self.apply(update_boost_strength)
        self.apply(rezero_weights)

    def _init_layers(self):
        self.layers = nn.Sequential()

        n = self.hparams.model_n.n
        k = self.hparams.model_k.k
        weight_sparsity = self.hparams.model.weight_sparsity
        normalize_weights = self.hparams.model.normalize_weights
        dropout = self.hparams.model.dropout
        k_inference_factor = self.hparams.model.k_inference_factor
        boost_strength = self.hparams.model.boost_strength
        boost_strength_factor = self.hparams.model.boost_strength_factor
        next_input_size = self.hparams.model.input_size
        for i in range(len(n)):
            linear = nn.Linear(next_input_size, n[i])
            if 0 < weight_sparsity < 1:
                linear = SparseWeights(linear, sparsity=weight_sparsity)
                if normalize_weights:
                    linear.apply(normalize_sparse_weights)
            self.layers.add_module(f"linear_{i+1}", linear)
            self.layers.add_module(f"selu_{i+1}", nn.SELU())
            self.layers.add_module(f"bn_{i+1}", nn.BatchNorm1d(n[i], affine=False))
            # dropout
            self.layers.add_module(f"dropout_{i+1}", nn.Dropout(dropout))
            # add kwinner layer
            kwinner = KWinners(
                n=n[i],
                percent_on=k[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                break_ties=True,
                relu=False,
                inplace=False,
            )
            self.layers.add_module(f"kwinner_{i+1}", kwinner)
            next_input_size = n[i]
        # save output_size
        self.output_size = next_input_size
