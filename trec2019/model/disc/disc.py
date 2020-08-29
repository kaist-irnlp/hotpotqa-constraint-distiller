from torch import nn
import torch
import torch.nn.functional as F
import abc


class DiscModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()

    def forward(self, x):
        return self.layers(x)

    def _init_layers(self):
        in_dim = self.hparams.model_n.n[-1] * 2
        h_dims = self.hparams.disc.hidden
        out_dim = self.hparams.disc.out
        # weight_sparsity = self.hparams.disc.weight_sparsity
        ## build
        self.layers = nn.Sequential()
        for i in range(len(h_dims)):
            linear = nn.Linear(in_dim, h_dims[i])
            # use sparse weights
            # if 0 < weight_sparsity < 1:
            #     linear = SparseWeights(linear, sparsity=weight_sparsity)
            #     linear.apply(normalize_sparse_weights)
            # add modules
            self.layers.add_module(f"disc_linear_{i+1}", linear)
            self.layers.add_module(
                f"disc_bn_{i+1}", nn.BatchNorm1d(h_dims[i], affine=False)
            )
            self.layers.add_module(f"disc_selu_{i+1}", nn.SELU())
            # prepare next connection
            in_dim = h_dims[i]
        ## add out layer
        self.layers.add_module(f"disc_out_linear", nn.Linear(in_dim, out_dim))
        self.layers.add_module(f"disc_out_bn", nn.BatchNorm1d(out_dim, affine=False))
        self.layers.add_module(f"disc_out_selu", nn.SELU())
