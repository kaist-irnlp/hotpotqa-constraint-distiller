from torch import nn
import torch
import math
import torch.nn.functional as F
import abc
from trec2019.model.wta.helper import *


class WTAModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()
        # self.apply(self._init_weights)

    def forward(self, x):
        features = self.layers(x)
        return F.normalize(features, dim=-1)

    def on_epoch_end(self):
        self.apply(updateBoostStrength)
        self.apply(rezeroWeights)

    # TODO: May consider weight sharing (https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc)
    # TODO: Also see (https://pytorch.org/blog/pytorch-0_4_0-migration-guide/)
    def _init_layers(self):
        self.layers = nn.Sequential()

        n = self.hparams.model_n.n
        k = self.hparams.model_k.k
        weight_sparsity = self.hparams.model.weight_sparsity
        normalize_weights = self.hparams.model.normalize_weights
        k_inference_factor = self.hparams.model.k_inference_factor
        boost_strength = self.hparams.model.boost_strength
        boost_strength_factor = self.hparams.model.boost_strength_factor
        next_input_size = self.hparams.model.input_size
        for i in range(len(n)):
            linear = nn.Linear(next_input_size, n[i])
            if 0 < weight_sparsity < 1:
                linear = SparseWeights(linear, weightSparsity=weight_sparsity)
                if normalize_weights:
                    linear.apply(normalizeSparseWeights)
            self.layers.add_module(f"linear_{i+1}", linear)
            self.layers.add_module(f"bn_{i+1}", nn.BatchNorm1d(n[i], affine=False))
            # add kwinner layer
            k = math.floor(n[i] * k[i])
            kwinner = KWinners(
                n=n[i],
                k=k,
                kInferenceFactor=k_inference_factor,
                boostStrength=boost_strength,
                boostStrengthFactor=boost_strength_factor,
            )
            self.layers.add_module(f"kwinner_{i+1}", kwinner)
            # self.layers.add_module(f"relu_{i+1}", nn.ReLU())
            # self.layers.add_module(f"kwinner_{i+1}", BatchTopK(k[i]))
            next_input_size = n[i]
        # save output_size
        self.output_size = next_input_size

    # def _init_weights(self, m):
    #     if type(m) == nn.Linear:
    #         w = m.weight
    #         if self.hparams.model.use_sparse_weights:
    #             nn.init.sparse_(w, sparsity=0.3)
    #         else:
    #             torch.nn.init.kaiming_normal_(w)
    #         m.bias.data.fill_(0.01)


class BatchTopK(nn.Module):
    def __init__(self, k_ratio=1.0, batchwise=False):
        super().__init__()
        self.k_ratio = k_ratio
        self.batchwise = batchwise
        if batchwise:
            self.k_dim = 0  # batch
        else:
            self.k_dim = -1  # emb

    # TODO: https://discuss.pytorch.org/t/implementing-k-sparse-autoencoder-on-fasttext-embedding-the-output-is-strange/39245/2
    def forward(self, x):
        if self.batchwise:
            batch_size = x.shape[0]
            k = math.ceil(self.k_ratio * batch_size)
        else:
            emb_size = x.shape[-1]
            k = math.ceil(self.k_ratio * emb_size)
        _, self.indices = torch.topk(x, k, dim=self.k_dim)
        mask = torch.zeros(x.size()).type_as(x)
        mask.scatter_(self.k_dim, self.indices, 1)
        output = torch.mul(x, mask)

        if self.training:
            output.register_hook(self._backward_hook)

            # buffer, self.indices = self._topk(x, k, dim=0, largest=True)
            # # output
            # output = torch.zeros_like(x).scatter(0, self.indices, buffer)
            # output.register_hook(self._backward_hook)
        # else:
        #     output = x

        return output

    def _backward_hook(self, grad):
        if self.training:
            mask = torch.zeros(grad.size()).type_as(grad)
            mask.scatter_(self.k_dim, self.indices, 1)
            grad.mul_(mask)
            return grad

            # _grad = torch.zeros_like(grad).scatter(
            #     self.k_dim,
            #     self.indices,
            #     torch.gather(grad, self.k_dim, self.indices),
            # )
        else:
            return grad

