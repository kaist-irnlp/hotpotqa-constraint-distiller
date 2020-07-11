from torch import nn
import torch
import math
import torch.nn.functional as F


class WTAModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()
        self.apply(self._init_weights)

    def forward(self, x):
        features = self.layers(x)
        return F.normalize(features, p=2, dim=1)

    def on_epoch_end(self):
        pass

    def _init_layers(self):
        self.layers = nn.Sequential()

        n = self.hparams.model.n
        k = self.hparams.model.k
        next_input_size = self.hparams.model.input_size
        # TODO: May consider weight sharing (https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc)
        # TODO: Also see (https://pytorch.org/blog/pytorch-0_4_0-migration-guide/)
        for i in range(len(n)):
            self.layers.add_module(f"linear_{i+1}", nn.Linear(next_input_size, n[i]))
            self.layers.add_module(f"bn_{i+1}", nn.BatchNorm1d(n[i]))
            self.layers.add_module(f"relu_{i+1}", nn.ReLU())
            self.layers.add_module(f"kwinner_{i+1}", BatchTopK(k[i]))
            next_input_size = n[i]
        # save output_size
        self.output_size = next_input_size

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            w = m.weight
            if self.hparams.model.use_sparse_weights:
                nn.init.sparse_(w, sparsity=0.3)
            else:
                torch.nn.init.kaiming_normal_(w)
            m.bias.data.fill_(0.01)


class BatchTopK(nn.Module):
    DIM_BATCH = 0

    def __init__(self, k=1.0):
        super().__init__()
        self._k = k
        self._h = None

    # TODO: https://discuss.pytorch.org/t/implementing-k-sparse-autoencoder-on-fasttext-embedding-the-output-is-strange/39245/2
    def forward(self, x):
        batch_size = x.shape[0]
        # TODO: k * 1.5 (inference)?
        k = math.ceil(self._k * batch_size)
        _, self.indices = torch.topk(x, k, dim=self.DIM_BATCH)
        mask = torch.zeros(x.size()).type_as(x)
        mask.scatter_(self.DIM_BATCH, self.indices, 1)
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

    def set_k(self, k):
        self._k = k

    def _backward_hook(self, grad):
        if self.training:
            mask = torch.zeros(grad.size()).type_as(grad)
            mask.scatter_(self.DIM_BATCH, self.indices, 1)
            _grad = torch.mul(grad, mask)

            # _grad = torch.zeros_like(grad).scatter(
            #     self.DIM_BATCH,
            #     self.indices,
            #     torch.gather(grad, self.DIM_BATCH, self.indices),
            # )
        else:
            _grad = grad
        return _grad
