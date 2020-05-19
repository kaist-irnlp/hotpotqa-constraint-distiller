from torch import nn
import torch
import math


class WTAModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()
        self.apply(self._init_weights)

    def _init_layers(self):
        self.layers = nn.Sequential()

        n = self.hparams.model.n
        k = self.hparams.model.k
        next_input_size = self.hparams.model.input_size
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
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.layers(x)


class BatchTopK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k or 1

    def forward(self, x):
        if self.training:
            # assert x.dim() == 2
            batch_size = x.shape[0]
            # k = self.k_list[self.curr_epoch]
            k_size = math.ceil(self.k * batch_size)

            buffer, self.indices = torch.topk(x, k_size, 0, True)
            output = torch.zeros_like(x).scatter(0, self.indices, buffer)
        else:
            output = x

        # register backward hook
        output.register_hook(self._backward_hook)
        return output

    def set_k(self, k):
        self.k = k

    def _backward_hook(self, grad):
        if self.training:
            _grad = torch.zeros_like(grad).scatter(
                0, self.indices, grad.gather(0, self.indices)
            )
        else:
            _grad = grad
        return _grad
