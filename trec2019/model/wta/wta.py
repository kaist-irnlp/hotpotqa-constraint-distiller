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

    def forward(self, x):
        return self.layers(x)

    def on_epoch_end(self):
        pass

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
            # nn.init.sparse_(w, sparsity=0.1) # TODO 적용 고려
            m.bias.data.fill_(0.01)


class BatchTopK(nn.Module):
    def __init__(self, k=1.0):
        super().__init__()
        self._k = k
        self._h = None

    def forward(self, x):
        if self.training:
            # assert x.dim() == 2
            batch_size = x.shape[0]
            # k = self.k_list[self.curr_epoch]
            k = math.ceil(self._k * batch_size)

            self.buffer, self.indices = torch.topk(x, k, dim=0, largest=True)
            output = torch.zeros_like(x).scatter(0, self.indices, self.buffer)
            self._h = output.register_hook(self._backward_hook)
        else:
            output = x

        return output

    def set_k(self, k):
        self._k = k

    def _backward_hook(self, grad):
        if self.training:
            _grad = torch.zeros_like(grad).scatter(
                0, self.indices, torch.gather(grad, 0, self.indices)
            )
        else:
            _grad = grad
        return _grad
