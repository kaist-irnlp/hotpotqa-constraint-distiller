import math
import torch
from torch import nn


class BatchTopK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k or 1.0

    def forward(self, x):
        assert x.dim() == 2
        batch_size = x.shape[0]
        # determine k
        if self.training:
            k = self.k
        else:
            k = self.k * 1.5
        k_size = math.ceil(k * batch_size)
        buffer, self.indices = torch.topk(x, k_size, 0, True)
        output = torch.zeros_like(x).scatter(0, self.indices, buffer)

        # register backward hook
        if self.training:
            output.register_hook(self._backward_hook)
        return output

    def _backward_hook(self, grad):
        _grad = torch.zeros_like(grad).scatter(
            0, self.indices, grad.gather(0, self.indices)
        )
        return _grad
