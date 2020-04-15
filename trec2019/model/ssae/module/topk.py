import math
import torch


class BatchTopK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k or 1

    def forward(self, x):
        if self.training:
            assert x.dim() == 2
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

    def _backward_hook(self, grad):
        if self.training:
            _grad = torch.zeros_like(grad).scatter(
                0, self.indices, grad.gather(0, self.indices)
            )
        else:
            _grad = grad
        return _grad
