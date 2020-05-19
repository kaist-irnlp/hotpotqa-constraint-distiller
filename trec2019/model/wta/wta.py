from torch import nn
import torch
import math


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


class WTAAutoencoder(nn.Module):
    def __init__(self, target_dims, orig_dims, k):
        super().__init__()
        # parameters
        self.target_dims = target_dims
        self.orig_dims = orig_dims

        # encoder
        self.encoder = nn.Sequential(
            GaussianNoise(),
            nn.Linear(self.orig_dims, self.target_dims),
            nn.BatchNorm1d(self.target_dims),
            nn.ReLU(),
            BatchTopK(k),
        )

        # decoder
        self.decoder = nn.Sequential(nn.Linear(self.target_dims, self.orig_dims),)

        # init weights
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        encoded = self.encoder(x)
        # encoded = encoded.clamp(min=0, max=1) # cap-ReLU
        self.register_buffer("encoded", encoded.detach())
        decoded = self.decoder(encoded)
        return decoded

    def set_curr_epoch(self, curr_epoch):
        self.curr_epoch = curr_epoch

    def get_encoded(self):
        for name, buf in self.named_buffers():
            if name in ["encoded"]:
                return buf
