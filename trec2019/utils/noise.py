from torch import nn
import numpy as np
import torch


class Noise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        orig_x = x.detach()
        if self.training:
            x = self.add_noise(x)
        return x.type_as(orig_x)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.4, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.type_as(x).repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


# class GaussianNoise(Noise):
#     def __init__(self):
#         super().__init__()

#     def add_noise(self, X, corruption_ratio=0.2, range_=[0, 1]):
#         # X_noisy = X + corruption_ratio * np.random.normal(
#         #     loc=0.0, scale=1.0, size=X.shape
#         # )
#         # X_noisy = np.clip(X_noisy, range_[0], range_[1])
#         X_noisy = (X + corruption_ratio * torch.zeros(X.size()).normal_(
#             mean=0, std=1
#         ).type_as(X))
#         X_noisy.requires_grad = True

#         return X_noisy


class MaskingNoise(Noise):
    def __init__(self):
        super().__init__()

    def add_noise(self, X, fraction=0.25):
        assert fraction >= 0 and fraction <= 1
        X_noisy = np.copy(X)
        nrow, ncol = X.shape
        n = int(ncol * fraction)
        for i in range(nrow):
            idx_noisy = np.random.choice(ncol, n, replace=False)
            X_noisy[i, idx_noisy] = 0

        return X_noisy
