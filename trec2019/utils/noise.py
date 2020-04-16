from torch import nn
import numpy as np


class Noise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            noisy_x = self.add_noise(x)
        return noisy_x.type_as(x)


class GaussianNoise(Noise):
    def __init__(self):
        super().__init__()

    def add_noise(self, X, corruption_ratio=0.2, range_=[0, 1]):
        X_noisy = X + corruption_ratio * np.random.normal(
            loc=0.0, scale=1.0, size=X.shape
        )
        X_noisy = np.clip(X_noisy, range_[0], range_[1])

        return X_noisy


class MaskingNoise(Noise):
    def __init__(self):
        super().__init__()

    def add_noise(self, X, fraction=0.2):
        assert fraction >= 0 and fraction <= 1
        X_noisy = np.copy(X)
        nrow, ncol = X.shape
        n = int(ncol * fraction)
        for i in range(nrow):
            idx_noisy = np.random.choice(ncol, n, replace=False)
            X_noisy[i, idx_noisy] = 0

        return X_noisy
