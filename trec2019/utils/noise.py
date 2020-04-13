from torch import nn
import numpy as np


class Noise(nn.Module):
    def __init__(self, noise_type="gaussian"):
        super().__init__()
        self.add_noise = None
        if noise_type == "gaussian":
            self.add_noise = self.add_gaussian_noise
        elif noise_type == "masking":
            self.add_noise = self.add_masking_noise
        else:
            raise ValueError("Unknown noise type")

    def add_gaussian_noise(self, X, corruption_ratio=0.1, range_=[0, 1]):
        X_noisy = X + corruption_ratio * np.random.normal(
            loc=0.0, scale=1.0, size=X.shape
        )
        X_noisy = np.clip(X_noisy, range_[0], range_[1])

        return X_noisy

    def add_masking_noise(self, X, fraction=0.2):
        assert fraction >= 0 and fraction <= 1
        X_noisy = np.copy(X)
        nrow, ncol = X.shape
        n = int(ncol * fraction)
        for i in range(nrow):
            idx_noisy = np.random.choice(ncol, n, replace=False)
            X_noisy[i, idx_noisy] = 0

        return X_noisy

    def forward(self, x):
        if self.training:
            x = self.add_noise(x)
        return x
