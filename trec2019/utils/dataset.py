import torch
import pandas as pd
import gzip
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
import h5py
import numpy as np
import gc
import zarr
from numcodecs import blosc
from torchtext.vocab import Vocab
import torchtext
from collections import Counter
import json
from sklearn.preprocessing import normalize
from trec2019.utils.dense import *


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class AbstractNoisyDataset(Dataset):
    def __init__(self, noise_ratio=0.0):
        self._noise_funcs = {
            "gaussian": self.add_gaussian_noise,
            "masking": self.add_masking_noise,
            "salt": self.add_salt_pepper_noise,
        }
        self._noise_ratio = noise_ratio

    def add_gaussian_noise(self, X, range_=[0, 1]):
        X_noisy = X + self._noise_ratio * np.random.normal(
            loc=0.0, scale=1.0, size=X.shape
        )
        # X_noisy = np.clip(X_noisy, range_[0], range_[1])

        return X_noisy

    def add_masking_noise(self, X):
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = 1, X.shape[0]
        n = int(ncol * self._noise_ratio)
        idx_noisy = np.random.choice(ncol, n, replace=False)
        X_noisy[idx_noisy] = 0

        return X_noisy

    def add_salt_pepper_noise(self, X):
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = 1, X.shape[0]
        n = int(ncol * self._noise_ratio)
        idx_noisy = np.random.choice(ncol, n, replace=False)
        X_noisy[idx_noisy] = np.random.binomial(1, 0.5, n)

        return X_noisy


class EmbeddingDataset(Dataset):
    def __init__(
        self, data_path, emb_path, on_memory=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.emb_path = emb_path
        self.on_memory = on_memory
        self._load_data()

    def _load_data(self):
        data = zarr.open(str(self.data_path), "r")
        self.embedding = data[self.emb_path]
        if self.on_memory:
            self.embedding = self.embedding[:]

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        data = self.embedding[index]
        return {
            "data": data.astype("f4"),
        }


class EmbeddingLabelDataset(EmbeddingDataset):
    def __init__(
        self, data_path, emb_path, on_memory=False,
    ):
        super().__init__(
            data_path, emb_path, on_memory=on_memory,
        )

    def _load_data(self):
        super()._load_data()
        data = zarr.open(str(self.data_path), "r")
        self.label = data.label[:]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["target"] = self.label[index].astype("i8")
        return item


class TripleEmbeddingDataset(Dataset):
    def __init__(
        self, data_path, emb_path, on_memory=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.emb_path = emb_path
        self.on_memory = on_memory
        self._load_data()

    def _load_data(self):
        z = zarr.open(str(self.data_path), "r")
        self.query, self.pos, self.neg, self.target_pos, self.target_neg = (
            z.query[self.emb_path],
            z.pos[self.emb_path],
            z.neg[self.emb_path],
            z.target_pos[:],
            z.target_neg[:],
        )
        if self.on_memory:
            self.query, self.pos, self.neg = self.query[:], self.pos[:], self.neg[:]

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        query, pos, neg, target_pos, target_neg = (
            self.query[index],
            self.pos[index],
            self.neg[index],
            self.target_pos[index],
            self.target_neg[index],
        )
        return {
            "query": query,
            "pos": pos,
            "neg": neg,
            "target_pos": target_pos,
            "target_neg": target_neg,
        }


if __name__ == "__main__":
    # data
    data_dir = "D:/Data/news20/test.zarr"
    dataset = EmbeddingLabelDataset(data_dir, "dense/bert-base-cased")
    # test
    loader = DataLoader(dataset, batch_size=2)
    for i, batch in enumerate(loader):
        print(batch["data"].shape)
        if i > 1:
            break
