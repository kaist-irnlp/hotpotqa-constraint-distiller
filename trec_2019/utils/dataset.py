import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
import h5py
import numpy as np


class TRECTripleEmbeddingDataset(Dataset):
    HDF_KEY = '/data'

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = None
        self._length = self._get_length()

    def _get_length(self):
        with h5py.File(self.data_path, 'r') as f:
            return len(f[self.HDF_KEY]["table"])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.data_path, "r")[self.HDF_KEY]["table"]
        # return a sample
        sample = self.data[index]
        return {}


class TRECTripleIdDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep="\t", names=[
                                "q_id", "pos_id", "neg_id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
