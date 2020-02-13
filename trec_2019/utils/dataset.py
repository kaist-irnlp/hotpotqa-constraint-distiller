import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
import h5py
import numpy as np


class TRECTripleEmbeddingDataset(Dataset):
    HDF_KEY = "/data"
    IDX_Q_EMB = 0
    IDX_POS_EMB = 1
    IDX_NEG_EMB = 2

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = None
        self._num_samples = self._read_num_samples()
        self._dimension = self._read_dimension()

    def get_dimension(self):
        return self._dimension

    def _read_dimension(self):
        with h5py.File(self.data_path, "r") as fin:
            dataset = fin[self.HDF_KEY]["table"]
            return dataset[0][0].shape[0]

    def _read_num_samples(self):
        with h5py.File(self.data_path, "r") as fin:
            dataset = fin[self.HDF_KEY]["table"]
            return dataset.shape[0]

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.data_path, "r")[self.HDF_KEY]["table"]
        # return a sample
        sample = self.data[index]
        q_emb = np.array(sample[self.IDX_Q_EMB], dtype=np.float32)
        pos_emb = np.array(sample[self.IDX_POS_EMB], dtype=np.float32)
        neg_emb = np.array(sample[self.IDX_NEG_EMB], dtype=np.float32)
        return {"q_emb": q_emb, "pos_emb": pos_emb, "neg_emb": neg_emb}


class TRECTripleIdDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep="\t", names=["q_id", "pos_id", "neg_id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
