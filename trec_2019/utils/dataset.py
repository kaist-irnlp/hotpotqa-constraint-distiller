import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
import h5py
import numpy as np


class TRECTripleEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a sample
        tr = self.data.iloc[index]
        return {
            'query': tr.query,
            'doc_pos': tr.doc_pos,
            'doc_neg': tr.doc_neg
        }


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
