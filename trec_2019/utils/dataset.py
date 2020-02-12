import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset


class TRECTripleEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep='\t', names=[
                                'q_id', 'pos_id', 'neg_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()


class TRECTripleTextIterableDataset(IterableDataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            self.length = sum(1 for line in f)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
        else:


class TRECTripleIdDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep='\t', names=[
                                'q_id', 'pos_id', 'neg_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
