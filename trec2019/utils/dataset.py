import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
import h5py
import numpy as np
from trec2019.utils.encoder import BertEncoder


class TRECTripleDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a sample
        tr = self.data.iloc[index]
        return {
            "query": tr["query"],
            "doc_pos": tr["doc_pos"],
            "doc_neg": tr["doc_neg"],
        }


class TRECTripleEmbeddingDataset(TRECTripleDataset):
    def __init__(self, data_path, encoder):
        super().__init__(data_path)
        self.encoder = encoder

    def __getitem__(self, index):
        # return a sample
        sample = super().__getitem__(index)
        for fld in ["query", "doc_pos", "doc_neg"]:
            sample[fld] = self.encoder.encode(sample[fld])
        return sample

    def get_dim(self):
        return self.encoder.get_dim()


class TRECTripleBERTDataset(TRECTripleEmbeddingDataset):
    def __init__(self, data_path):
        encoder = BertEncoder()
        super().__init__(data_path, encoder)


class TRECTripleIdDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep="\t", names=["q_id", "pos_id", "neg_id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()


if __name__ == "__main__":
    encoder = BertEncoder()
    fpath = "/Users/kyoungrok/Resilio Sync/Dataset/2019 TREC/passage_ranking/triples.train.tiny.parquet"
    dset = TRECTripleEmbeddingDataset(fpath, encoder)
    # dset = TRECTripleDataset(fpath)
    for d in dset:
        print(d)
        break

