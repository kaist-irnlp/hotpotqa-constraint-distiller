import torch
import pandas as pd
import gzip
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
import h5py
import numpy as np
from transformers import BertTokenizer
import gc
import zarr
from torchtext.vocab import Vocab
from torchtext.data import Field, BucketIterator, Example
import torchtext

TEXT = Field(sequential=True, tokenize="spacy", lower=True)


class TripleDataset(torchtext.data.Dataset):
    def __init__(self, data_path):
        fields = [
            ("query", TEXT),
            ("doc_pos", TEXT),
            ("doc_neg", TEXT),
        ]
        data = zarr.open(data_path, "r")
        super().__init__(data, fields)


class TRECTripleDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = zarr.open(data_path, "r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a sample
        query, doc_pos, doc_neg = self.data[index]
        # sample["index"] = index
        # for k, v in sample.items():
        #     blob = TextBlob(v)
        #     if self.lower:
        #         blob = blob.lower()
        #     sample[k] = np.array(
        #         list(
        #             filter(
        #                 lambda i: i >= 0,
        #                 [self.word2idx.get(w, -1) for w in blob.tokens],
        #             )
        #         ),
        #         dtype=np.int32,
        #     )
        return {"query": query, "doc_pos": doc_pos, "doc_neg": doc_neg}


if __name__ == "__main__":
    data_path = Path(
        "/Users/kyoungrok/Resilio Sync/Dataset/2019 TREC/passage_ranking/triples.train.small.tsv.zarr.zip"
    )
    dataset = TripleDataset(str(data_path))
    train, val, test = dataset.splits(
        stratified=True, strata_field="query", random_state=2020
    )
    # loader = DataLoader(dset, batch_size=32, num_workers=2)
    # for batch in loader:
    #     res = batch["doc_pos"]
    #     print(res, len(res))
    #     break
