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
import torchtext
from collections import Counter
from textblob import TextBlob
from transformers.tokenization_auto import AutoTokenizer
from trec2019.utils.dense import *


class TripleDataset(Dataset):
    def __init__(self, data_path, indices, tokenizer):
        super().__init__()
        synchronizer = zarr.ProcessSynchronizer("sync/triple_dataset.sync")
        self.data = zarr.open(data_path, "r", synchronizer=synchronizer)
        self.indices = indices
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # get a sample
        index = self.indices[index]
        query, doc_pos, doc_neg = self.data[index]
        query_ids = self.tokenizer.encode(query, 128)
        doc_pos_ids = self.tokenizer.encode(doc_pos)
        doc_neg_ids = self.tokenizer.encode(doc_neg)
        return {"query": query_ids, "doc_pos": doc_pos_ids, "doc_neg": doc_neg_ids}


def load_vocab_counts(vocab_count_path):
    df = pd.read_parquet(vocab_count_path)
    return Counter({row.word: row.count for row in df.itertuples()})


if __name__ == "__main__":
    # tokenizer
    weights = "bert-base-uncased"
    tokenizer = BertTokenizer(weights=weights)
    # data
    data_path = "C:/Users/kyoun/Desktop/triples.train.small.tsv.zarr.zip"
    data = zarr.open(data_path, "r")
    indices = np.array(range(len(data)))
    dataset = TripleDataset(data_path, indices, tokenizer)
    # test
    loader = DataLoader(dataset, batch_size=2)
    for i, sample in enumerate(loader):
        print(sample)
        break
