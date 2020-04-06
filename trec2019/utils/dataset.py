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
from numcodecs import blosc
from torchtext.vocab import Vocab
import torchtext
from collections import Counter
import json
from textblob import TextBlob
from transformers.tokenization_auto import AutoTokenizer
from trec2019.utils.dense import *

root_dir = str(Path(__file__).parent.absolute())


class News20EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = zarr.open(data_path, "r")

    def get_dim(self):
        return self.data[0].shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {"emb": row.emb, "target": row.label}


class News20Dataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text_ids = self.tokenizer.encode(row.text)
        return {"text": text_ids, "target": row.label}


class TripleEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = zarr.open(data_path, "r")

    def get_dim(self):
        return self.data[0].shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get a sample
        query, doc_pos, doc_neg = self.data[index]
        return {"query_emb": query, "doc_pos_emb": doc_pos, "doc_neg_emb": doc_neg}


class TripleDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        # synchronizer = zarr.ProcessSynchronizer("./sync/triple_dataset.sync")
        self.data = zarr.open(data_path, "r")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get a sample
        query, doc_pos, doc_neg = self.data[index]
        query_ids = self.tokenizer.encode(query, 128)
        doc_pos_ids = self.tokenizer.encode(doc_pos)
        doc_neg_ids = self.tokenizer.encode(doc_neg)
        return {"query": query_ids, "doc_pos": doc_pos_ids, "doc_neg": doc_neg_ids}


if __name__ == "__main__":
    # tokenizer
    vocab_path = Path(root_dir) / "../vocab/vocab.json.gz"
    with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
        vocab_counts = Counter(json.load(f))
    vocab = Vocab(
        vocab_counts, vectors="fasttext.en.300d", min_freq=10, max_size=100000,
    )
    tokenizer = BowTokenizer(vocab)
    # data
    data_path = "D:/Dropbox/Project/naver/data/20newsgroup/train.parquet"
    dataset = News20Dataset(data_path, tokenizer)
    # test
    loader = DataLoader(dataset, batch_size=2)
    for i, sample in enumerate(loader):
        print(sample)
        break
