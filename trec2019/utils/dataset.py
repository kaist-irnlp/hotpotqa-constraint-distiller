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
    def __init__(self, data_path, model="fse"):
        super().__init__()
        self.data = zarr.open(data_path, "r")
        self.embedding = self.data.embedding[model]
        self.label = self.data.label[:]

    def get_dim(self):
        return self.embedding[0].shape[0]

    def __len__(self):
        return len(self.data.label)

    def __getitem__(self, index):
        emb, lbl = self.embedding[index], self.label[index]
        return {"data": emb, "target": lbl}


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
        return {"data": text_ids, "target": row.label}


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
        return {"query": query, "doc_pos": doc_pos, "doc_neg": doc_neg}


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
    # vocab_path = Path(root_dir) / "../vocab/vocab.json.gz"
    # with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
    #     vocab_counts = Counter(json.load(f))
    # vocab = Vocab(
    #     vocab_counts, vectors="fasttext.en.300d", min_freq=10, max_size=100000,
    # )
    # tokenizer = BowTokenizer(vocab)
    # data
    data_path = "/Users/kyoungrok/Dropbox/Project/naver/data/news20/test.zarr.zip"
    dataset = News20EmbeddingDataset(data_path)
    # test
    loader = DataLoader(dataset, batch_size=32)
    for i, sample in enumerate(loader):
        print(sample)
        if i > 3:
            break
