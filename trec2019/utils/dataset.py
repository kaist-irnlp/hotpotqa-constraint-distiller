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


class TripleDataset(Dataset):
    UNK_IDX = 0
    PAD_IDX = 1

    def __init__(self, data_path, vocab, max_length=512):
        super().__init__()
        self.data = zarr.open(str(data_path), "r")
        self.vocab = vocab
        self.max_length = max_length
        self.unk_token = self.vocab.itos[self.UNK_IDX]
        self.pad_token = self.vocab.itos[self.PAD_IDX]

    def numericalize(self, arr):
        return torch.tensor([self.vocab.stoi[x] for x in arr], dtype=torch.long)
        # arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
        # var = torch.tensor(arr, dtype=torch.long).contiguous()
        # return var

    def pad(self, x):
        # Do real-time tokenization here
        x = [str(tok) for tok in TextBlob(x).lower().tokens]

        # pad & numericalize
        pad_length = max(0, self.max_length - len(x))
        padded = x[: self.max_length] + [self.pad_token] * pad_length

        return padded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a sample
        query, doc_pos, doc_neg = self.data[index]
        query, doc_pos, doc_neg = self.pad(query), self.pad(doc_pos), self.pad(doc_neg)
        query_ids, doc_pos_ids, doc_neg_ids = (
            self.numericalize(query),
            self.numericalize(doc_pos),
            self.numericalize(doc_neg),
        )
        return {"query": query_ids, "doc_pos": doc_pos_ids, "doc_neg": doc_neg_ids}


def load_vocab_counts(vocab_count_path):
    df = pd.read_parquet(vocab_count_path)
    return Counter({row.word: row.count for row in df.itertuples()})


if __name__ == "__main__":
    counts = load_vocab_counts("vocab/vocab.parquet")
    vocab = Vocab(counts, vectors="fasttext.simple.300d", min_freq=2,)
    data_path = Path(
        "/Users/kyoungrok/Resilio Sync/Dataset/2019 TREC/passage_ranking/triples.train.small.tsv.zarr.zip"
    )
    dataset = TripleDataset(data_path, vocab)
    loader = DataLoader(dataset, batch_size=2)
    for i, sample in enumerate(loader):
        print(sample)
        break
