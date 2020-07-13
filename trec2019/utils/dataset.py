import torch
import pandas as pd
import gzip
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
import h5py
import numpy as np
import gc
import zarr
from numcodecs import blosc
from torchtext.vocab import Vocab
import torchtext
from collections import Counter
import json
from sklearn.preprocessing import normalize

# from textblob import TextBlob
# from gensim.corpora import Dictionary

# from transformers import BertTokenizer
# from transformers.tokenization_auto import AutoTokenizer
from trec2019.utils.dense import *

_root_dir = str(Path(__file__).parent.absolute())
blosc.use_threads = False


class AbstractNoisyDataset(Dataset):
    def __init__(self, noise_ratio=0.0, on_memory=False):
        self._noise_funcs = {
            "gaussian": self.add_gaussian_noise,
            "masking": self.add_masking_noise,
            "salt": self.add_salt_pepper_noise,
        }
        self._noise_ratio = noise_ratio
        self.on_memory = on_memory

    def add_gaussian_noise(self, X, range_=[0, 1]):
        X_noisy = X + self._noise_ratio * np.random.normal(
            loc=0.0, scale=1.0, size=X.shape
        )
        X_noisy = np.clip(X_noisy, range_[0], range_[1])

        return X_noisy

    def add_masking_noise(self, X):
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = 1, X.shape[0]
        n = int(ncol * self._noise_ratio)
        idx_noisy = np.random.choice(ncol, n, replace=False)
        X_noisy[idx_noisy] = 0

        return X_noisy

    def add_salt_pepper_noise(self, X):
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = 1, X.shape[0]
        n = int(ncol * self._noise_ratio)
        idx_noisy = np.random.choice(ncol, n, replace=False)
        X_noisy[idx_noisy] = np.random.binomial(1, 0.5, n)

        return X_noisy


class EmbeddingDataset(AbstractNoisyDataset):
    def __init__(
        self, data_path, emb_path, noise_ratio=0.0, on_memory=False,
    ):
        super().__init__(noise_ratio=noise_ratio, on_memory=on_memory)
        self.data_path = data_path
        self.emb_path = emb_path
        self._load_data()

    def _load_data(self):
        data = zarr.open(str(self.data_path), "r")
        self.embedding = data[self.emb_path]
        if self.on_memory:
            self.embedding = self.embedding[:]

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        data = self.embedding[index]
        # generate noised version (nbs, n_view, ...)
        data = np.stack(
            [data] + [noise_f(data) for noise_f in self._noise_funcs.values()], axis=1
        )
        return {
            "index": index,
            "data": data.astype("f4"),
        }


class EmbeddingLabelDataset(EmbeddingDataset):
    def __init__(
        self, data_path, emb_path, noise_ratio=0.0, on_memory=False,
    ):
        super().__init__(
            data_path, emb_path, noise_ratio=noise_ratio, on_memory=on_memory,
        )

    def _load_data(self):
        super()._load_data()
        data = zarr.open(str(self.data_path), "r")
        self.label = data.label[:]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["target"] = self.label[index].astype("i8")
        return item


class TripleEmbeddingDataset(AbstractNoisyDataset):
    def __init__(
        self, data_dir, emb_path, dset_type, noise_ratio=0.0, on_memory=False,
    ):
        super().__init__(noise_ratio=noise_ratio, on_memory=on_memory)
        self.data_dir = Path(data_dir)
        self.emb_path = str(emb_path)
        self.dset_type = dset_type
        self._load_data()
        self._build_indexer()

    def _build_indexer(self):
        """"""
        target_query_ids = set(self.triples[:, 0].tolist())
        target_doc_ids = set(self.triples[:, 1:].flatten().tolist())

        self.idx_queries = {
            u_id: seq_id
            for (seq_id, u_id) in enumerate(self.queries.id)
            if u_id in target_query_ids
        }
        self.idx_docs = {
            u_id: seq_id
            for (seq_id, u_id) in enumerate(self.docs.id)
            if u_id in target_doc_ids
        }

    def _load_data(self):
        self.triples = zarr.open(
            str(self.data_dir / f"triples.{self.dset_type}.zarr"), "r"
        )[:]
        self.queries = zarr.open(
            str(self.data_dir / f"queries.{self.dset_type}.zarr"), "r"
        )
        self.docs = zarr.open(str(self.data_dir / f"docs.{self.dset_type}.zarr"), "r")
        self.queries_emb = self.queries[self.emb_path]
        self.docs_emb = self.docs[self.emb_path]

        if self.on_memory:
            # self.triples = self.triples[:] # triples are always on memory
            self.queries_emb = self.queries_emb[:]
            self.docs_emb = self.docs_emb[:]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        q_id, pos_id, neg_id = self.triples[index]
        q = orig_q = self.queries_emb[self.idx_queries[q_id]]
        pos = orig_pos = self.docs_emb[self.idx_docs[pos_id]]
        neg = orig_neg = self.docs_emb[self.idx_docs[neg_id]]

        # add noise
        if self._add_noise:
            q = self._add_noise(q)
            pos = self._add_noise(pos)
            neg = self._add_noise(neg)

        # Float
        q = q.astype("f4")
        pos = pos.astype("f4")
        neg = neg.astype("f4")
        orig_q = orig_q.astype("f4")
        orig_pos = orig_pos.astype("f4")
        orig_neg = orig_neg.astype("f4")

        # return
        return {
            "index": index,
            "q": q,
            "pos": pos,
            "neg": neg,
            "orig_q": orig_q,
            "orig_pos": orig_pos,
            "orig_neg": orig_neg,
        }


if __name__ == "__main__":
    # data
    data_dir = "D:/Data/news20/test.zarr"
    dataset = EmbeddingDataset(data_dir, "dense/bert-base-cased")
    # test
    loader = DataLoader(dataset, batch_size=32)
    for i, batch in enumerate(loader):
        print(batch["data"].shape)
        if i > 3:
            break
