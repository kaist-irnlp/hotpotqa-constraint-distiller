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

# from textblob import TextBlob
# from gensim.corpora import Dictionary

# from transformers import BertTokenizer
# from transformers.tokenization_auto import AutoTokenizer
from trec2019.utils.dense import *

_root_dir = str(Path(__file__).parent.absolute())
blosc.use_threads = False


class AbstractNoisyDataset(Dataset):
    def __init__(self, noise=None, noise_ratio=0.0):
        self._add_noise = {
            "gaussian": self.add_gaussian_noise,
            "masking": self.add_masking_noise,
            "salt": self.add_salt_pepper_noise,
        }.get(noise, None)
        self._noise_ratio = noise_ratio

    def add_gaussian_noise(self, X, range_=[0, 1]):
        X_noisy = X + self._noise_ratio * np.random.normal(
            loc=0.0, scale=1.0, size=X.shape
        )
        X_noisy = np.clip(X_noisy, range_[0], range_[1])

        return X_noisy

    def add_masking_noise(self, X):
        # TODO: non-zero만 골라서 제거하도록 구현
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = X.shape
        n = int(ncol * self._noise_ratio)
        for i in range(nrow):
            idx_noisy = np.random.choice(ncol, n, replace=False)
            X_noisy[i, idx_noisy] = 0

        return X_noisy

    def add_salt_pepper_noise(self, X):
        assert self._noise_ratio >= 0 and self._noise_ratio <= 1
        X_noisy = np.copy(X)
        nrow, ncol = X.shape
        n = int(ncol * self._noise_ratio)
        for i in range(nrow):
            idx_noisy = np.random.choice(ncol, n, replace=False)
            X_noisy[i, idx_noisy] = np.random.binomial(1, 0.5, n)

        return X_noisy


class TripleEmbeddingDataset(AbstractNoisyDataset):
    def __init__(self, data_dir, emb_path, dset_type, noise=None, noise_ratio=0.0):
        super().__init__(noise=noise, noise_ratio=noise_ratio)
        self.data_dir = Path(data_dir)
        self.emb_path = str(emb_path)
        self.dset_type = dset_type
        self._load_data()
        self._build_indexer()

    def _build_indexer(self):
        """"""
        target_query_ids = set(self.triples[:, 0].tolist())
        target_doc_ids = set(self.triples[:, 1].tolist() + self.triples[:, 2].tolist())

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
        )
        self.queries = zarr.open(
            str(self.data_dir / f"queries.{self.dset_type}.zarr"), "r"
        )
        self.queries_emb = self.queries[self.emb_path]
        self.docs = zarr.open(str(self.data_dir / f"docs.{self.dset_type}.zarr"), "r")
        self.docs_emb = self.docs[self.emb_path]

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


class EmbeddingDataset(AbstractNoisyDataset):
    def __init__(self, data_path, emb_path, noise=None, noise_ratio=0.0):
        super().__init__(noise=noise, noise_ratio=noise_ratio)
        self.data_path = data_path
        self.emb_path = emb_path
        self._load_data()

    def _load_data(self):
        data = zarr.open(str(self.data_path), "r")
        self.embedding = data[self.emb_path]

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        data = self.embedding[index].astype("f4")
        orig_data = np.copy(data)
        # add noise
        if self._add_noise:
            data = self._add_noise(data)
        return {
            "index": index,
            "data": data,
            "orig_data": orig_data,
        }


class EmbeddingLabelDataset(EmbeddingDataset):
    def __init__(self, data_path, emb_path, noise=None, noise_ratio=0.0):
        super().__init__(data_path, emb_path, noise=noise, noise_ratio=noise_ratio)

    def _load_data(self):
        super()._load_data()
        data = zarr.open(str(self.data_path), "r")
        self.label = data.label[:]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["target"] = self.label[index].astype("i8")
        return item


# class EmbeddingLabelDataset(Dataset):
#     def __init__(self, data_path, arr_path):
#         super().__init__()
#         self.data_path = Path(data_path)
#         self.data = None
#         self.arr_path = arr_path
#         data_name = f"{self.data_path.parent.stem}_{self.data_path.stem}"
#         self.sync = zarr.ProcessSynchronizer(f"sync/{data_name}.sync")

#     def _load_data(self):
#         self.data = zarr.open(str(self.data_path), "r", synchronizer=self.sync)
#         self.embedding = self.data[self.arr_path]
#         self.label = self.data.label[:]

#     def __len__(self):
#         data = zarr.open(str(self.data_path), "r", synchronizer=self.sync)
#         return len(data.label)

#     def __getitem__(self, index):
#         if self.data is None:
#             self._load_data()

#         emb, lbl = (
#             self.embedding[index].astype(np.float32),
#             self.label[index],
#         )
#         return {"index": index, "data": emb, "target": lbl}


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


# class TripleDataset(Dataset):
#     def __init__(self, data_path, tokenizer):
#         super().__init__()
#         # synchronizer = zarr.ProcessSynchronizer("./sync/triple_dataset.sync")
#         self.data = zarr.open(data_path, "r")
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         # get a sample
#         query, doc_pos, doc_neg = self.data[index]
#         query_ids = self.tokenizer.encode(query, 128)
#         doc_pos_ids = self.tokenizer.encode(doc_pos)
#         doc_neg_ids = self.tokenizer.encode(doc_neg)
#         return {"query": query_ids, "doc_pos": doc_pos_ids, "doc_neg": doc_neg_ids}


if __name__ == "__main__":
    # data
    data_dir = r"E:\msmarco-passages"
    dataset = TripleDataset(data_dir, "dense/fse_Average")
    # test
    loader = DataLoader(dataset, batch_size=128)
    for i, sample in enumerate(loader):
        print(sample["q"].shape)
        if i > 3:
            break
