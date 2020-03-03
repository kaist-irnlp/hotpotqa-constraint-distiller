import torch
import pandas as pd
import gzip
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
import h5py
import numpy as np
from trec2019.utils.encoder import BertEncoder
from transformers import BertTokenizer


class TRECTripleDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a sample
        tr = self.data.iloc[index]
        return tr.to_dict()


class TRECTripleBERTTokenizedDataset(TRECTripleDataset):
    MAX_LENGTH = 512

    def __init__(self, data_path, tokenizer):
        super().__init__(data_path)
        self.tokenizer = tokenizer

    def _tokenize(self, text):
        return torch.tensor(
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.MAX_LENGTH,
                pad_to_max_length="left",
            ),
            dtype=torch.long,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        tokenized = {}
        tokenized["query"] = self._tokenize(sample["query"])
        tokenized["doc_pos"] = self._tokenize(sample["doc_pos"])
        tokenized["doc_neg"] = self._tokenize(sample["doc_neg"])
        return tokenized


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
    data_dir = Path(
        "/Users/kyoungrok/Resilio Sync/Dataset/2019 TREC/passage_ranking/dataset"
    )
    fpath = data_dir / "valid.parquet"
    dset = TRECTripleBERTTokenizedDataset(fpath)
    loader = DataLoader(dset, batch_size=32, num_workers=2)
    for batch in loader:
        res = batch["doc_pos"]
        print(res, len(res))
        break
