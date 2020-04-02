from textblob import TextBlob
from transformers import BertModel, DistilBertModel, AutoModel
from transformers import BertTokenizer, DistilBertTokenizer, AutoTokenizer
import gensim
from gensim.models.keyedvectors import KeyedVectors
import torch
from torch import nn
from torch import tensor
from torchtext.vocab import Vocab
import gc
import abc
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import Counter
import json
from fse import IndexedList
from fse.models import uSIF, SIF
from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH


FLOAT = torch.float32
root_dir = str(Path(__file__).parent.absolute())


def _get_bow_vocab(self):
    VOCAB_PATH = Path(root_dir) / "../../vocab/vocab.json"
    VECTORS = "fasttext.en.300d"
    MIN_FREQ = 10
    MAX_SIZE = 100000
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_counts = Counter(json.load(f))
        # vocab_counts = Counter(
        #     {row.word: row.count for row in pd.read_parquet(VOCAB_PATH).itertuples()}
        # )
    return Vocab(
        vocab_counts,
        vectors=VECTORS,
        min_freq=MIN_FREQ,
        max_size=MAX_SIZE,
        # unk_init=torch.Tensor.normal_,
    )
    # return SubwordVocab(
    #     vocab_counts,
    #     vectors=VECTORS,
    #     min_freq=MIN_FREQ,
    #     max_size=MAX_SIZE,
    #     # unk_init=torch.Tensor.normal_,
    # )


class BowTokenizer:
    UNK_IDX = 0
    PAD_IDX = 1

    def __init__(self, vocab):
        self.stoi = vocab.stoi.copy()
        self.itos = vocab.itos.copy()
        self.unk_token = self.itos[self.UNK_IDX]
        self.pad_token = self.itos[self.PAD_IDX]

    def encode(self, text, max_length=512):
        tokens = self.tokenize(text)
        padded = self.pad(tokens, max_length)
        ids = self.numericalize(padded)
        return ids

    def tokenize(self, text):
        return [str(tok) for tok in TextBlob(text).lower().tokens]

    def pad(self, tokens, max_length):
        # pad & numericalize
        pad_length = max(0, max_length - len(tokens))
        padded = tokens[:max_length] + [self.pad_token] * pad_length

        return padded

    def numericalize(self, tokens):
        return torch.tensor([self.stoi[x] for x in tokens], dtype=torch.long)


class BertTokenizer:
    def __init__(self, weights):
        self.tokenizer = AutoTokenizer.from_pretrained(weights)

    def encode(self, text, max_length=512):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            pad_to_max_length=True,
            max_length=max_length,
            return_tensors="pt",
            padding_side="right",
        ).squeeze()
        return ids


class FseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def get_dim(self):
        pass

    def forward(self, batch):
        pass


class BowEmbedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(
            vocab.vectors, freeze=True, sparse=True
        )

    def get_dim(self):
        return self.embedding.embedding_dim

    def forward(self, batch):
        embs = self.embedding(batch)
        return embs


class BertEmbedding(nn.Module):
    IDX_CLS = 0

    def __init__(self, weights):
        super().__init__()
        self.model = AutoModel.from_pretrained(weights)

    def forward(self, batch_ids):
        last_hidden_states = self.model(batch_ids)[0]
        return last_hidden_states[:, self.IDX_CLS, :]

    def get_dim(self):
        return self.model.config.hidden_size
