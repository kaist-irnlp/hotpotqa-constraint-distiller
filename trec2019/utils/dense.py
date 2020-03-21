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


FLOAT = torch.float32


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
        self.tokenizer = AutoModel.from_pretrained(weights)

    def encode(self, text, max_length=512):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            pad_to_max_length=True,
            max_length=max_length,
        )
        return ids


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
        self.weights = weights

    def forward(self, batch_ids):
        with torch.no_grad():
            last_hidden_states = self.model(batch_ids)[0]

        return last_hidden_states[:, self.IDX_CLS, :]

    def get_dim(self):
        return self.model.config.hidden_size


# class OldBowEmbedding(BasePretrainedEmbedding):
#     def __init__(self, embedding_path):
#         super().__init__(embedding_path)

#     def forward(self, batch_text):
#         batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
#         return batch_embeddings

#     def get_dim(self):
#         return self.emb_dim

#     def _tokenize(self, text):
#         return list(TextBlob(text).lower().tokens)

#     def _embed(self, text):
#         tokens = self._tokenize(text)
#         # ids = [self.word2idx.get(w, -1) for w in tokens]
#         # ids = (
#         #     tensor([i for i in ids if i >= 0]).long().to(self.embeddings.weight.device)
#         # )
#         embs = self.get_embeddings(tokens)
#         embs = torch.mean(embs, 0)
#         return embs
