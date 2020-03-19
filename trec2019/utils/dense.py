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
IDX_CLS = 0


class BowEmbedding(nn.Module):
    def __init__(self, vocab_path, vectors="fasttext.en.300d"):
        super().__init__()
        self._init_embedding(vocab_path, vectors)
        # self.device = self.embeddings.weight.device

    def _init_embedding(self, vocab_path, vectors):
        self.vocab = self._load_vocab(vocab_path)
        self.vocab.load_vectors(vectors)
        self.embeddings = nn.EmbeddingBag.from_pretrained(
            self.vocab.vectors, freeze=True, sparse=True
        )

    def _load_vocab(self, vocab_path):
        """load vocab as dictionary
        """
        vocab_counts = pd.read_parquet(vocab_path).as_dict()
        return Vocab(vocab_counts)

    def get_dim(self):
        return self.vocab.vectors.shape[1]

    def forward(self, batch_text):
        batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
        return batch_embeddings

    def get_ids(self, text):
        return list(TextBlob(text).lower().tokens)

    def _embed(self, text):
        tokens = self.get_ids(text)
        # ids = [self.word2idx.get(w, -1) for w in tokens]
        # ids = (
        #     tensor([i for i in ids if i >= 0]).long().to(self.embeddings.weight.device)
        # )
        embs = self.get_embeddings(tokens)
        embs = torch.mean(embs, 0)
        return embs


class BertEmbedding(nn.Module):
    def __init__(self, weights="bert-base-uncased", max_length=512):
        super().__init__()
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(weights)
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        self.weights = weights

    def forward(self, batch_text):
        batch_token_ids = (
            self.tokenizer.batch_encode_plus(
                batch_text,
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length="left",
                return_tensors="pt",
            )["input_ids"]
            .long()
            .to(next(self.model.parameters()).device)
        )

        with torch.no_grad():
            last_hidden_states = self.model(batch_token_ids)[0]

        return last_hidden_states[:, IDX_CLS, :]

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
