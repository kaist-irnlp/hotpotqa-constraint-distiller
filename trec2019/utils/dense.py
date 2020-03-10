from textblob import TextBlob
from transformers import BertModel
from transformers import BertTokenizer
import gensim
from gensim.models.keyedvectors import KeyedVectors
import torch
from torch import nn
from torch import tensor
import gc
import abc
import numpy as np
import sys

FLOAT = torch.float32


class BertEmbedding(nn.Module):
    def __init__(self, weights="bert-base-uncased", max_length=512):
        super().__init__()
        self.max_length = max_length
        self.model = BertModel.from_pretrained(weights)
        self.tokenizer = BertTokenizer.from_pretrained(weights)
        self.device = next(self.parameters()).device

    def forward(self, batch_text):
        batch_token_ids = self.tokenizer.batch_encode_plus(
            batch_text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length="left",
            return_tensors="pt",
        )["input_ids"].to(self.device)

        with torch.no_grad():
            last_hidden_states = self.model(batch_token_ids)[0]

        IDX_CLS = 0
        return last_hidden_states[:, IDX_CLS, :]

    def get_dim(self):
        return self.model.config.hidden_size


class BasePretrainedEmbedding(nn.Module):
    def __init__(self, embedding_path):
        super().__init__()
        self.embedding_path = str(embedding_path)
        # modules
        self.embeddings = None
        self.word2idx = {}
        self.idx2word = []
        # init
        self._init_embeddings()
        # self.device = self.embeddings.weight.device

    def get_dim(self):
        raise NotImplementedError

    def get_embeddings(self, tokens):
        ids = (
            tensor([self.word2idx.get(w, -1) for w in tokens])
            .long()
            .to(self.embeddings.weight.device)
        )
        ids = ids[(ids != -1).nonzero(as_tuple=True)]
        embeddings = self.embeddings(ids)
        return (
            embeddings
            if len(embeddings) > 0
            else torch.randn(1, self.emb_dim).to(self.embeddings.weight.device)
        )

    def _tokenize(self, text):
        raise NotImplementedError

    def _embed(self, text):
        raise NotImplementedError

    def _init_embeddings(self):
        # load
        model = KeyedVectors.load(self.embedding_path)
        # save
        self.embeddings = nn.Embedding.from_pretrained(
            tensor(model.vectors, dtype=FLOAT), freeze=True
        )
        self.vocab_size = self.embeddings.num_embeddings
        self.emb_dim = self.embeddings.embedding_dim
        self.word2idx = {w: idx for (idx, w) in enumerate(model.index2word)}
        self.idx2word = model.index2word
        # clean
        del model
        gc.collect()


class BowEmbedding(BasePretrainedEmbedding):
    def __init__(self, embedding_path):
        super().__init__(embedding_path)

    def forward(self, batch_text):
        batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
        return batch_embeddings

    def get_dim(self):
        return self.emb_dim

    def _tokenize(self, text):
        return list(TextBlob(text).lower().tokens)

    def _embed(self, text):
        tokens = self._tokenize(text)
        # ids = [self.word2idx.get(w, -1) for w in tokens]
        # ids = (
        #     tensor([i for i in ids if i >= 0]).long().to(self.embeddings.weight.device)
        # )
        embs = self.get_embeddings(tokens)
        embs = torch.mean(embs, 0)
        return embs


class DiscEmbedding(BasePretrainedEmbedding):
    def __init__(self, embedding_path, ngram=3, normalize=True):
        super().__init__(embedding_path)
        self.ngram = ngram
        self.normalize = normalize

    def forward(self, batch_text):
        batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
        return batch_embeddings
        # with self.nlp.disable_pipes("tagger", "parser"):
        #     docs = self.nlp.pipe(batch_text)
        # batch_token_ids = [[self.word2idx(w) for w in doc] for doc in docs]
        # DO n-gram embedding lookup for each sample (row)
        # embeddings = self.embeddings(docs).view(1, -1)

    def get_dim(self):
        return self.emb_dim * self.ngram

    def _embed(self, text):
        blob = TextBlob(text).lower()
        out_dim = self.get_dim()
        out = torch.zeros(out_dim).to(self.embeddings.weight.device)
        scaling = np.sqrt(self.emb_dim)

        for n in range(1, self.ngram + 1):
            ngrams = blob.ngrams(n=n)
            ngrams_emb = torch.zeros(self.emb_dim).to(self.embeddings.weight.device)
            for i, ng in enumerate(ngrams):
                ng_embs = self.get_embeddings(ng)
                ng_embs_reduced = torch.prod(ng_embs, 0) * (
                    scaling ** (ng_embs.shape[0] - 1)
                )
                ngrams_emb += ng_embs_reduced
            # out[ngram_range] = ngrams_emb
            out[self.emb_dim * (n - 1) : self.emb_dim * n] = ngrams_emb / n
        return out
