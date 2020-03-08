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

DTYPE_FLOAT = torch.float32


class BaseEmbedding(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_dim(self):
        pass


class BertEmbedding(nn.Module):
    def __init__(self, weights="bert-base-uncased", max_length=512):
        self.max_length = max_length
        self.model = BertModel.from_pretrained(self.BERT_WEIGHTS)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_WEIGHTS)

    def forward(self, batch_text):
        batch_token_ids = self.tokenizer.batch_encode_plus(
            batch_text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length="left",
            return_tensors="pt",
        )["input_ids"]

        with torch.no_grad():
            last_hidden_states = self.model(batch_token_ids)[0]

        IDX_CLS = 0
        return last_hidden_states[:, IDX_CLS, :]

    def get_dim(self):
        return self.model.config.hidden_size


class BowEmbedding(nn.Module):
    def __init__(self, embedding_path):
        super().__init__()
        self.embedding_path = str(embedding_path)
        # modules
        self.embeddings = None
        self.word2idx = {}
        self.idx2word = []
        # init
        self._load_embeddings()

    def forward(self, batch_text):
        batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
        return batch_embeddings

    def get_dim(self):
        return self.emb_dim

    def _embed(self, text):
        blob = TextBlob(text).lower()
        ids = [self.word2idx.get(w, -1) for w in blob.tokens]
        ids = tensor([i for i in ids if i != -1], dtype=torch.long)
        embs = self.embeddings(ids)
        return torch.mean(embs, 0)

    def _load_embeddings(self):
        # load
        model = KeyedVectors.load(self.embedding_path)
        # save
        self.embeddings = nn.Embedding.from_pretrained(
            tensor(model.vectors, dtype=torch.float32), freeze=True
        )
        self.vocab_size = self.embeddings.num_embeddings
        self.emb_dim = self.embeddings.embedding_dim
        self.word2idx = {w: idx for (idx, w) in enumerate(model.index2word)}
        self.idx2word = model.index2word
        # clean
        del model
        gc.collect()


class DiscEmbedding(nn.Module):
    def __init__(self, embedding_path, ngram=3, normalize=True):
        super().__init__()
        # params
        self.embedding_path = str(embedding_path)
        self.ngram = ngram
        self.normalize = normalize
        # modules
        self.embeddings = None
        self.word2idx = {}
        self.idx2word = []
        # init
        self._load_embeddings()

    def forward(self, batch_text):
        batch_embeddings = torch.stack([self._embed(text) for text in batch_text])
        return batch_embeddings
        # with self.nlp.disable_pipes("tagger", "parser"):
        #     docs = self.nlp.pipe(batch_text)
        # batch_token_ids = [[self.word2idx(w) for w in doc] for doc in docs]
        # DO n-gram embedding lookup for each sample (row)
        # embeddings = self.embeddings(docs).view(1, -1)

    def get_dim(self):
        return self.emb_dim

    def _embed(self, text):
        blob = TextBlob(text).lower()
        out_dim = self.emb_dim * self.ngram
        out = torch.zeros(out_dim)
        for n in range(1, self.ngram + 1):
            ngrams = blob.ngrams(n=n)
            ngrams_emb = torch.zeros(self.emb_dim)
            for i, ng in enumerate(ngrams):
                ng_ids = [self.word2idx.get(w, -1) for w in ng]
                ng_ids = tensor([i for i in ng_ids if i != -1], dtype=torch.long)
                ng_embs = self.embeddings(ng_ids)
                ngrams_emb += torch.prod(ng_embs, 0)
            # out[ngram_range] = ngrams_emb
            out[self.emb_dim * (n - 1) : self.emb_dim * n] = ngrams_emb / (i + 1)
        return out

    def _init_tokenizer(self):
        self.nlp = spacy.load("en")

    def _load_embeddings(self):
        # load
        model = KeyedVectors.load(self.embedding_path)
        # save
        self.embeddings = nn.Embedding.from_pretrained(
            tensor(model.vectors, dtype=torch.float32), freeze=True
        )
        self.vocab_size = self.embeddings.num_embeddings
        self.emb_dim = self.embeddings.embedding_dim
        self.word2idx = {w: idx for (idx, w) in enumerate(model.index2word)}
        self.idx2word = model.index2word
        # clean
        del model
        gc.collect()

