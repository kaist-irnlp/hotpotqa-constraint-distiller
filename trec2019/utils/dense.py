import logging
from abc import ABC, abstractclassmethod, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import pickle

import gensim.downloader as api
import numpy as np
import torch.nn.functional as F
from fse import IndexedList
from fse.models import SIF
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import (
    preprocess_string,
    remove_stopwords,
    split_alphanum,
    stem_text,
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_numeric,
    strip_punctuation,
    strip_short,
    strip_tags,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from textblob import TextBlob

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__file__)

PREPROCESS_FILTERS = [
    lambda x: x.lower(),
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    # strip_numeric,
    remove_stopwords,
    strip_short,
    split_alphanum,
    strip_non_alphanum,
]


class AbstractDense(ABC):
    def __init__(self, args):
        self.args = args
        self._init()

    @abstractmethod
    def _init(self):
        pass

    @abstractclassmethod
    def add_argparse_args(parser):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def encode_batch(self, texts):
        pass

    @property
    def dim(self):
        return self._dim


class DenseTFIDF(AbstractDense):
    def _init(self):
        with open(self.args.model_path, "rb") as f:
            self.model = pickle.load(f)
        self._dim = len(self.model.get_feature_names())

    def _tokenize(self, text):
        return preprocess_string(text, filters=PREPROCESS_FILTERS)

    def add_argparse_args(parser):
        parser.add_argument("--model_path", type=str, required=True)
        return parser

    def encode(self, text):
        return self.model.transform([text]).toarray().squeeze()

    def encode_batch(self, texts):
        return self.model.transform(texts).toarray()


class DenseSIF(AbstractDense):
    def _init(self):
        self.model = SIF.load(self.args.model_path)
        self._dim = self.model.wv.vector_size

    def _tokenize(self, text):
        return preprocess_string(text, filters=PREPROCESS_FILTERS)

    def add_argparse_args(parser):
        parser.add_argument("--model_path", type=str, required=True)
        return parser

    def encode(self, text):
        sent = IndexedList([self._tokenize(text)])
        return self.model.infer(sent).squeeze()

    def encode_batch(self, texts):
        tokens = [self._tokenize(txt) for txt in texts]
        sents = IndexedList(tokens)
        return self.model.infer(sents)


class DenseDisc(AbstractDense):
    def __init__(self, args):
        super().__init__(args)

    def _init(self):
        self.oov_vector = {}
        model_choice = self.args.emb
        model_name = {
            "word2vec": "word2vec-google-news-300",
            "glove": "glove-wiki-gigaword-50",
        }[model_choice]
        logger.info(f"Loading {model_name}...")
        self.model = api.load(model_name)
        self.model.init_sims(replace=True)
        self._dim = self.model.vector_size

    def add_argparse_args(parser):
        parser.add_argument(
            "--emb", type=str, choices=("word2vec", "glove"), default="glove"
        )
        parser.add_argument("--ngram", type=int, default=2)
        return parser

    def _get_randn_vector(self):
        v = np.random.randn(self.dim)
        return self._normalize_vector(v)

    def _normalize_vector(self, vec):
        return vec / np.linalg.norm(vec)

    def _get_vector(self, token):
        if token in self.model.vocab:
            return self.model.get_vector(token)
        else:
            if token not in self.oov_vector:
                self.oov_vector[token] = self._get_randn_vector()
            return self.oov_vector[token]

    def encode(self, text):
        blob = TextBlob(text)
        ngram = self.args.ngram
        out_parts = []
        for n in range(1, ngram + 1):
            ng_vec_all = []
            # collect all ngram vectors
            for ng in blob.ngrams(n=n):
                ng_vec = np.ones(self.model.vector_size)
                for tok in ng:
                    ng_vec *= self._get_vector(tok)
                ng_vec_all.append(self._normalize_vector(ng_vec))
            # normalize ngram vectors
            ng_vec_all = self._normalize_vector(np.sum(ng_vec_all, axis=0))
            out_parts.append(ng_vec_all)
        return np.concatenate(out_parts, axis=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DenseSIF.add_argparse_args(parser)
    args = parser.parse_args()
    # encoder
    encoder = DenseSIF(args)
    encoded = encoder.encode_batch(["this is a good day"] * 3)
    print(encoded)
    print(encoded.shape)
