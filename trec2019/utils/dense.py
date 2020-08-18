from abc import ABC, abstractclassmethod, abstractmethod
from textblob import TextBlob
from gensim.models import KeyedVectors
import numpy as np
import torch.nn.functional as F
import gensim.downloader as api
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__file__)


class AbstractDense(ABC):
    def __init__(self, args):
        self.args = args
        self.__dim = None
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

    @property
    def dim(self):
        return self.__dim


class DenseDisc(AbstractDense):
    def __init__(self, args):
        super().__init__(args)

    def _init(self):
        self.oov_vector = {}
        self._init_embedding()

    def _init_embedding(self):
        model_choice = self.args.emb_model
        model_name = {
            "word2vec": "word2vec-google-news-300",
            "glove": "glove-wiki-gigaword-300",
        }[model_choice]
        logger.info(f"Loading {model_name}...")
        self.model = api.load(model_name)
        self.vector = normalize(self.model.vectors, axis=1)
        self.vocab = self.model.vocab
        self.__dim = self.vector.shape[1]

    def add_argparse_args(parser):
        parser.add_argument(
            "--emb_model", type=str, choices=("word2vec", "glove"), default="glove"
        )
        parser.add_argument("--ngram", type=int, default=3)
        return parser

    def get_rand_vector(self):
        v = np.random.rand(self.dim)
        v = v / (v ** 2).sum() ** 0.5
        return v

    def get_vector(self, token):
        if token in self.vocab:
            return self.vector[token]
        else:
            if token not in self.oov_vector:
                self.oov_vector[token] = self.get_rand_vector()
            return self.oov_vector[token]

    def encode(self, text):
        vector_parts = []
        blob = TextBlob(text)
        for n in range(1, self.args.ngram + 1):
            vec_part = np.zeros(self.dim)
            i = 0
            for i, ng in enumerate(blob.ngrams(n=n)):
                vec_ng = np.ones(self.dim)
                for tok in ng:
                    vec_ng *= self.get_vector(tok)
                vec_part += vec_ng
            vector_parts.append(vec_part / (i + 1))

        return np.concatenate(vector_parts, axis=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DenseDisc.add_argparse_args(parser)
    args = parser.parse_args()
    # encoder
    encoder = DenseDisc(args)
    encoded = encoder.encode("this is a good day")
    print(encoded)
    print(encoded.shape)
