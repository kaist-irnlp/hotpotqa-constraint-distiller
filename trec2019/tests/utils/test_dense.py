import pytest
from pathlib import Path

import torch.nn.functional as F

from trec2019.utils.dense import *

EMBED_DIR = Path("D:/Sync/Dataset/embedding/word2vec")
EMBED_PATH = EMBED_DIR / "GoogleNews-vectors-negative300.gensim"


@pytest.fixture
def texts():
    texts = [
        "Jerry loves to eat a slice of cheese",
        "Tom got a small piece of pie.",
        "I ate a cake.",
        "Andy loved to sleep on a bed of nails.",
        "Henry loves to rest.",
        "He always wore his sunglasses at night.",
        "Before he moved to the inner city, he had always believed that security complexes were psychological.",
        "She works two jobs to make ends meet; at least, that was her reason for not having time to join us.",
    ]
    return texts


class TestBowEmbedding:
    def test_run(self, texts):
        model = BowEmbedding(EMBED_PATH)
        embeds = model(texts)
        e0 = embeds[0]
        print(e0.shape)
        for e in embeds:
            score = F.cosine_similarity(e0, e, dim=0)
            print(score)
        assert False


# class TestDiscEmbedding:
#     def test_run(self, texts):
#         # TODO: git@github.com:facebookresearch/SentEval.git
#         model = DiscEmbedding(EMBED_PATH, ngram=3)
#         embeds = model(texts)
#         e0 = embeds[0]
#         for e in embeds:
#             score = F.cosine_similarity(e0, e, dim=0)
#             print(score)
#         assert False
