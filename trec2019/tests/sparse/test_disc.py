import pytest
from pathlib import Path

import torch.nn.functional as F

from trec2019.model.sparsenet.sparsenet import DiscEmbedding

EMBED_DIR = Path("D:/Sync/Dataset/embedding/word2vec")
EMBED_PATH = EMBED_DIR / "GoogleNews-vectors-negative300.gensim"


class TestDiscEmbedding:
    def test_run(self):
        # TODO: git@github.com:facebookresearch/SentEval.git
        model = DiscEmbedding(EMBED_PATH, ngram=3)
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
        embeds = model(texts)
        e0 = embeds[0]
        for e in embeds:
            score = F.cosine_similarity(e0, e, dim=0)
            print(score)
        assert False
