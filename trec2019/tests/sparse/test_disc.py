import pytest
from pathlib import Path

from trec2019.model.sparsenet.sparsenet import DiscEmbedding

EMBED_DIR = Path("D:/Sync/Dataset/embedding/word2vec")
EMBED_PATH = EMBED_DIR / "GoogleNews-vectors-negative300.gensim"


class TestDiscEmbedding:
    def test_run(self):
        model = DiscEmbedding(EMBED_PATH, ngram=3)
        texts = [
            "Tom got a small piece of pie.",
            "Andy loved to sleep on a bed of nails.",
            "He always wore his sunglasses at night.",
            "Before he moved to the inner city, he had always believed that security complexes were psychological.",
            "She works two jobs to make ends meet; at least, that was her reason for not having time to join us.",
        ]
        embeds = model(texts)
        print(embeds)
