"""Two lexical retrievers: BM25 over Finnish stems, and character n-gram TF-IDF.

BM25 needs the query in Finnish (after translation) because it matches stems.
Character n-grams need no language knowledge at all: "kirjastokortin" and "kirjastokortti"
share most of their 3-5 letter pieces. That makes n-grams a fair language-agnostic baseline,
and a useful partner for BM25 when the stemmer misses a form (lippu / lipun).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalise


class BM25:
    """Okapi BM25 over pre-analysed term lists (written out so each step is visible)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()

    def fit(self, ids: list[str], docs: list[list[str]]) -> BM25:
        self.ids = list(ids)
        self.tfs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.df = Counter(term for tf in self.tfs for term in tf)
        return self

    def idf(self, term: str) -> float:
        """Rare terms weigh more. A term the corpus has never seen gets the highest weight."""
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, terms: list[str]) -> dict[str, float]:
        if not self.ids:
            return {}
        avg_len = sum(self.lengths) / len(self.ids)
        out: dict[str, float] = {}
        for term in set(terms):
            if term not in self.df:
                continue
            idf = self.idf(term)
            for doc_id, tf, length in zip(self.ids, self.tfs, self.lengths, strict=True):
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / avg_len)
                    out[doc_id] = out.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / norm
        return out

    def search(self, terms: list[str], k: int) -> list[tuple[str, float]]:
        scores = self.scores(terms)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class CharNgramIndex:
    """TF-IDF over character 3-5-grams inside word boundaries, cosine similarity."""

    def __init__(self, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True, preprocessor=normalise
        )
        self.ids: list[str] = []

    def fit(self, ids: list[str], texts: list[str]) -> CharNgramIndex:
        self.ids = list(ids)
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Rows are L2-normalised by TfidfVectorizer, so the dot product is the cosine.
        sims = np.asarray((self.matrix @ self.vectorizer.transform([query]).T).todense()).ravel()
        order = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return [(self.ids[i], float(sims[i])) for i in order if sims[i] > 0]
