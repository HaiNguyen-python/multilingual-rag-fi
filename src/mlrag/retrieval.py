"""Retrieval strategies compared per question language.

- ``translate_bm25``: translate the question to Finnish, then BM25 over Finnish stems.
- ``ngram``: character n-gram TF-IDF on the question as typed. Needs no translation, so it is
  the language-agnostic baseline. It works for Finnish and fails across languages, which is
  exactly the gap this project measures.
- ``hybrid``: Reciprocal Rank Fusion of translated BM25 and n-grams over the translated query.
  The n-gram side catches inflected and compound forms that the stemmer misses.
- ``hybrid_rerank``: hybrid, then a reranker over the top candidates (lexical in CI, a
  multilingual cross-encoder with the [neural] extra).
- ``dense``: multilingual embeddings on the raw question ([neural] extra, not run in CI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk
from .lexical import BM25, CharNgramIndex
from .text import FinnishAnalyzer
from .translate import Translation, Translator

RRF_K = 60
LEXICAL_STRATEGIES = ("translate_bm25", "ngram", "hybrid", "hybrid_rerank")
STRATEGIES = (*LEXICAL_STRATEGIES, "dense")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[Hit]
    relevance: float  # 0..1; the assistant refuses below a threshold
    translation: Translation


class DenseSearch(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, finnish: str, hits: list[Hit]) -> list[Hit]: ...


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """score(doc) = sum over rankings of 1 / (k + rank). Uses ranks only, so BM25 scores and
    cosine similarities never need to be put on one scale."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def weighted_coverage(query_terms: list[str], doc_terms: set[str], idf: dict[str, float]) -> float:
    """IDF-weighted share of the query's terms found in a chunk.

    A plain share would let a common word ("maksaa", costs) carry an off-topic question past the
    threshold. Weighting by IDF makes the rare, topical words decide; a word that is not in the
    corpus at all (e.g. "kalastuslupa") gets the highest weight and pulls relevance down.
    """
    unique = set(query_terms)
    total = sum(idf[t] for t in unique)
    if total == 0:
        return 0.0
    return sum(idf[t] for t in unique if t in doc_terms) / total


class LexicalReranker:
    """Reorders candidates by weighted term coverage, then by the fused score."""

    name = "lexical"

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def rerank(self, question: str, finnish: str, hits: list[Hit]) -> list[Hit]:
        terms = self.retriever.analyzer.terms(finnish)
        scored = [
            Hit(h.chunk, self.retriever.coverage(terms, h.chunk.chunk_id) + h.score) for h in hits
        ]
        return sorted(scored, key=lambda h: (-h.score, h.chunk.chunk_id))


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        translator: Translator,
        dense: DenseSearch | None = None,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.translator = translator
        self.dense = dense
        self.candidates = candidates
        ids = list(self.chunks)
        texts = [self.chunks[i].search_text for i in ids]
        self.analyzer = FinnishAnalyzer.from_texts(texts)
        self.doc_terms = {i: self.analyzer.terms(t) for i, t in zip(ids, texts, strict=True)}
        self._term_sets = {i: set(t) for i, t in self.doc_terms.items()}
        self.bm25 = BM25().fit(ids, [self.doc_terms[i] for i in ids])
        self.ngrams = CharNgramIndex().fit(ids, texts)
        self.reranker: Reranker = reranker or LexicalReranker(self)

    def coverage(self, query_terms: list[str], chunk_id: str) -> float:
        idf = {t: self.bm25.idf(t) for t in set(query_terms)}
        return weighted_coverage(query_terms, self._term_sets[chunk_id], idf)

    def _hybrid(self, finnish: str) -> list[tuple[str, float]]:
        bm = self.bm25.search(self.analyzer.terms(finnish), self.candidates)
        ng = self.ngrams.search(finnish, self.candidates)
        fused = rrf([[i for i, _ in bm], [i for i, _ in ng]])
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

    def retrieve(
        self, question: str, lang: str, strategy: str = "hybrid", k: int = 10
    ) -> RetrievalResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if strategy == "ngram" or strategy == "dense":
            # These strategies read the question as typed; no translation is used.
            translation = Translation(question, lang, 1.0)
        else:
            translation = self.translator.to_finnish(question, lang)
        finnish = translation.text

        if strategy == "translate_bm25":
            ranked = self.bm25.search(self.analyzer.terms(finnish), k)
        elif strategy == "ngram":
            ranked = self.ngrams.search(question, k)
        elif strategy == "dense":
            if self.dense is None:
                raise ValueError("dense retrieval needs the [neural] extra (pip install .[neural])")
            ranked = self.dense.search(question, k)
        else:
            ranked = self._hybrid(finnish)

        hits = [Hit(self.chunks[i], s) for i, s in ranked]
        if strategy == "hybrid_rerank":
            hits = self.reranker.rerank(question, finnish, hits[: self.candidates])
        hits = hits[:k]
        if not hits:
            return RetrievalResult([], 0.0, translation)
        if strategy in ("ngram", "dense"):
            relevance = max(0.0, min(1.0, hits[0].score))  # cosine similarity of the best hit
        else:
            relevance = self.coverage(self.analyzer.terms(finnish), hits[0].chunk.chunk_id)
        return RetrievalResult(hits, round(relevance, 4), translation)
