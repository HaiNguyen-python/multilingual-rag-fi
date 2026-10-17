"""Optional neural parts ([neural] extra): multilingual dense retrieval and a cross-encoder.

Not run in CI or in the sandbox this repo was built in (no model downloads there). The classes
take the model object as an argument, so the logic around the model is still unit-tested with a
fake encoder. Only the two ``load_*`` functions touch sentence-transformers.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .corpus import Chunk
from .retrieval import Hit

E5_MODEL = "intfloat/multilingual-e5-small"
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Encoder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = ...) -> Any: ...


class DenseIndex:
    """Cosine search over e5 embeddings. e5 expects "query: " and "passage: " prefixes."""

    def __init__(self, encoder: Encoder, chunks: list[Chunk]) -> None:
        self.encoder = encoder
        self.ids = [c.chunk_id for c in chunks]
        passages = [f"passage: {c.search_text}" for c in chunks]
        self.matrix = np.asarray(encoder.encode(passages, normalize_embeddings=True), float)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = np.asarray(self.encoder.encode([f"query: {query}"], normalize_embeddings=True), float)
        sims = self.matrix @ q[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


class CrossEncoderReranker:
    """Scores (question, chunk) pairs jointly. The multilingual model reads the raw question,
    so no translation is needed for reranking."""

    name = "cross-encoder"

    def __init__(self, model: Any) -> None:
        self.model = model

    def rerank(self, question: str, finnish: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(question, h.chunk.search_text) for h in hits])
        rescored = [Hit(h.chunk, float(s)) for h, s in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda h: -h.score)


def load_e5(name: str = E5_MODEL) -> Encoder:  # pragma: no cover - needs a model download
    from sentence_transformers import SentenceTransformer

    model: Encoder = SentenceTransformer(name)
    return model


def load_cross_encoder(name: str = CROSS_ENCODER_MODEL) -> Any:  # pragma: no cover
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)
