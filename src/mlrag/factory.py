"""Wire the parts together from settings (environment variables with safe offline defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .answer import Assistant, ExtractiveGenerator, Generator, LLMGenerator
from .index import build_index, load_chunks
from .llm import LLMClient, make_client
from .retrieval import DenseSearch, Reranker, Retriever
from .translate import GlossaryTranslator, LLMTranslator, Translator


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path("index")
    provider: str = "offline"  # offline | ollama | openai
    neural: bool = False  # load multilingual-e5 and a cross-encoder ([neural] extra)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            corpus_dir=Path(os.environ.get("MLRAG_CORPUS_DIR", "corpus")),
            index_dir=Path(os.environ.get("MLRAG_INDEX_DIR", "index")),
            provider=os.environ.get("MLRAG_PROVIDER", "offline"),
            neural=os.environ.get("MLRAG_NEURAL", "0") == "1",
        )


def make_retriever(settings: Settings, client: LLMClient | None = None) -> Retriever:
    """Build the index if needed (incremental, so this is cheap when nothing changed)."""
    build_index(settings.corpus_dir, settings.index_dir)
    chunks = load_chunks(settings.index_dir)
    translator: Translator = LLMTranslator(client) if client else GlossaryTranslator()
    dense: DenseSearch | None = None
    reranker: Reranker | None = None
    if settings.neural:  # pragma: no cover - needs model downloads
        from .neural import CrossEncoderReranker, DenseIndex, load_cross_encoder, load_e5

        dense = DenseIndex(load_e5(), chunks)
        reranker = CrossEncoderReranker(load_cross_encoder())
    return Retriever(chunks, translator, dense=dense, reranker=reranker)


def make_generator(retriever: Retriever, client: LLMClient | None = None) -> Generator:
    if client is not None:
        return LLMGenerator(client)
    glossary = GlossaryTranslator().glossary
    return ExtractiveGenerator(retriever.analyzer, glossary)


def make_assistant(settings: Settings, strategy: str = "hybrid") -> Assistant:
    client = make_client(settings.provider)
    retriever = make_retriever(settings, client)
    return Assistant(retriever, make_generator(retriever, client), strategy)
