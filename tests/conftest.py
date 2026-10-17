from __future__ import annotations

from pathlib import Path

import pytest

from mlrag.answer import Assistant, ExtractiveGenerator
from mlrag.corpus import Chunk, chunk_page, load_pages
from mlrag.retrieval import Retriever
from mlrag.translate import GlossaryTranslator, load_glossary

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "evals" / "questions.yaml"

TINY_CORPUS = """\
<!-- illustrative test page, not official guidance -->
<!-- page: kirjastokortti | en: Library card | vi: Thẻ thư viện -->
# Kirjastokortti

Kirjastokortti on maksuton.

## Kadonnut kortti

Ilmoita kadonneesta kortista heti. Uusi kortti maksaa 3 euroa.

<!-- page: uimahalli | en: Swimming hall | vi: Hồ bơi trong nhà -->
# Uimahalli

## Liput

Aikuisen kertalippu maksaa 6,50 euroa.
"""


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "01-test.md").write_text(TINY_CORPUS, encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    return [c for page in load_pages(CORPUS) for c in chunk_page(page)]


@pytest.fixture(scope="session")
def retriever(chunks: list[Chunk]) -> Retriever:
    return Retriever(chunks, GlossaryTranslator())


@pytest.fixture(scope="session")
def assistant(retriever: Retriever) -> Assistant:
    generator = ExtractiveGenerator(retriever.analyzer, load_glossary())
    return Assistant(retriever, generator, strategy="hybrid")
