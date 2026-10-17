"""Answering with citations, a citation validator and refusal below a relevance threshold.

Two generators share one interface:
- ``ExtractiveGenerator`` (offline, CI): quotes the best Finnish sentences and cites their
  chunks. For English and Vietnamese it wraps the quote in a template in the question's language
  and adds a glossary gloss of the key terms. It does not translate the sentence: the reader gets
  the exact Finnish source plus help to read it. That is the honest limit of an offline system.
- ``LLMGenerator``: a chat model writes the answer in the question's language from the Finnish
  context, and must cite chunk ids. Its citations are checked, because models invent them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLMClient
from .retrieval import Hit, RetrievalResult, Retriever
from .text import FinnishAnalyzer, detect_language, stem, tokenize
from .translate import Glossary

# Chosen by `mlrag eval` with a threshold sweep on the eval set (balanced accuracy, middle of
# the best range) for the hybrid strategy. tests/test_evals.py fails if it goes stale.
DEFAULT_MIN_RELEVANCE = 0.41
DISCLAIMER = (
    "Illustrative answer from illustrative Finnish example pages, not official guidance. "
    "Check the official site."
)
REFUSALS = {
    "fi": "En löydä tähän vastausta aineistosta.",
    "en": "I can't find this in the documents.",
    "vi": "Tôi không tìm thấy thông tin này trong tài liệu.",
}
CITATION_RE = re.compile(r"\[([a-z0-9-]+#\d+)\]")
_LANG_NAMES = {"fi": "Finnish", "en": "English", "vi": "Vietnamese"}


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    cited: tuple[str, ...]
    unknown: tuple[str, ...]  # cited ids that were not in the context: invented or stale


def validate_citations(text: str, context_ids: set[str]) -> CitationCheck:
    """An answer is valid only if it cites at least one chunk and every cited chunk was in the
    context the generator saw."""
    cited = tuple(dict.fromkeys(CITATION_RE.findall(text)))
    unknown = tuple(c for c in cited if c not in context_ids)
    return CitationCheck(bool(cited) and not unknown, cited, unknown)


def is_refusal(text: str) -> bool:
    return any(r.lower().rstrip(".") in text.lower() for r in REFUSALS.values())


@dataclass
class Answer:
    question: str
    lang: str
    text: str
    refused: bool
    citations: CitationCheck
    retrieved: list[tuple[str, float]]
    relevance: float
    query_fi: str
    coverage: float
    generator: str
    extra: dict[str, object] = field(default_factory=dict)


class Generator(Protocol):
    name: str

    def generate(self, question: str, lang: str, query_fi: str, hits: list[Hit]) -> str: ...


class ExtractiveGenerator:
    name = "extractive"

    def __init__(
        self, analyzer: FinnishAnalyzer, glossary: Glossary, max_sentences: int = 2
    ) -> None:
        self.analyzer = analyzer
        self.max_sentences = max_sentences
        self._reverse = {lang: glossary.reverse(lang) for lang in ("en", "vi")}
        self._reverse_stems = {
            lang: {stem(fi): src for fi, src in table.items()}
            for lang, table in self._reverse.items()
        }

    def best_sentences(self, query_fi: str, hits: list[Hit]) -> list[tuple[str, Hit]]:
        """Score each sentence by how many query *words* it matches.

        Counting words, not terms, matters: "kirjastokortti" expands to three terms (the whole
        stem and both compound parts), and counting terms would let any sentence that repeats
        the page topic beat the sentence that actually answers the question.
        """
        words = [set(self.analyzer.terms(w)) for w in query_fi.split()]
        words = [w for w in words if w]
        scored = []
        for rank, hit in enumerate(hits):
            for pos, sentence in enumerate(hit.chunk.sentences()):
                terms = set(self.analyzer.terms(sentence))
                overlap = sum(bool(w & terms) for w in words)
                if overlap:
                    # More shared terms first, then higher-ranked chunks, then earlier sentences.
                    scored.append((-overlap, rank, pos, sentence, hit))
        scored.sort(key=lambda s: s[:3])
        return [(s[3], s[4]) for s in scored[: self.max_sentences]]

    def gloss(self, sentence: str, lang: str, limit: int = 5) -> list[tuple[str, str]]:
        """Finnish word -> English/Vietnamese glossary term, for the words the glossary knows."""
        stems = self._reverse_stems[lang]
        out: dict[str, str] = {}
        for token in tokenize(sentence):
            if token[0].isdigit() or token in out:
                continue
            match = stems.get(stem(token))
            if match is None:
                parts = self.analyzer.split_compound(token)
                found = [stems[p] for p in parts if p in stems]
                match = " + ".join(found) if len(found) == len(parts) and found else None
            if match:
                out[token] = match
        return list(out.items())[:limit]

    def generate(self, question: str, lang: str, query_fi: str, hits: list[Hit]) -> str:
        picked = self.best_sentences(query_fi, hits)
        if not picked:
            return REFUSALS.get(lang, REFUSALS["en"])
        if lang == "fi":
            return " ".join(f"{s} [{h.chunk.chunk_id}]" for s, h in picked)
        quotes = " ".join(f'"{s}" [{h.chunk.chunk_id}]' for s, h in picked)
        chunk = picked[0][1].chunk
        terms = self.gloss(" ".join(s for s, _ in picked), lang)
        glossed = "; ".join(f"{fi} = {src}" for fi, src in terms)
        if lang == "vi":
            head = f'Trang tiếng Phần Lan "{chunk.title}" ({chunk.title_vi}) viết: {quotes}'
            tail = "Đây là trích dẫn tiếng Phần Lan, không phải bản dịch."
            return f"{head}\nThuật ngữ chính: {glossed}\n{tail}" if glossed else f"{head}\n{tail}"
        head = f'The Finnish page "{chunk.title}" ({chunk.title_en}) says: {quotes}'
        tail = "This is a quote in Finnish, not a translation."
        return f"{head}\nKey terms: {glossed}\n{tail}" if glossed else f"{head}\n{tail}"


class LLMGenerator:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = f"llm:{client.name}"

    def generate(self, question: str, lang: str, query_fi: str, hits: list[Hit]) -> str:
        refusal = REFUSALS.get(lang, REFUSALS["en"])
        system = (
            f"You answer questions about Finnish public services in {_LANG_NAMES[lang]}. "
            "Use ONLY the Finnish context below. After each sentence cite the chunk id in "
            "square brackets, e.g. [kirjastokortti#1]. If the context does not contain the "
            f"answer, reply exactly: {refusal}"
        )
        context = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.title} > {h.chunk.section}\n{h.chunk.text}"
            for h in hits
        )
        return self.client.complete(system, f"Context:\n{context}\n\nQuestion: {question}")


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        strategy: str = "hybrid",
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        top_k: int = 4,
    ) -> None:
        self.retriever, self.generator = retriever, generator
        self.strategy, self.min_relevance, self.top_k = strategy, min_relevance, top_k

    def ask(self, question: str, lang: str | None = None) -> Answer:
        lang = lang or detect_language(question)
        result = self.retriever.retrieve(question, lang, self.strategy)
        return self.answer_from(question, lang, result)

    def answer_from(self, question: str, lang: str, result: RetrievalResult) -> Answer:
        """Answer from an existing retrieval result (the eval reuses one result per question)."""
        context = result.hits[: self.top_k]
        if not context or result.relevance < self.min_relevance:
            text = REFUSALS[lang]
        else:
            text = self.generator.generate(question, lang, result.translation.text, context)
        return Answer(
            question=question,
            lang=lang,
            text=text,
            refused=is_refusal(text),
            citations=validate_citations(text, {h.chunk.chunk_id for h in context}),
            retrieved=[(h.chunk.chunk_id, round(h.score, 5)) for h in result.hits],
            relevance=result.relevance,
            query_fi=result.translation.text,
            coverage=round(result.translation.coverage, 4),
            generator=self.generator.name,
        )
