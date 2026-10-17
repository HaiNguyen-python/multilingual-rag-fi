"""Query translation into Finnish.

The documents are Finnish only, so an English or Vietnamese question must become Finnish before
Finnish keyword search can work. Two translators share one interface:

- ``GlossaryTranslator`` (offline, CI): longest-phrase lookup in a small hand-written glossary.
  It is a deterministic stand-in for machine translation, not a translator: it knows about 300
  domain phrases per language and nothing else. Words it does not know are kept as they are and
  reported, so the evaluation can show how much of each question it understood.
- ``LLMTranslator``: asks a chat model (Ollama or a hosted API) for a Finnish translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Protocol

import yaml

from .llm import LLMClient
from .text import strip_diacritics, tokenize


@dataclass(frozen=True)
class Translation:
    text: str  # Finnish query text
    source_lang: str
    coverage: float  # share of source content words the translator recognised
    unknown: tuple[str, ...] = ()


class Translator(Protocol):
    name: str

    def to_finnish(self, text: str, lang: str) -> Translation: ...


@dataclass
class Glossary:
    """Phrase tables per source language. An empty value marks a stopword."""

    tables: dict[str, dict[str, str]]
    max_phrase: dict[str, int] = field(default_factory=dict)
    stripped: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for lang, table in self.tables.items():
            self.max_phrase[lang] = max(len(k.split()) for k in table)
            # Vietnamese is often typed without diacritics. Single syllables collide too often
            # once marks are removed (mất / mật / mặt), so only multi-syllable phrases get a
            # diacritic-free alias.
            self.stripped[lang] = {}
            for key, value in table.items():
                if len(key.split()) >= 2:
                    self.stripped[lang].setdefault(strip_diacritics(key), value)

    def lookup(self, phrase: str, lang: str) -> str | None:
        table = self.tables.get(lang, {})
        if phrase in table:
            return table[phrase]
        return self.stripped.get(lang, {}).get(phrase)

    def reverse(self, lang: str) -> dict[str, str]:
        """Finnish word -> first source phrase that maps to it (used to gloss Finnish answers)."""
        out: dict[str, str] = {}
        for src, fi in self.tables.get(lang, {}).items():
            for word in fi.split():
                out.setdefault(word, src)
        return out


def load_glossary(path: Path | None = None) -> Glossary:
    if path is None:
        raw_text = resources.files("mlrag").joinpath("glossary.yaml").read_text(encoding="utf-8")
    else:
        raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    tables: dict[str, dict[str, str]] = {}
    # Keys go through the same tokenizer as questions, so "driver's license" is stored as
    # "driver s license" and matches the tokens of a real question.
    for lang in ("en", "vi"):
        table = {" ".join(tokenize(str(w))): "" for w in raw["stopwords"][lang]}
        table.update({" ".join(tokenize(str(k))): str(v) for k, v in raw[lang].items()})
        tables[lang] = table
    return Glossary(tables)


class GlossaryTranslator:
    name = "glossary"

    def __init__(self, glossary: Glossary | None = None) -> None:
        self.glossary = glossary or load_glossary()

    def to_finnish(self, text: str, lang: str) -> Translation:
        if lang == "fi":
            return Translation(text, lang, 1.0)
        tokens = tokenize(text)
        out: list[str] = []
        unknown: list[str] = []
        known = 0
        i = 0
        while i < len(tokens):
            for size in range(min(self.glossary.max_phrase.get(lang, 1), len(tokens) - i), 0, -1):
                value = self.glossary.lookup(" ".join(tokens[i : i + size]), lang)
                if value is not None:
                    if value:  # empty value = stopword, dropped silently
                        out.append(value)
                        known += 1
                    i += size
                    break
            else:
                token = tokens[i]
                out.append(token)  # numbers and names (YKI, Kela) survive translation as-is
                if token[0].isdigit():
                    known += 1
                else:
                    unknown.append(token)
                i += 1
        total = known + len(unknown)
        return Translation(" ".join(out), lang, known / total if total else 1.0, tuple(unknown))


_LANG_NAMES = {"en": "English", "vi": "Vietnamese", "fi": "Finnish"}


class LLMTranslator:
    """Machine translation with a chat model. Coverage is reported as 1.0 (not measurable)."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = f"llm:{client.name}"

    def to_finnish(self, text: str, lang: str) -> Translation:
        if lang == "fi":
            return Translation(text, lang, 1.0)
        system = (
            f"Translate the user's {_LANG_NAMES.get(lang, lang)} question into natural Finnish "
            "as used on Finnish public-service websites. Reply with the translation only."
        )
        return Translation(self.client.complete(system, text), lang, 1.0)
