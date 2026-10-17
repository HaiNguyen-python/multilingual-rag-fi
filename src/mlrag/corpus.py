"""Load the Finnish pages and split them into chunks by heading.

A corpus file holds one or more pages. Each page starts with a marker comment:

    <!-- page: kirjastokortti | en: Library card | vi: Thẻ thư viện -->
    # Kirjastokortti
    ...
    ## Kortin hankkiminen

The English and Vietnamese titles let an answer name its source in the question's language.
Other HTML comments (such as the "illustrative, not official" notice) are ignored.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PAGE_RE = re.compile(r"<!--\s*page:\s*(?P<meta>.*?)-->", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ0-9])")


@dataclass(frozen=True)
class Page:
    page_id: str
    title: str
    title_en: str
    title_vi: str
    body: str
    source: str

    @property
    def content_hash(self) -> str:
        payload = "\n".join([self.page_id, self.title, self.title_en, self.title_vi, self.body])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<page_id>#<n>", the id used in citations
    page_id: str
    title: str
    title_en: str
    title_vi: str
    section: str
    text: str

    @property
    def search_text(self) -> str:
        """Title and heading carry the topic ("Kadonnut kortti"), so they are indexed too."""
        return f"{self.title}. {self.section}. {self.text}"

    def sentences(self) -> list[str]:
        return [s.strip() for s in SENTENCE_RE.split(self.text) if s.strip()]


def _parse_meta(meta: str) -> dict[str, str]:
    parts = [p.strip() for p in meta.split("|")]
    out = {"id": parts[0]}
    for part in parts[1:]:
        key, _, value = part.partition(":")
        out[key.strip()] = value.strip()
    return out


def parse_pages(markdown: str, source: str) -> list[Page]:
    markers = list(PAGE_RE.finditer(markdown))
    pages = []
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
        body = COMMENT_RE.sub("", markdown[m.end() : end]).strip()
        title_match = re.match(r"#\s+(.+)", body)
        if not title_match:
            raise ValueError(f"{source}: page {m.group('meta')!r} has no '# title' line")
        meta = _parse_meta(m.group("meta"))
        pages.append(
            Page(
                page_id=meta["id"],
                title=title_match.group(1).strip(),
                title_en=meta.get("en", ""),
                title_vi=meta.get("vi", ""),
                body=body[title_match.end() :].strip(),
                source=source,
            )
        )
    return pages


def load_pages(corpus_dir: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(corpus_dir.glob("*.md")):
        pages.extend(parse_pages(path.read_text(encoding="utf-8"), path.name))
    ids = [p.page_id for p in pages]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate page ids: {duplicates}")
    return pages


def _pack_sentences(text: str, max_words: int) -> list[str]:
    """Split an over-long section at sentence borders so no chunk exceeds max_words."""
    sentences = [s for s in SENTENCE_RE.split(" ".join(text.split())) if s]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_page(page: Page, max_words: int = 120) -> list[Chunk]:
    """One chunk per '##' section (the intro before the first heading counts as a section).

    A citation then points to one topic. Sections longer than max_words are split at sentence
    borders, but a chunk never crosses a heading.
    """
    sections: list[tuple[str, str]] = []
    heading = page.title
    lines: list[str] = []
    for line in page.body.splitlines():
        if line.startswith("## "):
            sections.append((heading, " ".join(lines)))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line.strip())
    sections.append((heading, " ".join(lines)))

    chunks: list[Chunk] = []
    for section, text in sections:
        for piece in _pack_sentences(text, max_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.page_id}#{len(chunks)}",
                    page_id=page.page_id,
                    title=page.title,
                    title_en=page.title_en,
                    title_vi=page.title_vi,
                    section=section,
                    text=piece,
                )
            )
    return chunks
