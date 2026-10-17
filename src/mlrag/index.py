"""Incremental index: re-chunk only the pages whose content hash changed.

The index is one JSON file. Each page entry stores its content hash and its chunks. On every
run we compare hashes: unchanged pages are kept as they are, changed and new pages are chunked
again, pages that disappeared are removed. Here chunking is cheap; in a system that stores
neural embeddings per chunk, the same check is what saves time and money.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus import Chunk, chunk_page, load_pages

INDEX_FILE = "index.json"


@dataclass
class IndexStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks: int = 0


def _read(index_dir: Path) -> dict[str, dict]:
    path = index_dir / INDEX_FILE
    if not path.exists():
        return {}
    pages: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))["pages"]
    return pages


def build_index(corpus_dir: Path, index_dir: Path, max_words: int = 120) -> IndexStats:
    old = _read(index_dir)
    new: dict[str, dict] = {}
    stats = IndexStats()
    for page in load_pages(corpus_dir):
        previous = old.get(page.page_id)
        if previous and previous["hash"] == page.content_hash:
            new[page.page_id] = previous
            stats.unchanged.append(page.page_id)
            continue
        (stats.updated if previous else stats.added).append(page.page_id)
        new[page.page_id] = {
            "hash": page.content_hash,
            "source": page.source,
            "chunks": [asdict(c) for c in chunk_page(page, max_words)],
        }
    stats.removed = sorted(set(old) - set(new))
    stats.chunks = sum(len(p["chunks"]) for p in new.values())

    index_dir.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so a crash never leaves a half-written index.
    tmp = index_dir / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps({"pages": new}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, index_dir / INDEX_FILE)
    return stats


def load_chunks(index_dir: Path) -> list[Chunk]:
    pages = _read(index_dir)
    if not pages:
        raise FileNotFoundError(f"no index in {index_dir}; run `mlrag index` first")
    return [Chunk(**c) for page in pages.values() for c in page["chunks"]]
