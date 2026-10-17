from pathlib import Path

import pytest
from conftest import CORPUS, TINY_CORPUS

from mlrag.corpus import Page, chunk_page, load_pages, parse_pages
from mlrag.index import build_index, load_chunks


def test_parse_pages_reads_markers_titles_and_ignores_notices():
    pages = parse_pages(TINY_CORPUS, "01-test.md")
    assert [p.page_id for p in pages] == ["kirjastokortti", "uimahalli"]
    assert pages[0].title == "Kirjastokortti"
    assert pages[0].title_en == "Library card"
    assert pages[0].title_vi == "Thẻ thư viện"
    assert "illustrative" not in pages[0].body


def test_chunking_follows_headings_and_ids_are_stable():
    page = parse_pages(TINY_CORPUS, "x.md")[0]
    chunks = chunk_page(page)
    assert [(c.chunk_id, c.section) for c in chunks] == [
        ("kirjastokortti#0", "Kirjastokortti"),
        ("kirjastokortti#1", "Kadonnut kortti"),
    ]
    assert chunks[1].sentences() == [
        "Ilmoita kadonneesta kortista heti.",
        "Uusi kortti maksaa 3 euroa.",
    ]
    assert "Kadonnut kortti" in chunks[1].search_text  # the heading is searchable


def test_long_section_is_split_at_sentences_never_mid_sentence():
    body = "## Pitkä\n\n" + " ".join(f"Lause numero {i} on tässä." for i in range(40))
    page = Page("p", "Otsikko", "", "", body, "x.md")
    chunks = chunk_page(page, max_words=30)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 30 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert {c.section for c in chunks} == {"Pitkä"}


def test_duplicate_page_ids_are_rejected(tmp_path: Path):
    (tmp_path / "a.md").write_text(TINY_CORPUS, encoding="utf-8")
    (tmp_path / "b.md").write_text(TINY_CORPUS, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_pages(tmp_path)


def test_shipped_corpus_is_labelled_illustrative_and_has_enough_pages():
    pages = load_pages(CORPUS)
    assert 15 <= len(pages) <= 20
    for path in CORPUS.glob("*.md"):
        assert "not official guidance" in path.read_text(encoding="utf-8")


def test_incremental_index_only_rechunks_changed_pages(tiny_corpus: Path, tmp_path: Path):
    index_dir = tmp_path / "index"
    first = build_index(tiny_corpus, index_dir)
    assert first.added == ["kirjastokortti", "uimahalli"] and first.chunks == 3

    second = build_index(tiny_corpus, index_dir)  # idempotent: nothing to do
    assert second.unchanged == ["kirjastokortti", "uimahalli"]
    assert not (second.added or second.updated or second.removed)

    path = tiny_corpus / "01-test.md"
    text = path.read_text(encoding="utf-8").replace("6,50", "7,00")
    path.write_text(text, encoding="utf-8")
    third = build_index(tiny_corpus, index_dir)
    assert third.updated == ["uimahalli"] and third.unchanged == ["kirjastokortti"]
    assert any("7,00" in c.text for c in load_chunks(index_dir))

    path.write_text(text.split("<!-- page: uimahalli")[0], encoding="utf-8")
    fourth = build_index(tiny_corpus, index_dir)
    assert fourth.removed == ["uimahalli"]
    assert {c.page_id for c in load_chunks(index_dir)} == {"kirjastokortti"}


def test_load_chunks_without_index_explains_what_to_do(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="mlrag index"):
        load_chunks(tmp_path)
