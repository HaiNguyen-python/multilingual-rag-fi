import numpy as np
import pytest

from mlrag.corpus import Chunk
from mlrag.lexical import BM25, CharNgramIndex
from mlrag.neural import CrossEncoderReranker, DenseIndex
from mlrag.retrieval import Hit, Retriever, rrf, weighted_coverage
from mlrag.translate import GlossaryTranslator


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == fused["b"]  # symmetric ranks, equal score
    assert fused["a"] > fused["c"] > 0 and fused["c"] == fused["d"]


def test_bm25_prefers_rare_terms_and_unknown_terms_score_nothing():
    bm = BM25().fit(["x", "y", "z"], [["kortti", "maksu"], ["maksu"], ["maksu", "uinti"]])
    assert bm.idf("kortti") > bm.idf("maksu")
    assert bm.idf("never-seen") > bm.idf("kortti")  # unseen words weigh most in coverage
    assert bm.search(["kortti", "maksu"], 3)[0][0] == "x"
    assert bm.scores(["never-seen"]) == {}


def test_char_ngrams_match_inflected_forms_without_any_stemmer():
    index = CharNgramIndex().fit(["a", "b"], ["Kirjastokortti on maksuton.", "Uimahalli aukeaa."])
    assert index.search("kirjastokortin", 2)[0][0] == "a"
    assert index.search("zzzz", 2) == []


def test_weighted_coverage_lets_rare_words_decide():
    idf = {"maksaa": 0.1, "kalastuslupa": 3.0}
    assert weighted_coverage(["maksaa", "kalastuslupa"], {"maksaa"}, idf) == pytest.approx(
        0.1 / 3.1
    )
    assert weighted_coverage([], {"maksaa"}, idf) == 0.0


def test_english_question_finds_the_finnish_page_after_translation(retriever: Retriever):
    q = "How much is a 30-day season ticket?"
    translated = retriever.retrieve(q, "en", "translate_bm25")
    assert translated.hits[0].chunk.page_id == "matkakortti"
    assert "kausilippu" in translated.translation.text
    # The same question without translation: character n-grams have nothing to match.
    raw = retriever.retrieve(q, "en", "ngram")
    assert all(h.chunk.page_id != "matkakortti" for h in raw.hits[:3])


def test_hybrid_and_rerank_return_the_answer_page_for_vietnamese(retriever: Retriever):
    for strategy in ("hybrid", "hybrid_rerank"):
        result = retriever.retrieve("Thuốc cũ hết hạn thì mang đi đâu?", "vi", strategy)
        assert result.hits[0].chunk.page_id == "vaarallinen-jate"
        assert 0.0 <= result.relevance <= 1.0


def test_off_topic_question_gets_low_relevance(retriever: Retriever):
    on_topic = retriever.retrieve("Paljonko vuorokausilippu maksaa?", "fi").relevance
    off_topic = retriever.retrieve("Paljonko kalastuslupa maksaa?", "fi").relevance
    assert off_topic < 0.41 <= on_topic


def test_unknown_strategy_and_missing_dense_model_fail_clearly(retriever: Retriever):
    with pytest.raises(ValueError, match="strategy"):
        retriever.retrieve("x", "fi", "magic")
    with pytest.raises(ValueError, match="neural"):
        retriever.retrieve("x", "fi", "dense")


class FakeEncoder:
    """Bag-of-letters 'embedding': enough to test the plumbing without a model download."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 26))
        for i, text in enumerate(texts):
            for ch in text.lower().split(":", 1)[-1]:
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(cid, cid.split("#")[0], "T", "", "", "S", text)


def test_dense_strategy_uses_the_injected_encoder_on_the_raw_question():
    chunks = [_chunk("a#0", "zzz zzz"), _chunk("b#0", "abc abc")]
    dense = DenseIndex(FakeEncoder(), chunks)
    retriever = Retriever(chunks, GlossaryTranslator(), dense=dense)
    result = retriever.retrieve("abc", "en", "dense", k=2)
    assert [h.chunk.chunk_id for h in result.hits] == ["b#0", "a#0"]
    assert result.relevance == pytest.approx(result.hits[0].score, abs=1e-4)


def test_cross_encoder_reranker_reorders_by_model_score():
    class FakeModel:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    hits = [Hit(_chunk("a#0", "short"), 1.0), Hit(_chunk("b#0", "a much longer text"), 0.5)]
    reranked = CrossEncoderReranker(FakeModel()).rerank("q", "q", hits)
    assert [h.chunk.chunk_id for h in reranked] == ["b#0", "a#0"]
    assert CrossEncoderReranker(FakeModel()).rerank("q", "q", []) == []
