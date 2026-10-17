import pytest
import yaml
from conftest import CORPUS, QUESTIONS, ROOT

from mlrag.answer import DEFAULT_MIN_RELEVANCE, Answer, Assistant, CitationCheck
from mlrag.corpus import load_pages
from mlrag.evals import (
    EvalItem,
    check_gate,
    cross_lingual_gap,
    evaluate,
    first_relevant_rank,
    load_eval_set,
    score,
    sweep_threshold,
)
from mlrag.retrieval import LEXICAL_STRATEGIES


def _answer(retrieved: list[str], text: str = "x [a#0]", refused: bool = False) -> Answer:
    return Answer(
        question="q",
        lang="en",
        text=text,
        refused=refused,
        citations=CitationCheck(not refused, ("a#0",), ()),
        retrieved=[(c, 1.0) for c in retrieved],
        relevance=0.5,
        query_fi="",
        coverage=0.5,
        generator="test",
    )


def test_metrics_on_a_hand_computed_example():
    items = [
        EvalItem("q1", "en", "?", ("a",), True, key="3 euroa"),
        EvalItem("q2", "en", "?", ("b",), True, key="9"),
        EvalItem("q3", "en", "?", ("c",), True, key="z"),
        EvalItem("u1", "en", "?", (), False),
        EvalItem("u2", "en", "?", (), False),
    ]
    answers = [
        _answer(["a#0", "x#0"], text="Uusi kortti maksaa 3 euroa. [a#0]"),  # rank 1, key found
        _answer(["x#0", "y#0", "z#0", "b#1"]),  # rank 4: hit@5 but not hit@3, key missing
        _answer(["x#0"], refused=True),  # missed and refused
        _answer([], refused=True),  # correct refusal
        _answer(["x#0"]),  # answered an unanswerable question
    ]
    m = score(items, answers)
    assert m["hit_at_3"] == pytest.approx(1 / 3)
    assert m["hit_at_5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 4 + 0) / 3)
    assert m["answer_rate"] == pytest.approx(2 / 3)
    assert m["key_fact"] == pytest.approx(1 / 3)
    assert m["refusal_acc"] == pytest.approx(1 / 2)
    assert m["citation_valid"] == 1.0
    assert first_relevant_rank(["x#0", "b#3"], ("b",)) == 2


def test_sweep_picks_the_middle_of_the_widest_best_range():
    # Answerable relevances 0.6 and 0.7, unanswerable 0.2: every t in (0.2, 0.6] is perfect.
    sweep = sweep_threshold([0.6, 0.7, 0.2], [True, True, False], step=0.1)
    assert sweep.best_range == (0.3, 0.6)
    assert sweep.best == pytest.approx(0.45)
    assert max(sweep.balanced) == 1.0


def test_sweep_prefers_the_widest_of_several_tied_ranges():
    # Answerable 0.15 and 0.85, unanswerable 0.5 and 0.95. Balanced accuracy is 0.5 on three
    # separate ranges: t <= 0.15, 0.55 <= t <= 0.85 and t = 1.0. The widest one wins.
    sweep = sweep_threshold([0.15, 0.85, 0.5, 0.95], [True, True, False, False], step=0.05)
    assert sweep.best_range == pytest.approx((0.55, 0.85))
    assert sweep.best == pytest.approx(0.7)


def test_gap_and_share_of_baseline_gap_closed():
    metrics = {
        "ngram": {"fi": {"mrr": 1.0}, "en": {"mrr": 0.2}, "vi": {"mrr": 0.0}},
        "hybrid": {"fi": {"mrr": 1.0}, "en": {"mrr": 0.9}, "vi": {"mrr": 0.75}},
    }
    gap = cross_lingual_gap(metrics)
    assert gap["hybrid"]["gap_en"] == pytest.approx(0.1)
    assert gap["hybrid"]["closed_en"] == pytest.approx(1 - 0.1 / 0.8)
    assert gap["hybrid"]["closed_vi"] == pytest.approx(0.75)
    assert gap["ngram"]["closed_en"] == 0.0


def test_eval_set_is_consistent_with_the_corpus():
    items = load_eval_set(QUESTIONS)
    assert len(items) == (40 + 8) * 3
    pages = {p.page_id: p for p in load_pages(CORPUS)}
    for item in items:
        assert item.question.strip()
        if item.answerable:
            # every expected page exists and really contains the key fact
            assert all(p in pages for p in item.pages)
            assert any(item.key in pages[p].body for p in item.pages), item.qid


def test_shipped_threshold_matches_the_sweep_and_the_gate_passes(assistant: Assistant):
    items = load_eval_set(QUESTIONS)
    report = evaluate(assistant.retriever, assistant.generator, items, list(LEXICAL_STRATEGIES))
    # If the corpus, glossary or relevance score changes, re-run `mlrag eval` and update
    # DEFAULT_MIN_RELEVANCE; this test keeps the code and the docs honest.
    assert report.sweeps["hybrid"].best == DEFAULT_MIN_RELEVANCE
    config = yaml.safe_load((ROOT / "evals" / "thresholds.yaml").read_text(encoding="utf-8"))
    assert check_gate(report, config) == []
    strict = {**config, "max_mrr_gap": 0.0}
    assert any("MRR gap" in f for f in check_gate(report, strict))
