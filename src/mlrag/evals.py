"""Evaluation per question language and retrieval strategy.

Metrics for each (strategy, language):
- hit@3, hit@5   share of answerable questions with a chunk from an expected page in the top k
- mrr            mean reciprocal rank of the first relevant chunk (top 10, 0 if none)
- answer_rate    share of answerable questions that were answered (not refused)
- key_fact       share of answerable questions whose answer contains the expected key string
- citation_valid share of given answers whose citations pass the validator
- refusal_acc    share of unanswerable questions that were refused

The headline is the cross-lingual gap: MRR on Finnish questions minus MRR on the same questions
asked in English or Vietnamese, and how much of the baseline's gap each strategy closes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .answer import Answer, Assistant, Generator
from .retrieval import RetrievalResult, Retriever
from .text import LANGUAGES

BASELINE = "ngram"


@dataclass(frozen=True)
class EvalItem:
    qid: str
    lang: str
    question: str
    pages: tuple[str, ...]
    answerable: bool
    expected: str = ""
    key: str = ""


def load_eval_set(path: Path) -> list[EvalItem]:
    """Expand each YAML entry into one item per language (same meaning, same expected pages)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items: list[EvalItem] = []
    for entry in raw["answerable"]:
        for lang in LANGUAGES:
            items.append(
                EvalItem(
                    entry["id"],
                    lang,
                    entry[lang],
                    tuple(entry["pages"]),
                    True,
                    entry["expected"],
                    str(entry["key"]),
                )
            )
    for entry in raw["unanswerable"]:
        items.extend(EvalItem(entry["id"], lang, entry[lang], (), False) for lang in LANGUAGES)
    return items


def page_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def first_relevant_rank(ranking: list[str], pages: tuple[str, ...]) -> int | None:
    for rank, chunk_id in enumerate(ranking, start=1):
        if page_of(chunk_id) in pages:
            return rank
    return None


def _mean(values: list[float] | list[bool]) -> float:
    """Mean, or NaN when there is nothing to average (e.g. citation validity with no answers):
    0.0 would wrongly read as "every citation was invalid"."""
    return float(np.mean(values)) if values else float("nan")


def score(items: list[EvalItem], answers: list[Answer]) -> dict[str, float]:
    ranks, answered, key_found, cite_ok, refused_ok, coverage = [], [], [], [], [], []
    for item, ans in zip(items, answers, strict=True):
        if not ans.refused:
            cite_ok.append(ans.citations.valid)
        if not item.answerable:
            refused_ok.append(ans.refused)
            continue
        ranks.append(first_relevant_rank([c for c, _ in ans.retrieved], item.pages))
        answered.append(not ans.refused)
        key_found.append(not ans.refused and item.key.lower() in ans.text.lower())
        coverage.append(ans.coverage)
    return {
        "hit_at_3": _mean([r is not None and r <= 3 for r in ranks]),
        "hit_at_5": _mean([r is not None and r <= 5 for r in ranks]),
        "mrr": _mean([1 / r if r else 0.0 for r in ranks]),
        "answer_rate": _mean(answered),
        "key_fact": _mean(key_found),
        "citation_valid": _mean(cite_ok),
        "refusal_acc": _mean(refused_ok),
        "translation_coverage": _mean(coverage),
    }


@dataclass
class SweepResult:
    thresholds: list[float]
    answer_rate: list[float]  # answerable questions at or above the threshold
    refusal_rate: list[float]  # unanswerable questions below the threshold
    balanced: list[float]
    best: float
    best_range: tuple[float, float]


def sweep_threshold(
    relevances: list[float], answerable: list[bool], step: float = 0.01
) -> SweepResult:
    """Choose the refusal threshold from data instead of guessing.

    At threshold t the assistant answers when relevance >= t. The score is balanced accuracy,
    the mean of the answer rate (answerable) and refusal rate (unanswerable), so the 8
    unanswerable questions per language weigh as much as the 40 answerable ones. Many
    thresholds tie; we take the middle of the widest run of best thresholds, which keeps the
    largest margin on both sides.
    """
    rel = np.asarray(relevances, float)
    ok = np.asarray(answerable, bool)
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 4)
    ans = np.array([(rel[ok] >= t).mean() if ok.any() else 0.0 for t in grid])
    ref = np.array([(rel[~ok] < t).mean() if (~ok).any() else 0.0 for t in grid])
    balanced = np.round((ans + ref) / 2, 10)
    is_best = balanced == balanced.max()

    runs: list[tuple[int, int]] = []  # (start, end) index pairs of consecutive best thresholds
    start = None
    for i, flag in enumerate(is_best):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(grid) - 1):
            runs.append((start, i if flag else i - 1))
            start = None
    lo, hi = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    best = float(np.round((grid[lo] + grid[hi]) / 2, 3))
    return SweepResult(
        grid.tolist(),
        ans.tolist(),
        ref.tolist(),
        balanced.tolist(),
        best,
        (float(grid[lo]), float(grid[hi])),
    )


@dataclass
class EvalReport:
    strategies: list[str]
    thresholds: dict[str, float]
    metrics: dict[str, dict[str, dict[str, float]]]  # strategy -> lang -> metric -> value
    sweeps: dict[str, SweepResult]
    sweep_by_lang: dict[str, SweepResult]  # hybrid strategy, one sweep per language (for charts)
    answers: dict[str, list[Answer]] = field(default_factory=dict)


def evaluate(
    retriever: Retriever,
    generator: Generator,
    items: list[EvalItem],
    strategies: list[str],
    fixed_thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """Retrieve once per (strategy, question), sweep a threshold per strategy, then answer.

    With ``fixed_thresholds`` the sweep is still computed and reported, but the given threshold
    is used for answering (this is how the shipped default is checked in the CI gate).
    """
    report = EvalReport(strategies, {}, {}, {}, {})
    for strategy in strategies:
        results: list[RetrievalResult] = [
            retriever.retrieve(it.question, it.lang, strategy) for it in items
        ]
        relevances = [r.relevance for r in results]
        flags = [it.answerable for it in items]
        sweep = sweep_threshold(relevances, flags)
        threshold = (fixed_thresholds or {}).get(strategy, sweep.best)
        assistant = Assistant(retriever, generator, strategy, threshold)
        answers = [
            assistant.answer_from(it.question, it.lang, r)
            for it, r in zip(items, results, strict=True)
        ]
        report.sweeps[strategy] = sweep
        report.thresholds[strategy] = threshold
        report.answers[strategy] = answers
        report.metrics[strategy] = {}
        for lang in LANGUAGES:
            idx = [i for i, it in enumerate(items) if it.lang == lang]
            report.metrics[strategy][lang] = score(
                [items[i] for i in idx], [answers[i] for i in idx]
            )
            if strategy == "hybrid":
                report.sweep_by_lang[lang] = sweep_threshold(
                    [relevances[i] for i in idx], [flags[i] for i in idx]
                )
    return report


def cross_lingual_gap(
    metrics: dict[str, dict[str, dict[str, float]]], metric: str = "mrr"
) -> dict[str, dict[str, float]]:
    """Per strategy: FI score minus EN / VI score, and the share of the baseline gap closed."""
    out: dict[str, dict[str, float]] = {}
    base = metrics.get(BASELINE)
    for strategy, by_lang in metrics.items():
        row: dict[str, float] = {}
        for lang in ("en", "vi"):
            gap = by_lang["fi"][metric] - by_lang[lang][metric]
            row[f"gap_{lang}"] = gap
            if base is not None:
                base_gap = base["fi"][metric] - base[lang][metric]
                row[f"closed_{lang}"] = 1 - gap / base_gap if base_gap > 0 else 0.0
        out[strategy] = row
    return out


def check_gate(report: EvalReport, config: dict) -> list[str]:
    """Compare the gated strategy against evals/thresholds.yaml. Empty list = pass."""
    strategy = config["strategy"]
    failures = []
    for lang, minimums in config["min"].items():
        for name, minimum in minimums.items():
            value = report.metrics[strategy][lang][name]
            if not value >= minimum:  # also fails on NaN (metric could not be computed)
                failures.append(f"{strategy}/{lang} {name}: {value:.3f} < {minimum}")
    gaps = cross_lingual_gap(report.metrics)[strategy]
    for lang in ("en", "vi"):
        if gaps[f"gap_{lang}"] > config["max_mrr_gap"]:
            failures.append(
                f"{strategy} MRR gap fi-{lang}: {gaps[f'gap_{lang}']:.3f} > {config['max_mrr_gap']}"
            )
    return failures


def write_json(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": report.thresholds,
        "metrics": report.metrics,
        "gap": cross_lingual_gap(report.metrics),
        "answers": {s: [asdict(a) for a in answers] for s, answers in report.answers.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
