"""Results page and charts for the README.

Writes:
    docs/results.md                 all tables, every number from one eval run
    docs/img/mrr_by_language.png    MRR by question language x retrieval strategy
    docs/img/refusal_sweep.png      balanced accuracy of the refusal threshold, per language

Usage: mlrag report   (matplotlib comes with the [report] or [dev] extra)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .evals import BASELINE, EvalReport, cross_lingual_gap
from .text import LANGUAGES

HIGHLIGHT, GREY = "#2a78d6", "#b7b7b2"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LANG_LABEL = {"fi": "FI", "en": "EN", "vi": "VI"}
STRATEGY_LABEL = {
    "translate_bm25": "translate + BM25",
    "ngram": "char n-grams (baseline)",
    "hybrid": "hybrid (RRF)",
    "hybrid_rerank": "hybrid + rerank",
}
METRIC_COLS = [
    ("hit_at_3", "hit@3"),
    ("hit_at_5", "hit@5"),
    ("mrr", "MRR"),
    ("answer_rate", "answer rate"),
    ("key_fact", "key fact"),
    ("citation_valid", "citation valid"),
    ("refusal_acc", "refusal acc."),
]


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def results_markdown(report: EvalReport, data_label: str, gated: str) -> str:
    gap = cross_lingual_gap(report.metrics)
    lines = [
        "# Results",
        "",
        f"Data: {data_label}.",
        "Regenerate with `mlrag report`; every number below comes from that run.",
        "",
        "## Retrieval and answers per question language",
        "",
        "| strategy | lang | " + " | ".join(label for _, label in METRIC_COLS) + " |",
        "|" + " --- |" * (len(METRIC_COLS) + 2),
    ]
    for strategy in report.strategies:
        for lang in LANGUAGES:
            m = report.metrics[strategy][lang]
            cells = " | ".join(_fmt(m[key]) for key, _ in METRIC_COLS)
            lines.append(f"| {STRATEGY_LABEL[strategy]} | {LANG_LABEL[lang]} | {cells} |")
    lines.append(
        "| dense (multilingual-e5) | all | not run here: needs the `[neural]` extra and a model "
        "download |" + " |" * (len(METRIC_COLS) - 1)
    )
    lines += [
        "",
        "## The cross-lingual gap (MRR)",
        "",
        "Gap = MRR on the Finnish questions minus MRR on the same questions in English or "
        f"Vietnamese. 'Closed' = share of the {STRATEGY_LABEL[BASELINE]} gap that the strategy "
        "removes.",
        "",
        "| strategy | gap FI-EN | gap FI-VI | closed EN | closed VI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strategy in report.strategies:
        g = gap[strategy]
        lines.append(
            f"| {STRATEGY_LABEL[strategy]} | {g['gap_en']:.3f} | {g['gap_vi']:.3f} | "
            f"{g.get('closed_en', 0.0):.0%} | {g.get('closed_vi', 0.0):.0%} |"
        )
    lines += [
        "",
        "## Refusal thresholds",
        "",
        "Chosen by a sweep over 0.00-1.00 (step 0.01) on all three languages pooled: balanced "
        "accuracy of answer rate (answerable) and refusal rate (unanswerable), middle of the "
        f"widest best range. `{gated}` uses the shipped default (`DEFAULT_MIN_RELEVANCE`).",
        "",
        "| strategy | relevance score | best range | threshold used |",
        "| --- | --- | --- | --- |",
    ]
    for strategy in report.strategies:
        lo, hi = report.sweeps[strategy].best_range
        kind = "cosine of best hit" if strategy == "ngram" else "IDF-weighted term coverage"
        lines.append(
            f"| {STRATEGY_LABEL[strategy]} | {kind} | {lo:.2f}-{hi:.2f} | "
            f"{report.thresholds[strategy]:.3f} |"
        )
    coverage = {lang: report.metrics[gated][lang]["translation_coverage"] for lang in ("en", "vi")}
    lines += [
        "",
        "## Glossary translator coverage",
        "",
        "Share of content words in the answerable questions that the offline glossary "
        f"translator recognised: EN {coverage['en']:.0%}, VI {coverage['vi']:.0%}. "
        "Unknown words are kept as typed and lower the relevance score.",
        "",
        "## Read with care",
        "",
        "- Small eval set: 40 answerable questions per language, so one question moves a rate "
        "by 0.025. Differences of 0.01-0.03 between strategies are one or two questions.",
        "- I wrote the pages, the questions and the glossary myself. The glossary was written "
        "from the corpus vocabulary and not tuned after the eval ran, but the same author bias "
        "makes these numbers optimistic compared with real newcomers' questions.",
        "- The refusal threshold is chosen on the same questions it is scored on, and the best "
        "range is a single grid point for the coverage-based strategies, so it is fragile. "
        "Two code fixes (the stemmer's final-vowel trim and word-level sentence scoring) "
        "were made after reading eval misses.",
        "- English and Vietnamese answers quote the Finnish source sentence with a glossary "
        "gloss; they are not translations. 'key fact' only checks that the quoted sentence "
        "holds the expected value.",
        "- The n-gram baseline answers nothing in EN/VI: the extractive generator picks "
        "sentences that share words with the question, and an untranslated question shares "
        "none with Finnish text ('n/a' = no answers to check).",
        "",
    ]
    return "\n".join(lines)


def _style(ax: Any, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)


def plot_mrr_by_language(report: EvalReport, path: Path, highlight: str, data_label: str) -> None:
    """Grouped bars: one group per strategy, one bar per question language."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    width, gap = 0.26, 0.14
    for g, strategy in enumerate(report.strategies):
        colour = HIGHLIGHT if strategy == highlight else GREY
        for j, lang in enumerate(LANGUAGES):
            x = g * (3 * width + gap) + j * width
            value = report.metrics[strategy][lang]["mrr"]
            ax.bar(x, value, width * 0.92, color=colour, linewidth=0)
            ax.text(x, value + 0.015, f"{value:.2f}", ha="center", fontsize=8.5, color=INK)
            ax.text(x, -0.06, LANG_LABEL[lang], ha="center", fontsize=8.5, color=INK_2)
    centres = [g * (3 * width + gap) + width for g in range(len(report.strategies))]
    ax.set_xticks(centres, [STRATEGY_LABEL[s] for s in report.strategies])
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    gaps = cross_lingual_gap(report.metrics)
    base, best = gaps[BASELINE], gaps[highlight]
    _style(
        ax,
        f"Translation closes the cross-lingual gap: FI-VI MRR gap {base['gap_vi']:.2f} -> "
        f"{best['gap_vi']:.2f}",
        f"MRR of the first relevant chunk, 40 questions per language. {data_label}.",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_refusal_sweep(report: EvalReport, path: Path, threshold: float, data_label: str) -> None:
    """Balanced accuracy vs threshold: pooled (highlight) and per language (grey)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    pooled = report.sweeps["hybrid"]
    ends = []
    for lang in LANGUAGES:
        sweep = report.sweep_by_lang[lang]
        ax.plot(sweep.thresholds, sweep.balanced, color=GREY, linewidth=1.6)
        ends.append((sweep.balanced[-1], lang))
    # Direct labels at the right end, nudged apart so close lines stay readable.
    last_y = -1.0
    for y, lang in sorted(ends):
        y = max(y, last_y + 0.04)
        ax.text(1.01, y, LANG_LABEL[lang], fontsize=9, color=INK, va="center")
        last_y = y
    ax.plot(pooled.thresholds, pooled.balanced, color=HIGHLIGHT, linewidth=2.2)
    ax.axvline(threshold, color=INK_2, linewidth=0.9, linestyle="--")
    best = max(pooled.balanced)
    ax.text(
        threshold + 0.012,
        0.08,
        f"chosen threshold {threshold:.2f}\npooled balanced accuracy {best:.2f}",
        fontsize=9,
        color=INK,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("refusal threshold on relevance (IDF-weighted term coverage)", color=INK_2)
    per_lang = {lang: report.metrics["hybrid"][lang]["answer_rate"] for lang in LANGUAGES}
    _style(
        ax,
        f"One shared threshold answers {per_lang['fi']:.0%} of FI but only "
        f"{per_lang['vi']:.0%} of VI questions",
        f"Hybrid strategy. Blue: all languages pooled; grey: per language. {data_label}.",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_report(
    report: EvalReport, docs_dir: Path, data_label: str, gated: str, chart_label: str
) -> Path:
    img = docs_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "results.md"
    path.write_text(results_markdown(report, data_label, gated), encoding="utf-8")
    plot_mrr_by_language(report, img / "mrr_by_language.png", gated, chart_label)
    plot_refusal_sweep(report, img / "refusal_sweep.png", report.thresholds[gated], chart_label)
    return path
