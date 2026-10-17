"""CLI: mlrag index | ask | eval | report | serve"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .answer import DEFAULT_MIN_RELEVANCE, DISCLAIMER
from .evals import EvalReport
from .factory import Settings, make_assistant, make_generator, make_retriever
from .retrieval import LEXICAL_STRATEGIES, STRATEGIES

GATED = "hybrid"
DATA_LABEL = (
    "synthetic, illustrative corpus (18 Finnish pages written for this project) and a "
    "hand-written eval set (40 answerable + 8 unanswerable questions, each in FI, EN and VI); "
    "offline glossary translator and extractive answers"
)
CHART_LABEL = "Synthetic, illustrative corpus and questions"


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    return Settings(
        corpus_dir=Path(args.corpus) if args.corpus else base.corpus_dir,
        index_dir=Path(args.index) if args.index else base.index_dir,
        provider=args.provider or base.provider,
        neural=base.neural,
    )


def cmd_index(args: argparse.Namespace) -> int:
    from .index import build_index

    s = _settings(args)
    print(json.dumps(asdict(build_index(s.corpus_dir, s.index_dir)), indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    assistant = make_assistant(_settings(args), args.strategy)
    answer = assistant.ask(args.question, args.lang)
    print(answer.text)
    print(
        f"\nlanguage={answer.lang}  finnish query={answer.query_fi!r}  "
        f"relevance={answer.relevance} (threshold {assistant.min_relevance})"
    )
    print("sources: " + ", ".join(c for c, _ in answer.retrieved[: assistant.top_k]))
    print(DISCLAIMER)
    return 0


def _run_eval(args: argparse.Namespace) -> EvalReport:
    from . import evals

    s = _settings(args)
    retriever = make_retriever(s)
    generator = make_generator(retriever)
    items = evals.load_eval_set(Path(args.questions))
    strategies = list(LEXICAL_STRATEGIES) + (["dense"] if s.neural else [])
    report = evals.evaluate(
        retriever, generator, items, strategies, fixed_thresholds={GATED: DEFAULT_MIN_RELEVANCE}
    )
    evals.write_json(report, Path(args.out) / "eval_results.json")
    return report


def cmd_eval(args: argparse.Namespace) -> int:
    import yaml

    from . import evals

    report = _run_eval(args)
    for strategy, by_lang in report.metrics.items():
        for lang, m in by_lang.items():
            print(
                f"{strategy:14s} {lang}  hit@3={m['hit_at_3']:.2f} hit@5={m['hit_at_5']:.2f} "
                f"mrr={m['mrr']:.2f} answer={m['answer_rate']:.2f} "
                f"refusal={m['refusal_acc']:.2f} cite={m['citation_valid']:.2f}"
            )
    swept = report.sweeps[GATED].best
    print(f"\nswept threshold for {GATED}: {swept} (shipped default {DEFAULT_MIN_RELEVANCE})")
    config = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    failures = evals.check_gate(report, config)
    if failures:
        print("EVAL GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("eval gate passed")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    report = _run_eval(args)
    path = write_report(report, Path(args.docs), DATA_LABEL, GATED, CHART_LABEL)
    print(f"wrote {path} and charts in {Path(args.docs) / 'img'}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlrag", description=__doc__)
    parser.add_argument("--corpus", help="corpus folder (default: corpus or $MLRAG_CORPUS_DIR)")
    parser.add_argument("--index", help="index folder (default: index or $MLRAG_INDEX_DIR)")
    parser.add_argument(
        "--provider", choices=["offline", "ollama", "openai"], help="default: $MLRAG_PROVIDER"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="build or update the index").set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--lang", choices=["fi", "en", "vi"], help="default: detected")
    ask.add_argument("--strategy", choices=STRATEGIES, default=GATED)
    ask.set_defaults(func=cmd_ask)

    for name, func, text in [
        ("eval", cmd_eval, "evaluate all strategies and apply the CI gate"),
        ("report", cmd_report, "write docs/results.md and charts"),
    ]:
        p = sub.add_parser(name, help=text)
        p.add_argument("--questions", default="evals/questions.yaml")
        p.add_argument("--thresholds", default="evals/thresholds.yaml")
        p.add_argument("--out", default="reports")
        p.add_argument("--docs", default="docs")
        p.set_defaults(func=func)

    serve = sub.add_parser("serve", help="run the FastAPI app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
