import json
from pathlib import Path

import pytest
from conftest import QUESTIONS, ROOT
from fastapi.testclient import TestClient

from mlrag.answer import Assistant
from mlrag.api import create_app
from mlrag.cli import main


@pytest.fixture
def client(assistant: Assistant) -> TestClient:
    return TestClient(create_app(assistant=assistant))


def test_ask_detects_language_and_returns_citations_and_disclaimer(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/ask", json={"question": "Paljonko vuorokausilippu maksaa?"}).json()
    assert body["language"] == "fi" and not body["refused"]
    assert body["citations_valid"] and body["citations"][0].startswith("joukkoliikenne-liput#")
    assert "not official guidance" in body["disclaimer"]


def test_ask_with_explicit_language_and_refusal(client: TestClient):
    body = client.post(
        "/ask", json={"question": "How much does a fishing licence cost?", "language": "en"}
    ).json()
    assert body["refused"] and body["answer"] == "I can't find this in the documents."
    assert body["citations"] == []


def test_ask_rejects_bad_input(client: TestClient):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "Hei?", "language": "sv"}).status_code == 422


def test_cli_index_ask_eval_and_report(tmp_path: Path, capsys):
    index = str(tmp_path / "index")
    corpus = str(ROOT / "corpus")
    assert main(["--corpus", corpus, "--index", index, "index"]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] > 50

    assert main(["--corpus", corpus, "--index", index, "ask", "Vé ngày giá bao nhiêu?"]) == 0
    out = capsys.readouterr().out
    assert "language=vi" in out and "joukkoliikenne-liput#" in out

    common = ["--questions", str(QUESTIONS), "--out", str(tmp_path / "reports")]
    gate = ["--thresholds", str(ROOT / "evals" / "thresholds.yaml")]
    assert main(["--corpus", corpus, "--index", index, "eval", *common, *gate]) == 0
    assert "eval gate passed" in capsys.readouterr().out
    assert (tmp_path / "reports" / "eval_results.json").exists()

    strict = tmp_path / "strict.yaml"
    strict.write_text("strategy: hybrid\nmax_mrr_gap: 0.0\nmin: {fi: {mrr: 1.01}}\n")
    assert (
        main(["--corpus", corpus, "--index", index, "eval", *common, "--thresholds", str(strict)])
        == 1
    )

    docs = tmp_path / "docs"
    assert main(["--corpus", corpus, "--index", index, "report", *common, "--docs", str(docs)]) == 0
    results = (docs / "results.md").read_text(encoding="utf-8")
    assert "cross-lingual gap" in results and "not run here" in results
    assert (docs / "img" / "mrr_by_language.png").stat().st_size > 10_000
    assert (docs / "img" / "refusal_sweep.png").exists()
