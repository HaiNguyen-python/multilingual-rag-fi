import json

import httpx
import pytest

from mlrag.answer import (
    REFUSALS,
    Assistant,
    LLMGenerator,
    is_refusal,
    validate_citations,
)
from mlrag.llm import OllamaClient, OpenAIClient, make_client
from mlrag.retrieval import Retriever


@pytest.mark.parametrize(
    ("text", "valid", "unknown"),
    [
        ("Kortti on maksuton. [kirjastokortti#0]", True, ()),
        ("No citation at all.", False, ()),
        ("Mixed [kirjastokortti#0] and invented [uimahalli#9].", False, ("uimahalli#9",)),
    ],
)
def test_citation_validator(text, valid, unknown):
    check = validate_citations(text, {"kirjastokortti#0", "kirjastokortti#1"})
    assert check.valid is valid
    assert check.unknown == unknown


def test_refusal_is_recognised_in_every_language():
    assert all(is_refusal(r) for r in REFUSALS.values())
    assert not is_refusal("Uusi kortti maksaa 3 euroa.")


def test_finnish_answer_quotes_the_source_sentence_with_a_citation(assistant: Assistant):
    a = assistant.ask("Paljonko vuorokausilippu maksaa?")
    assert a.lang == "fi" and not a.refused
    assert "9 euroa" in a.text
    assert a.citations.valid and a.citations.cited[0].startswith("joukkoliikenne-liput#")


def test_english_and_vietnamese_answers_use_templates_and_say_they_are_quotes(
    assistant: Assistant,
):
    en = assistant.ask("How much is a 30-day season ticket?")
    assert en.text.startswith('The Finnish page "Matkakortti ja kausilippu" (Travel card')
    assert "55 euroa" in en.text and "not a translation" in en.text
    assert "kausilippu = season ticket" in en.text
    vi = assistant.ask("Vé tháng 30 ngày giá bao nhiêu?")
    assert vi.lang == "vi" and vi.text.startswith("Trang tiếng Phần Lan")
    assert "không phải bản dịch" in vi.text and vi.citations.valid


def test_low_relevance_question_is_refused_in_its_own_language(assistant: Assistant):
    a = assistant.ask("Giấy phép câu cá giá bao nhiêu?")
    assert a.refused and a.text == REFUSALS["vi"]
    assert a.relevance < assistant.min_relevance


def _mock(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_llm_answer_with_an_invented_citation_is_flagged(retriever: Retriever):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        reply = "Vuorokausilippu maksaa 9 euroa. [joukkoliikenne-liput#2] [uimahalli#7]"
        return httpx.Response(200, json={"message": {"content": reply}})

    client = OllamaClient(transport=_mock(handler))
    a = Assistant(retriever, LLMGenerator(client), min_relevance=0.0).ask(
        "How much is a day ticket?", "en"
    )
    assert not a.citations.valid and a.citations.unknown == ("uimahalli#7",)
    body = seen["body"]
    assert isinstance(body, dict) and "in English" in body["messages"][0]["content"]


def test_openai_client_sends_the_key_and_parses_the_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hei \n"}}]})

    client = OpenAIClient(api_key="test-key", transport=_mock(handler))
    assert client.complete("system", "user") == "Hei"


def test_make_client_never_requires_a_key_for_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert make_client("offline") is None
    assert make_client("ollama") is not None
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_client("openai")
    with pytest.raises(ValueError, match="unknown provider"):
        make_client("other")
