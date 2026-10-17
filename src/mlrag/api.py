"""FastAPI app: POST /ask and GET /health.

The assistant is created once at startup and injected with ``Depends``, so tests can pass a
small in-memory assistant instead of the real index.
"""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .answer import DISCLAIMER, Assistant
from .factory import Settings, make_assistant


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    language: Literal["fi", "en", "vi"] | None = Field(
        default=None, description="Question language; detected when omitted."
    )


class Source(BaseModel):
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    language: str
    refused: bool
    citations: list[str]
    citations_valid: bool
    sources: list[Source]
    relevance: float
    query_fi: str
    strategy: str
    disclaimer: str = DISCLAIMER


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    app = FastAPI(title="Multilingual RAG over Finnish public-service pages", version="0.1.0")
    state: dict[str, Assistant] = {}

    def get_assistant() -> Assistant:
        if "assistant" not in state:
            state["assistant"] = assistant or make_assistant(settings or Settings.from_env())
        return state["assistant"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, bot: Assistant = Depends(get_assistant)) -> AskResponse:
        a = bot.ask(req.question, req.language)
        return AskResponse(
            answer=a.text,
            language=a.lang,
            refused=a.refused,
            citations=list(a.citations.cited),
            citations_valid=a.citations.valid,
            sources=[Source(chunk_id=c, score=s) for c, s in a.retrieved[: bot.top_k]],
            relevance=a.relevance,
            query_fi=a.query_fi,
            strategy=bot.strategy,
        )

    return app
