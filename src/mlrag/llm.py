"""A tiny chat-model interface with two HTTP providers.

Translation and answer generation only need "send a system + user message, get text back".
Keeping that behind one small protocol means the offline CI path never imports or calls a model,
and switching between a local Ollama model and a hosted API is one setting.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Local models through Ollama's /api/chat endpoint (no key, runs on your machine)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def complete(self, system: str, user: str) -> str:
        resp = self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["message"]["content"]).strip()


class OpenAIClient:
    """Any OpenAI-compatible /chat/completions API. The key comes from OPENAI_API_KEY."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("set OPENAI_API_KEY to use the hosted provider")
        self.name = f"openai:{model}"
        self.model = model
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()


def make_client(provider: str) -> LLMClient | None:
    """'offline' -> None (deterministic glossary + extractive path), 'ollama', or 'openai'."""
    if provider == "offline":
        return None
    if provider == "ollama":
        return OllamaClient(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
    if provider == "openai":
        return OpenAIClient(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    raise ValueError(f"unknown provider {provider!r}; use offline, ollama or openai")
