"""Shared pytest fixtures for Persona-RAG tests.

Provides a deterministic 384-dim fake embedder so store/registry tests run
locally without downloading MiniLM weights.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest


class FakeEmbedder:
    """Deterministic 384-dim embedder. Shape matches MiniLM; content seeded off text hash."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in input]

    def embed_query(self, input: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(input, str):
            return self._embed(input)
        return [self._embed(t) for t in input]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dim)]

    # ChromaDB 1.x introspection surface.
    @staticmethod
    def name() -> str:
        return "fake-embedder"

    @staticmethod
    def is_legacy() -> bool:
        return False

    @staticmethod
    def default_space() -> str:
        return "cosine"

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def get_config() -> dict:
        return {}

    @classmethod
    def build_from_config(cls, config: dict) -> FakeEmbedder:
        return cls()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A deterministic 384-dim embedder that doesn't need a network round-trip."""
    return FakeEmbedder()


@pytest.fixture
def personas_dir() -> Path:
    """Repo-relative path to the shipped example personas."""
    return Path(__file__).resolve().parents[1] / "personas"


class FakeBackend:
    """Tiny `LLMBackend`-shaped stub for baseline tests — no model load.

    `format_persona_prompt` returns a string the tests can inspect for both
    persona text presence (B2) and absence (B1). `generate` echoes the prompt
    so tests can assert the generated text is a deterministic function of the
    prompt.
    """

    name = "fake-backend"
    model_id = "fake/fake"
    num_layers = 4
    hidden_dim = 16

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.generate_calls: list[str] = []

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history=None,
    ) -> str:
        history_block = ""
        if history:
            history_block = "\n".join(f"{t.role}: {t.content}" for t in history) + "\n"
        sys_block = (system_text or "").rstrip()
        return f"<<SYS>>\n{sys_block}\n<</SYS>>\n{history_block}<<USER>>\n{user_text}\n<</USER>>"

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.generate_calls.append(prompt)
        # Deterministic: hash a slice of the prompt so tests can compare runs.
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"[fake] generated reply for prompt sha={digest}"


@pytest.fixture
def fake_backend() -> FakeBackend:
    """A no-op `LLMBackend`-shaped stub for baseline tests."""
    return FakeBackend()
