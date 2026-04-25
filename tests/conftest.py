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
