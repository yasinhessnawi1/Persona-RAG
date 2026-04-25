"""Self-facts store: near-immutable atomic self-claims. Top-k semantic retrieval."""

from __future__ import annotations

from persona_rag.stores.base import TypedMemoryStore


class SelfFactsStore(TypedMemoryStore):
    """Stores `self_fact` chunks. Not runtime-writable."""

    COLLECTION_NAME = "persona_self_facts"
    ALLOW_RUNTIME_WRITE = False

    def _accepts_kind(self, kind: str) -> bool:
        return kind == "self_fact"
