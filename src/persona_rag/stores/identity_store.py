"""Identity store: always-retrieved identity + constraint chunks (ID-RAG pattern)."""

from __future__ import annotations

from typing import Any

from persona_rag.schema.chunker import PersonaChunk
from persona_rag.stores.base import TypedMemoryStore


class IdentityStore(TypedMemoryStore):
    """Holds identity + constraint chunks. Never runtime-writable.

    Unlike the other stores, `query` is overridden to return the full identity+
    constraint set for a persona, not a semantic top-k — "always retrieved" per
    spec. The `text` arg is accepted but unused (kept for interface symmetry).
    """

    COLLECTION_NAME = "persona_identity"
    ALLOW_RUNTIME_WRITE = False

    _KINDS = frozenset({"identity", "constraint"})

    def _accepts_kind(self, kind: str) -> bool:
        return kind in self._KINDS

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        persona_id: str | None = None,
        extra_where: dict[str, Any] | None = None,
    ) -> list[PersonaChunk]:
        """Return the full identity+constraint chunk set for `persona_id`."""
        return self.get_all(persona_id=persona_id)
