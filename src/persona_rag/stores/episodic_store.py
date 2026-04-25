"""Episodic store: runtime-writable, decay-ranked conversation memory."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from persona_rag.schema.chunker import PersonaChunk
from persona_rag.stores.base import TypedMemoryStore
from persona_rag.stores.decay import DEFAULT_TAU, combine_similarity_and_decay, decay_score


class EpisodicStore(TypedMemoryStore):
    """Stores `episodic` chunks. Runtime-writable. Decay-weighted retrieval.

    `tau` controls the Ebbinghaus decay half-life. Default is 24h;
    Hydra-configurable via ``config/stores/default.yaml``.
    """

    COLLECTION_NAME = "persona_episodic"
    ALLOW_RUNTIME_WRITE = True

    def __init__(
        self,
        *args: Any,
        tau: timedelta = DEFAULT_TAU,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tau = tau

    @property
    def tau(self) -> timedelta:
        """Ebbinghaus decay time constant."""
        return self._tau

    def _accepts_kind(self, kind: str) -> bool:
        return kind == "episodic"

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        persona_id: str | None = None,
        now: datetime | None = None,
        extra_where: dict[str, Any] | None = None,
    ) -> list[PersonaChunk]:
        """Top-k by `semantic_similarity * decay_score(decay_t0, now, tau)`.

        `now` pins the reference time (useful for deterministic tests); defaults
        to `datetime.now(timezone.utc)`.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        # Pull a generous candidate pool so the decay re-rank can reshuffle.
        candidate_k = max(top_k * 4, top_k)
        candidates = super().query(
            text,
            top_k=candidate_k,
            persona_id=persona_id,
            extra_where=extra_where,
        )
        if not candidates:
            return []

        now = now if now is not None else datetime.now(UTC)

        scored: list[tuple[float, PersonaChunk]] = []
        for chunk in candidates:
            similarity = _chroma_distance_to_similarity(chunk.distance)
            decay_t0 = _parse_iso_datetime(chunk.metadata.get("decay_t0"))
            if decay_t0 is None:
                continue
            decay = decay_score(decay_t0, now=now, tau=self._tau)
            combined = combine_similarity_and_decay(similarity, decay)
            scored.append((combined, replace(chunk, distance=1.0 - combined)))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def _chroma_distance_to_similarity(distance: float | None) -> float:
    """Convert ChromaDB cosine distance to similarity in [0, 1]. None → 0.0."""
    if distance is None:
        return 0.0
    # ChromaDB's cosine distance is in [0, 2]; similarity = 1 - distance/2 keeps it in [0, 1].
    sim = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, sim))


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
