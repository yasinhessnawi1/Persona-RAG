"""Persona chunker (v0.3): one chunk per atomic item, routed to the correct typed store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from persona_rag.schema.persona import Persona

ChunkKind = Literal["identity", "constraint", "self_fact", "worldview", "episodic"]


@dataclass(frozen=True)
class PersonaChunk:
    """One atomic unit of persona content. Metadata fields depend on `kind`.

    All metadata values are strings so they pass ChromaDB's scalar-only
    metadata constraint. Float confidences are stringified; the stores parse
    them back at query time.
    """

    id: str
    text: str
    kind: ChunkKind
    metadata: dict[str, str] = field(default_factory=dict)
    distance: float | None = None


def chunk_persona(persona: Persona) -> list[PersonaChunk]:
    """Flatten a `Persona` into one `PersonaChunk` per atomic item.

    Mapping:
      - identity -> 1 chunk combining name + role + background, kind="identity"
      - each constraint -> 1 chunk, kind="constraint"
      - each self_fact -> 1 chunk, kind="self_fact" (+ confidence, epistemic)
      - each worldview claim -> 1 chunk, kind="worldview"
                                (+ domain, epistemic, valid_time, confidence)
      - each episodic entry -> 1 chunk, kind="episodic"
                                (+ timestamp, decay_t0, turn_id)
    """
    if persona.persona_id is None:
        raise ValueError(
            "Persona.persona_id is None — set it via from_yaml() or explicitly before chunking."
        )
    pid = persona.persona_id
    chunks: list[PersonaChunk] = []

    # --- identity (one chunk) ---
    identity_text = (
        f"Name: {persona.identity.name}\n"
        f"Role: {persona.identity.role}\n"
        f"Background: {persona.identity.background}"
    )
    chunks.append(
        PersonaChunk(
            id=f"{pid}:identity:0",
            text=identity_text,
            kind="identity",
            metadata={
                "persona_id": pid,
                "kind": "identity",
                "source_field": "identity",
            },
        )
    )

    # --- constraints (one chunk each) ---
    for i, constraint in enumerate(persona.identity.constraints):
        chunks.append(
            PersonaChunk(
                id=f"{pid}:constraint:{i}",
                text=constraint,
                kind="constraint",
                metadata={
                    "persona_id": pid,
                    "kind": "constraint",
                    "source_field": "identity.constraints",
                },
            )
        )

    # --- self_facts (one chunk each) ---
    for i, sf in enumerate(persona.self_facts):
        chunks.append(
            PersonaChunk(
                id=f"{pid}:self_fact:{i}",
                text=sf.fact,
                kind="self_fact",
                metadata={
                    "persona_id": pid,
                    "kind": "self_fact",
                    "source_field": "self_facts",
                    "epistemic": sf.epistemic,
                    "confidence": f"{sf.confidence:.4f}",
                },
            )
        )

    # --- worldview (one chunk each) ---
    for i, wv in enumerate(persona.worldview):
        chunks.append(
            PersonaChunk(
                id=f"{pid}:worldview:{i}",
                text=wv.claim,
                kind="worldview",
                metadata={
                    "persona_id": pid,
                    "kind": "worldview",
                    "source_field": "worldview",
                    "domain": wv.domain,
                    "epistemic": wv.epistemic,
                    "valid_time": wv.valid_time,
                    "confidence": f"{wv.confidence:.4f}",
                },
            )
        )

    # --- episodic (one chunk each) ---
    for i, ep in enumerate(persona.episodic):
        chunks.append(
            PersonaChunk(
                id=f"{pid}:episodic:{i}",
                text=ep.text,
                kind="episodic",
                metadata={
                    "persona_id": pid,
                    "kind": "episodic",
                    "source_field": "episodic",
                    "timestamp": ep.timestamp.isoformat(),
                    "decay_t0": ep.decay_t0.isoformat(),
                    "turn_id": str(ep.turn_id),
                },
            )
        )

    return chunks


def chunks_by_kind(chunks: list[PersonaChunk]) -> dict[ChunkKind, list[PersonaChunk]]:
    """Group chunks by kind for routing into the four typed stores."""
    out: dict[ChunkKind, list[PersonaChunk]] = {
        "identity": [],
        "constraint": [],
        "self_fact": [],
        "worldview": [],
        "episodic": [],
    }
    for c in chunks:
        out[c.kind].append(c)
    return out
