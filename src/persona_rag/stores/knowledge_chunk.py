"""Knowledge-corpus retrieval-result type, mirror of `PersonaChunk`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeChunk:
    """One retrieved chunk from the knowledge (domain) vector store.

    Mirrors :class:`persona_rag.schema.chunker.PersonaChunk` so callers consume
    a uniform shape across the persona and knowledge stores. Metadata fields
    differ from `PersonaChunk` (knowledge has `doc_id`/`source`/`chunk_ix`;
    persona has `kind`/`epistemic`/etc.) — a Union type hint in `Response` is
    clearer than a shared superclass.

    All metadata values are strings to satisfy ChromaDB 1.5's scalar-only
    metadata constraint. `chunk_ix` is parsed back to int by callers that need it.

    Attributes
    ----------
    id:
        Globally-unique chunk id, prefixed with `doc_id` so multiple documents
        coexist in one collection without collision.
    text:
        The chunk's natural-language content as embedded.
    metadata:
        Scalar key/value pairs — at minimum ``doc_id``, ``source``, ``chunk_ix``.
    distance:
        Populated at retrieval time from ChromaDB's cosine-distance field.
        ``None`` at index time.
    """

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    distance: float | None = None
