"""Retrieval-pipeline contract: every pipeline implements `RetrievalPipeline`.

A `RetrievalPipeline` takes a query + persona + history and returns a structured
`Response`. The contract is typed end-to-end so future pipelines cannot silently
drift the shape:

- `Response.retrieved_persona: dict[ChunkKind, list[PersonaChunk]]` —
  per-typed-store retrieval results, keyed by `ChunkKind` rather than loose
  strings.
- `Response.retrieved_knowledge: list[KnowledgeChunk]` — knowledge-store
  hybrid-retrieval result.
- `Response.steering_applied: bool` and `Response.drift_signal: float | None` —
  reserved for activation-steering and drift-aware pipelines that may land
  later. Default `False` / `None` for the prompt-only baselines shipped here.
- `extra="forbid"` so a typo in a downstream pipeline ("retrieved_personas")
  fails at instantiation rather than silently dropping the field.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from persona_rag.schema.chunker import ChunkKind, PersonaChunk
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_chunk import KnowledgeChunk


class Turn(BaseModel):
    """One past chat turn threaded into the retrieval pipeline as conversation history.

    `role` is `"user"` or `"assistant"` — system messages are persona-derived and
    are reconstructed by the baseline at prompt-assembly time, not threaded in.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1)


class Response(BaseModel):
    """Structured output of every `RetrievalPipeline.respond` call.

    `extra="forbid"` is intentional — see module docstring.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str
    retrieved_knowledge: list[KnowledgeChunk] = Field(default_factory=list)
    retrieved_persona: dict[ChunkKind, list[PersonaChunk]] = Field(default_factory=dict)
    prompt_used: str
    steering_applied: bool = False
    drift_signal: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RetrievalPipeline(Protocol):
    """A pipeline that takes a query + persona + history and returns a `Response`."""

    @property
    def name(self) -> str:
        """Short identifier for logging / wandb run names (e.g. ``"vanilla_rag"``)."""

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Turn] | None = None,
        *,
        seed: int | None = None,
    ) -> Response:
        """Run the full retrieval + generation pipeline for one query.

        `seed` is passed through to the backend's generation call. ``None``
        means "let the backend pick" (typically the backend's own default for
        greedy decoding). Multi-seed dispatch is the caller's responsibility:
        invoke `respond` once per seed, log each result.
        """
