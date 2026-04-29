"""Shared evaluation-metric contract.

Every metric in this package implements the :class:`Metric` Protocol and
returns a :class:`MetricResult`. The result carries an aggregate value
plus a per-conversation distribution and a per-persona breakdown so the
runner can write a long-form CSV without reaching back into the metric
implementation.

The contract is intentionally narrow:

- ``score(conversations, persona)`` -- score a list of conversations under
  one persona. Multi-persona aggregation is the runner's job (one Metric
  call per persona, then aggregation across personas).
- The returned ``value`` is the headline number (typically a mean across
  the per-conversation distribution). Per-conversation values are
  preserved so distributions (std, P95, per-persona) can be computed
  downstream.
- ``metadata`` is free-form: each metric records configuration,
  judge-model identity, parse-failure counts, etc., so reproducibility
  audits can read the metric's exact behaviour off the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from persona_rag.schema.persona import Persona


@dataclass(frozen=True, slots=True)
class ScoredTurn:
    """One assistant turn extracted from a conversation, with its turn index.

    The metrics work on assistant turns; user turns are passed-through
    context. ``turn_index`` is the assistant turn's position in the
    full conversation (0 = first assistant reply, 1 = second, ...).
    """

    turn_index: int
    user_text: str
    assistant_text: str


@dataclass(frozen=True, slots=True)
class EvalConversation:
    """A conversation as the metrics see it.

    ``conversation_id`` is unique within a run. ``mechanism`` /
    ``persona_id`` are recorded so the metric output can be joined back
    to the source. ``turns`` is the (user, assistant) pair sequence --
    metrics typically iterate over these directly.

    ``metadata`` carries pipeline-specific fields the cost tracker and
    drift-quality metric read (e.g. ``gate_should_gate`` per turn).
    """

    conversation_id: str
    mechanism: str
    persona_id: str
    turns: tuple[ScoredTurn, ...]
    per_turn_metadata: tuple[dict[str, Any], ...] = ()


class MetricResult(BaseModel):
    """Output of one ``Metric.score`` call.

    ``value`` is the aggregate (typically the mean across
    ``per_conversation``). ``per_conversation`` is the distribution
    used for std / P95 / per-persona slicing. ``per_persona`` is a
    convenience map for one-persona-at-a-time slicing; for a single
    persona it has one entry.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    per_conversation: list[float] = Field(default_factory=list)
    per_persona: dict[str, float] = Field(default_factory=dict)
    per_conversation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Score a list of conversations under one persona."""

    @property
    def name(self) -> str:
        """Stable identifier (e.g. ``"minicheck_self_fact"``)."""

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        """Compute the metric over ``conversations`` for ``persona``."""


__all__ = [
    "EvalConversation",
    "Metric",
    "MetricResult",
    "ScoredTurn",
]
