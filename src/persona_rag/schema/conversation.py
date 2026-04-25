"""Pydantic schema for hand-authored multi-turn conversations.

The drift-trajectory experiment requires conversations authored by hand — the
methodological point is that assistant turns isolate generation-time content
from prompt-time content. Every conversation file is loaded through this
schema; any malformed file raises a clear validation error before the
experiment runs.

Layout on disk:

    benchmarks_data/drift_trajectory/<persona_id>/{in_persona,drifting}.yaml

Fields:

- ``persona_id``: must match a YAML in ``personas/``.
- ``condition``: ``"in_persona"`` or ``"drifting"``.
- ``turns``: ordered list of ``{role, text}`` pairs. Even-indexed turns must
  be ``"user"``, odd-indexed ``"assistant"``. Length must equal
  ``2 * n_pairs``.
- ``annotations`` (drifting only): per-turn drift status — ``"in"`` /
  ``"subtle"`` / ``"clear"`` / ``"break"`` — so the analysis script can
  compare authored drift level against measured drift signal.

The class lives next to ``persona.py`` because it shares the same Pydantic
patterns and lifecycle (load → validate → freeze).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ConversationCondition = Literal["in_persona", "drifting"]
TurnRole = Literal["user", "assistant"]
DriftLevel = Literal["in", "subtle", "clear", "break"]


class ConversationTurn(BaseModel):
    """One turn — either a user message or an assistant response."""

    role: TurnRole
    text: str = Field(min_length=1, max_length=4000)
    drift_level: DriftLevel | None = None


class DriftTrajectoryConversation(BaseModel):
    """A 6-turn (12-message) hand-authored conversation.

    Both the in-persona and drifting conditions share user turns by design
    (spec §1: "User turns are identical to in-persona conversation"). The
    schema does not enforce that across files — the analysis script asserts
    it. The schema enforces only the per-file invariants.
    """

    persona_id: str = Field(min_length=1, max_length=120)
    condition: ConversationCondition
    n_pairs: int = Field(default=6, ge=1, le=20)
    turns: list[ConversationTurn]
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> DriftTrajectoryConversation:
        expected_len = 2 * self.n_pairs
        if len(self.turns) != expected_len:
            raise ValueError(
                f"conversation must have {expected_len} turns "
                f"({self.n_pairs} pairs x 2 roles); got {len(self.turns)}"
            )
        for i, turn in enumerate(self.turns):
            expected_role: TurnRole = "user" if i % 2 == 0 else "assistant"
            if turn.role != expected_role:
                raise ValueError(
                    f"turn {i}: expected role {expected_role!r}, got {turn.role!r}"
                )
        # Annotations are mandatory on drifting conversations and forbidden on
        # in-persona ones (the in-persona condition has, by definition, no
        # drift to annotate).
        annotated = [
            t for t in self.turns if t.role == "assistant" and t.drift_level is not None
        ]
        if self.condition == "drifting":
            if len(annotated) != self.n_pairs:
                raise ValueError(
                    "drifting conversation: every assistant turn must carry a "
                    f"drift_level; got {len(annotated)} of {self.n_pairs} annotated"
                )
        else:
            if annotated:
                raise ValueError(
                    "in_persona conversation: assistant turns must not carry "
                    "a drift_level (the condition is 'no drift')."
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> DriftTrajectoryConversation:
        """Load + validate a conversation YAML."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def user_turn_texts(self) -> list[str]:
        """Plain list of user-turn texts in turn order."""
        return [t.text for t in self.turns if t.role == "user"]

    def assistant_turn_texts(self) -> list[str]:
        """Plain list of assistant-turn texts in turn order."""
        return [t.text for t in self.turns if t.role == "assistant"]


def assert_user_turns_match(
    convs: Iterable[DriftTrajectoryConversation],
) -> None:
    """Sanity check: across a (in_persona, drifting) pair, user turns must be identical.

    User turns are the experimental control: both conditions must share them
    verbatim. Without this, drift differences could be attributed to prompt
    content rather than generation content.

    Raises ``ValueError`` listing the divergent turn indices.
    """
    convs = list(convs)
    if len(convs) < 2:
        return
    base_user = convs[0].user_turn_texts()
    for c in convs[1:]:
        their = c.user_turn_texts()
        if their != base_user:
            mismatches = [
                i for i, (a, b) in enumerate(zip(base_user, their, strict=True)) if a != b
            ]
            raise ValueError(
                f"user-turn mismatch between {convs[0].condition!r} and {c.condition!r} "
                f"for persona {c.persona_id!r}: divergent turn indices {mismatches}"
            )
