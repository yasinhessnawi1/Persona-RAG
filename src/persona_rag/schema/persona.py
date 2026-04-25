"""Persona schema (v0.3): typed memory models for identity, self-facts, worldview, episodic."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------- constraints on field sizes (centralised so tests can import) ----------

SELF_FACT_MAX = 500
WORLDVIEW_CLAIM_MAX = 500
WORLDVIEW_DOMAIN_MAX = 50
CONSTRAINT_MAX = 200
BACKGROUND_MAX = 1000
PERSONA_ID_MAX = 100

EPISTEMIC_TAGS = ("fact", "belief", "hypothesis", "contested")

# valid_time grammar: "always" | "YYYY" | "YYYY-YYYY" | "YYYY-"
_VALID_TIME_RE = re.compile(r"^(always|\d{4}|\d{4}-\d{4}|\d{4}-)$")


class _StrictModel(BaseModel):
    """Base: reject unknown keys so YAML typos surface as validation errors."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


# ---------- typed memory models ----------


class SelfFact(_StrictModel):
    """An atomic claim the persona makes about themselves. Near-immutable."""

    fact: str = Field(..., min_length=1, max_length=SELF_FACT_MAX)
    epistemic: Literal["fact"] = "fact"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class WorldviewClaim(_StrictModel):
    """A stance, belief, or contested claim the persona holds — revisable, epistemically tagged."""

    claim: str = Field(..., min_length=1, max_length=WORLDVIEW_CLAIM_MAX)
    domain: str = Field(
        ...,
        min_length=1,
        max_length=WORLDVIEW_DOMAIN_MAX,
        pattern=r"^[a-z_]+$",
    )
    epistemic: Literal["fact", "belief", "hypothesis", "contested"] = "belief"
    valid_time: str = "always"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("valid_time")
    @classmethod
    def _valid_time_grammar(cls, v: str) -> str:
        if not _VALID_TIME_RE.match(v):
            raise ValueError(
                f"valid_time must match 'always' | 'YYYY' | 'YYYY-YYYY' | 'YYYY-'; got {v!r}"
            )
        # If YYYY-YYYY, require start <= end.
        if "-" in v and v != "always" and not v.endswith("-"):
            start_s, end_s = v.split("-", 1)
            if end_s and int(start_s) > int(end_s):
                raise ValueError(f"valid_time range start > end: {v!r}")
        return v


class PersonaIdentity(_StrictModel):
    """Who this persona is: name, role, background, and always-retrieved constraints."""

    name: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    background: str = Field(..., min_length=1, max_length=BACKGROUND_MAX)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("constraints")
    @classmethod
    def _constraints_non_empty_and_sized(cls, items: list[str]) -> list[str]:
        for i, item in enumerate(items):
            if not item or not item.strip():
                raise ValueError(f"identity.constraints[{i}] is empty")
            if len(item) > CONSTRAINT_MAX:
                raise ValueError(
                    f"identity.constraints[{i}] too long ({len(item)} > {CONSTRAINT_MAX})"
                )
        return items


class EpisodicEntry(_StrictModel):
    """A runtime-added memory of something that happened in the conversation."""

    text: str = Field(..., min_length=1)
    timestamp: datetime
    turn_id: int = Field(..., ge=0)
    decay_t0: datetime

    @field_validator("timestamp", "decay_t0")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # Treat naive timestamps as UTC rather than erroring — authoring-friendly.
            return v.replace(tzinfo=UTC)
        return v


# ---------- the top-level persona document ----------


class Persona(_StrictModel):
    """A validated persona document.

    Loaded from YAML via `Persona.from_yaml`. `persona_id` is the YAML field
    if present, else derived from the source filename stem.
    """

    persona_id: str | None = Field(default=None, min_length=1, max_length=PERSONA_ID_MAX)
    identity: PersonaIdentity
    self_facts: list[SelfFact] = Field(default_factory=list)
    worldview: list[WorldviewClaim] = Field(default_factory=list)
    episodic: list[EpisodicEntry] = Field(default_factory=list)

    @field_validator("persona_id")
    @classmethod
    def _persona_id_slug(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"persona_id must be alphanumeric + _/-: got {v!r}")
        return v

    # ---------- IO ----------

    @classmethod
    def from_yaml(cls, path: Path) -> Persona:
        """Load and validate a persona YAML file.

        Filename stem becomes persona_id unless the YAML sets one explicitly.
        Tolerates a top-level ``persona:`` wrapper.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
        if set(raw.keys()) == {"persona"} and isinstance(raw["persona"], dict):
            raw = raw["persona"]
        if not raw.get("persona_id"):
            raw["persona_id"] = path.stem
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        """Serialize this persona to YAML. Datetime fields become ISO-8601 strings."""
        path = Path(path)
        payload: dict[str, Any] = self.model_dump(mode="json", exclude_none=False)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
