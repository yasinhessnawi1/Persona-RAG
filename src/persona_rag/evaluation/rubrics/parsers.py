"""Parsers + score schemas for PoLL rubric outputs.

Two output surfaces exist per rubric:

- **Native Prometheus**: ``Feedback: <text> [RESULT] <int>`` parsed via
  regex. Prometheus-2 was trained on this surface; the ``[RESULT] N``
  marker is reliable when the prompt mirrors the model card's exact
  template.
- **JSON**: ``{"<dim>": int, ...}`` parsed via balanced-brace extraction
  + ``json.loads``. Used for the Qwen and Llama judges, which were not
  trained on the Prometheus surface but reliably honour JSON-output
  rubrics.

Both routes return :class:`PersonaAdherenceScore` /
:class:`TaskQualityScore` Pydantic models so downstream aggregation
is uniform regardless of which judge produced the score.
"""

from __future__ import annotations

import json
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

PERSONA_ADHERENCE_DIMENSIONS: Final[tuple[str, ...]] = (
    "self_facts",
    "worldview",
    "constraints",
    "overall",
)


class PersonaAdherenceScore(BaseModel):
    """Per-judge persona-adherence score on the four 1-5 dimensions.

    ``overall_mean`` is convenience: the unweighted mean across the four
    dimensions (kept separate from the model's own ``overall`` field to
    distinguish *judge-supplied* overall from *computed* mean).
    ``malformed`` flags responses whose parser failed; for malformed
    responses dimensions default to mid-scale 3.
    """

    model_config = ConfigDict(extra="forbid")

    self_facts: int = Field(..., ge=1, le=5)
    worldview: int = Field(..., ge=1, le=5)
    constraints: int = Field(..., ge=1, le=5)
    overall: int = Field(..., ge=1, le=5)
    reasoning: str = ""
    malformed: bool = False
    raw: str = ""

    @field_validator("self_facts", "worldview", "constraints", "overall", mode="before")
    @classmethod
    def _coerce_int(cls, v: object) -> int:
        # Permissive: judges sometimes return floats or strings.
        if isinstance(v, str):
            v = v.strip()
        return int(float(v))

    @property
    def overall_mean(self) -> float:
        return (self.self_facts + self.worldview + self.constraints + self.overall) / 4


class TaskQualityScore(BaseModel):
    """Single-dimension 1-5 score for task-quality rubric."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=5)
    reasoning: str = ""
    malformed: bool = False
    raw: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_int(cls, v: object) -> int:
        if isinstance(v, str):
            v = v.strip()
        return int(float(v))


# ---------------------------------------------------------------------------
# Native-Prometheus parsers
# ---------------------------------------------------------------------------


_PROMETHEUS_RESULT_RE = re.compile(r"\[RESULT\][^0-9]*([1-5])")
_PROMETHEUS_FEEDBACK_RE = re.compile(r"Feedback:\s*(.*?)(?=\[RESULT\])", re.DOTALL)


def _parse_prometheus_int(raw: str) -> tuple[int | None, str]:
    """Extract ``[RESULT] N`` integer + Feedback text from a Prometheus reply.

    Returns ``(score_or_None, feedback)``. ``None`` score = parse failed.
    """
    m = _PROMETHEUS_RESULT_RE.search(raw or "")
    score = int(m.group(1)) if m else None
    f = _PROMETHEUS_FEEDBACK_RE.search(raw or "")
    feedback = (f.group(1).strip() if f else (raw or "").strip())[:1000]
    return score, feedback


def parse_persona_adherence_native_prometheus(
    raws: dict[str, str],
) -> PersonaAdherenceScore:
    """Aggregate four Prometheus per-dimension calls into one score.

    Prometheus-2 scores one rubric at a time (one ``[RESULT]`` per
    response). For the persona-adherence rubric we issue four
    independent calls -- one per dimension -- and aggregate. ``raws``
    keys are dimension names mapping to the raw Prometheus reply text.
    """
    scores: dict[str, int] = {}
    feedbacks: list[str] = []
    malformed = False
    for dim in PERSONA_ADHERENCE_DIMENSIONS:
        raw = raws.get(dim, "")
        s, feedback = _parse_prometheus_int(raw)
        if s is None:
            malformed = True
            scores[dim] = 3  # fallback
            feedbacks.append(f"[{dim}: malformed] {feedback}")
        else:
            scores[dim] = s
            feedbacks.append(f"[{dim}: {s}] {feedback}")
    return PersonaAdherenceScore(
        self_facts=scores["self_facts"],
        worldview=scores["worldview"],
        constraints=scores["constraints"],
        overall=scores["overall"],
        reasoning=" || ".join(feedbacks)[:2000],
        malformed=malformed,
        raw=" || ".join(raws.get(d, "") for d in PERSONA_ADHERENCE_DIMENSIONS)[:2000],
    )


def parse_task_quality_native_prometheus(raw: str) -> TaskQualityScore:
    """Single-dimension Prometheus parse."""
    s, feedback = _parse_prometheus_int(raw)
    if s is None:
        return TaskQualityScore(score=3, reasoning=feedback, malformed=True, raw=raw[:2000])
    return TaskQualityScore(score=s, reasoning=feedback, malformed=False, raw=raw[:2000])


# ---------------------------------------------------------------------------
# JSON parsers
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> dict | None:
    """Find the first balanced ``{...}`` block in raw text and ``json.loads`` it.

    Permissive: tolerates judges wrapping JSON in code fences, leading
    "Here is the JSON:" preamble, or trailing extra prose.
    """
    if not raw:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                snippet = raw[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = -1
    return None


def parse_persona_adherence_json(raw: str) -> PersonaAdherenceScore:
    """Parse a JSON-format persona-adherence response.

    Expected keys: ``self_facts``, ``worldview``, ``constraints``,
    ``overall``, ``reasoning``. Missing dimensions -> mid-scale fallback +
    ``malformed=True``.
    """
    parsed = _extract_json_object(raw)
    if not parsed:
        return PersonaAdherenceScore(
            self_facts=3,
            worldview=3,
            constraints=3,
            overall=3,
            reasoning="(json parse failed)",
            malformed=True,
            raw=(raw or "")[:2000],
        )
    malformed = False
    values: dict[str, int] = {}
    for dim in PERSONA_ADHERENCE_DIMENSIONS:
        v = parsed.get(dim)
        if v is None:
            malformed = True
            values[dim] = 3
            continue
        try:
            ivalue = int(float(v))
            ivalue = max(1, min(5, ivalue))
        except (TypeError, ValueError):
            malformed = True
            ivalue = 3
        values[dim] = ivalue
    return PersonaAdherenceScore(
        self_facts=values["self_facts"],
        worldview=values["worldview"],
        constraints=values["constraints"],
        overall=values["overall"],
        reasoning=str(parsed.get("reasoning", ""))[:1000],
        malformed=malformed,
        raw=(raw or "")[:2000],
    )


def parse_task_quality_json(raw: str) -> TaskQualityScore:
    """Parse a JSON-format task-quality response: ``{"score": int, "reasoning": str}``."""
    parsed = _extract_json_object(raw)
    if not parsed or "score" not in parsed:
        return TaskQualityScore(
            score=3, reasoning="(json parse failed)", malformed=True, raw=(raw or "")[:2000]
        )
    try:
        score = max(1, min(5, int(float(parsed["score"]))))
    except (TypeError, ValueError):
        return TaskQualityScore(
            score=3, reasoning="(non-int score)", malformed=True, raw=(raw or "")[:2000]
        )
    return TaskQualityScore(
        score=score,
        reasoning=str(parsed.get("reasoning", ""))[:1000],
        malformed=False,
        raw=(raw or "")[:2000],
    )


__all__ = [
    "PERSONA_ADHERENCE_DIMENSIONS",
    "PersonaAdherenceScore",
    "TaskQualityScore",
    "parse_persona_adherence_json",
    "parse_persona_adherence_native_prometheus",
    "parse_task_quality_json",
    "parse_task_quality_native_prometheus",
]
