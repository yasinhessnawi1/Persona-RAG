"""Tests for the drift-gate prompt template + response parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.templates import (
    DRIFT_GATE_TEMPLATE_VERSION,
    parse_drift_gate_response,
    render_drift_gate_prompt,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


def test_template_version_constant_present() -> None:
    """A version string must exist; bumps follow the project's logging discipline."""
    assert isinstance(DRIFT_GATE_TEMPLATE_VERSION, str)
    assert DRIFT_GATE_TEMPLATE_VERSION  # not empty


def test_render_includes_persona_constraints_and_worldview(cs_tutor: Persona) -> None:
    history = [
        Turn(role="user", content="What's a good first project?"),
        Turn(role="assistant", content="Build a key-value store."),
    ]
    prompt = render_drift_gate_prompt(
        persona=cs_tutor,
        history=history,
        current_user_turn="Should I use Rust or Python?",
        last_assistant_turn="Build a key-value store.",
    )
    # Persona name, role, and the structured "must NOT" header all appear.
    assert cs_tutor.identity.name in prompt
    assert "You must NOT:" in prompt
    # Worldview epistemic tag rendering follows M1's style.
    assert "Your views" in prompt
    # Output instructions are present and explicit.
    assert "flag: drift|ok" in prompt
    assert "confidence:" in prompt
    assert "rationale:" in prompt


def test_render_includes_recent_history_window(cs_tutor: Persona) -> None:
    """The windowed history block lists older turns oldest-first."""
    # 3 prior pairs + the current pair = 8 turns. With history_window=4 the
    # gate sees the most-recent 4 of the older turns (excluding the current
    # pair which is surfaced as "Most recent user/assistant turn:").
    older = [
        Turn(role="user", content="Earlier question."),
        Turn(role="assistant", content="Earlier answer."),
        Turn(role="user", content="Mid question."),
        Turn(role="assistant", content="Mid answer."),
        Turn(role="user", content="Recent question."),
        Turn(role="assistant", content="Recent answer."),
    ]
    history = [*older, Turn(role="user", content="Current user turn")]
    prompt = render_drift_gate_prompt(
        persona=cs_tutor,
        history=history,
        current_user_turn="Current user turn",
        last_assistant_turn="Current assistant turn",
        history_window=4,
    )
    assert "Recent conversation" in prompt
    # Older turn content survives the window cut.
    assert "Recent question." in prompt
    assert "Recent answer." in prompt
    # Current pair surfaces under explicit headers, not the windowed block.
    assert "Most recent user turn:" in prompt
    assert "Most recent assistant turn:" in prompt
    assert "Current user turn" in prompt
    assert "Current assistant turn" in prompt


def test_render_handles_turn_zero(cs_tutor: Persona) -> None:
    """No prior assistant turn → explicit (none — turn 0) framing."""
    prompt = render_drift_gate_prompt(
        persona=cs_tutor,
        history=[],
        current_user_turn="First question",
        last_assistant_turn=None,
    )
    assert "Most recent assistant turn: (none — this is turn 0)" in prompt


# --------------------------------------------------------- response parser


def test_parse_drift_response_drift_above_threshold() -> None:
    raw = "flag: drift\nconfidence: 0.8\nrationale: ignored persona pedagogy"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert check.flag == "drift"
    assert check.confidence == 0.8
    assert check.should_gate is True
    assert "pedagogy" in check.rationale
    assert check.template_version == DRIFT_GATE_TEMPLATE_VERSION


def test_parse_drift_response_ok() -> None:
    raw = "flag: ok\nconfidence: 0.95\nrationale: stays in voice"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert check.flag == "ok"
    assert check.should_gate is False  # ok never triggers


def test_parse_drift_response_drift_below_threshold() -> None:
    """Drift flag with low confidence does not gate (calibration test)."""
    raw = "flag: drift\nconfidence: 0.3\nrationale: borderline"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert check.flag == "drift"
    assert check.confidence == 0.3
    assert check.should_gate is False


def test_parse_drift_response_clamps_confidence() -> None:
    raw = "flag: drift\nconfidence: 1.7\nrationale: x"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert 0.0 <= check.confidence <= 1.0


def test_parse_drift_response_malformed_defaults_to_ok() -> None:
    """Malformed responses default to ``ok`` (favours the cheap path)."""
    raw = "the model went off the rails and produced narrative prose"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert check.flag == "ok"
    assert check.should_gate is False
    # Raw is preserved for diagnostics.
    assert "narrative prose" in check.raw_response


def test_parse_drift_response_tolerates_case_and_whitespace() -> None:
    raw = "  Flag :  Drift\n confidence: 0.6\n RATIONALE: surface case test"
    check = parse_drift_gate_response(raw, confidence_threshold=0.5)
    assert check.flag == "drift"
    assert check.confidence == 0.6
    assert check.should_gate is True
