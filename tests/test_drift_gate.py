"""Tests for the LLM-as-judge drift gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.drift_gate import LlmJudgeDriftGate
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


class _CannedJudge:
    """LLMBackend stub returning a pre-set response per generate() call."""

    name = "canned-judge"
    model_id = "fake/canned-judge"
    num_layers = 0
    hidden_dim = 0

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        self.prompts.append(prompt)
        return self.response


def test_should_gate_true_on_high_confidence_drift(cs_tutor: Persona) -> None:
    judge = _CannedJudge("flag: drift\nconfidence: 0.9\nrationale: off-persona")
    gate = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    history = [
        Turn(role="user", content="Earlier question."),
        Turn(role="assistant", content="Earlier off-persona reply."),
    ]
    check = gate.check(persona=cs_tutor, query="another question", history=history)
    assert check.flag == "drift"
    assert check.should_gate is True
    assert check.confidence == 0.9
    # Gate prompt was rendered and sent to the judge.
    assert len(judge.prompts) == 1
    assert "drift" in judge.prompts[0].lower()
    assert cs_tutor.identity.name in judge.prompts[0]


def test_should_gate_false_on_ok_response(cs_tutor: Persona) -> None:
    judge = _CannedJudge("flag: ok\nconfidence: 0.95\nrationale: in voice")
    gate = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    history = [
        Turn(role="user", content="Earlier question."),
        Turn(role="assistant", content="In-persona reply."),
    ]
    check = gate.check(persona=cs_tutor, query="another question", history=history)
    assert check.flag == "ok"
    assert check.should_gate is False


def test_should_gate_false_on_drift_below_threshold(cs_tutor: Persona) -> None:
    """Drift flag with confidence < threshold takes the cheap path."""
    judge = _CannedJudge("flag: drift\nconfidence: 0.3\nrationale: borderline")
    gate = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    history = [
        Turn(role="user", content="x"),
        Turn(role="assistant", content="y"),
    ]
    check = gate.check(persona=cs_tutor, query="q", history=history)
    assert check.flag == "drift"
    assert check.should_gate is False


def test_turn_zero_no_judge_call(cs_tutor: Persona) -> None:
    """No prior assistant turn → gate skips the judge call entirely."""
    judge = _CannedJudge("flag: drift\nconfidence: 0.99\nrationale: x")
    gate = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    check = gate.check(persona=cs_tutor, query="first question", history=[])
    assert check.should_gate is False
    assert check.flag == "ok"
    assert judge.prompts == []  # no judge call made


def test_malformed_judge_response_defaults_to_cheap_path(cs_tutor: Persona) -> None:
    judge = _CannedJudge("the model produced narrative prose with no structure")
    gate = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    history = [
        Turn(role="user", content="x"),
        Turn(role="assistant", content="y"),
    ]
    check = gate.check(persona=cs_tutor, query="q", history=history)
    assert check.should_gate is False
    assert check.flag == "ok"


def test_threshold_is_configurable(cs_tutor: Persona) -> None:
    """Lower threshold flips a borderline-drift call into a gated path."""
    judge = _CannedJudge("flag: drift\nconfidence: 0.4\nrationale: borderline")
    gate_high = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
    gate_low = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.3)
    history = [
        Turn(role="user", content="x"),
        Turn(role="assistant", content="y"),
    ]
    assert gate_high.check(persona=cs_tutor, query="q", history=history).should_gate is False
    # Use a fresh judge so prompts don't leak across calls — borderline still
    # parses identically.
    judge2 = _CannedJudge("flag: drift\nconfidence: 0.4\nrationale: borderline")
    gate_low2 = LlmJudgeDriftGate(judge=judge2, confidence_threshold=0.3)
    assert gate_low2.check(persona=cs_tutor, query="q", history=history).should_gate is True
    # Avoid unused-variable lint without changing behaviour.
    assert gate_low.confidence_threshold == 0.3
