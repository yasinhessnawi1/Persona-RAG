"""Tests for the per-mechanism cost tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.evaluation.cost import CostTracker
from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")


def _conv(mechanism: str, per_turn_meta: list[dict]) -> EvalConversation:
    turns = tuple(
        ScoredTurn(turn_index=i, user_text=f"u{i}", assistant_text=f"a{i}")
        for i in range(len(per_turn_meta))
    )
    return EvalConversation(
        conversation_id=f"{mechanism}_test",
        mechanism=mechanism,
        persona_id="cs_tutor",
        turns=turns,
        per_turn_metadata=tuple(per_turn_meta),
    )


def test_cost_b1_one_call_per_turn(cs_tutor: Persona) -> None:
    """B1 always costs 1 LLM call per turn."""
    tracker = CostTracker(mechanism="b1")
    conv = _conv("b1", [{}, {}, {}])
    result = tracker.score([conv], cs_tutor)
    assert result.value == pytest.approx(1.0)
    assert result.metadata["total_llm_calls"] == 3
    assert result.metadata["total_judge_calls"] == 0


def test_cost_m3_cheap_path_two_calls(cs_tutor: Persona) -> None:
    """M3 cheap path = gate (1) + responder (1) = 2."""
    tracker = CostTracker(mechanism="m3")
    conv = _conv(
        "m3",
        [
            {"gate_should_gate": False},
            {"gate_should_gate": False},
        ],
    )
    result = tracker.score([conv], cs_tutor)
    assert result.value == pytest.approx(2.0)
    assert result.metadata["total_judge_calls"] == 2  # one gate per turn
    assert result.metadata["gate_trigger_rate"] == pytest.approx(0.0)


def test_cost_m3_gated_path_includes_candidates_and_ranker(cs_tutor: Persona) -> None:
    """M3 gated path = gate(1) + n_candidates + ranker_judge_calls."""
    tracker = CostTracker(mechanism="m3")
    conv = _conv(
        "m3",
        [
            {"gate_should_gate": True, "candidates_n": 3, "ranker_judge_calls": 2},
            {"gate_should_gate": False},
        ],
    )
    result = tracker.score([conv], cs_tutor)
    # turn0: 1 + 3 + 2 = 6; turn1: 2; mean = 4.
    assert result.value == pytest.approx(4.0)
    assert result.metadata["gated_turns"] == 1
    assert result.metadata["gate_trigger_rate"] == pytest.approx(0.5)
    # Judge calls: turn0: gate(1) + ranker(2) = 3; turn1: gate(1) = 1; total = 4.
    assert result.metadata["total_judge_calls"] == 4


def test_cost_records_latency_when_present(cs_tutor: Persona) -> None:
    tracker = CostTracker(mechanism="b1")
    conv = _conv(
        "b1",
        [
            {"latency_seconds": 1.5},
            {"latency_seconds": 2.5},
        ],
    )
    result = tracker.score([conv], cs_tutor)
    assert result.metadata["mean_latency_seconds"] == pytest.approx(2.0)


def test_cost_handles_nested_metadata_dict(cs_tutor: Persona) -> None:
    """run_baseline.py writes per-turn metadata under a 'metadata' sub-dict -- flatten it."""
    tracker = CostTracker(mechanism="m3")
    conv = _conv(
        "m3",
        [
            {
                "metadata": {
                    "gate_should_gate": True,
                    "candidates_n": 3,
                    "ranker_judge_calls": 2,
                }
            },
        ],
    )
    result = tracker.score([conv], cs_tutor)
    assert result.value == pytest.approx(6.0)
