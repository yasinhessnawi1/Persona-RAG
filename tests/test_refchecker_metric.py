"""Tests for the soft-optional RefChecker metric."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.refchecker_metric import (
    RefCheckerMetric,
    is_refchecker_available,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")


def test_refchecker_returns_nan_when_not_available(cs_tutor: Persona) -> None:
    """If refchecker is not installed, score returns NaN with available=False."""
    metric = RefCheckerMetric()
    conv = EvalConversation(
        conversation_id="x",
        mechanism="b1",
        persona_id="cs_tutor",
        turns=(ScoredTurn(turn_index=0, user_text="hi", assistant_text="hello"),),
    )
    result = metric.score([conv], cs_tutor)
    if not is_refchecker_available():
        assert math.isnan(result.value)
        assert result.metadata["available"] is False
    else:
        # If RefChecker is installed locally, just verify the metric returns the
        # expected shape without asserting a specific value.
        assert "available" in result.metadata
