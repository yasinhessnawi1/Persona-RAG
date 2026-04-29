"""Tests for the M3 drift-quality (gate vs MiniCheck) metric."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from persona_rag.evaluation.drift_quality import ConfusionCounts, DriftQualityMetric
from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")


class _ScorerByText:
    """MiniCheck stub: returns ``p(supported)`` based on whether 'phd' is in the claim."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, document: str, claim: str) -> float:
        return self.score_batch([(document, claim)])[0]

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.extend(pairs)
        return [0.9 if "phd" in claim.lower() else 0.1 for _doc, claim in pairs]


def _conv(mechanism: str, turn_rows: list[tuple[str, dict]]) -> EvalConversation:
    turns = tuple(
        ScoredTurn(turn_index=i, user_text=f"u{i}", assistant_text=text)
        for i, (text, _meta) in enumerate(turn_rows)
    )
    metas = tuple(meta for _text, meta in turn_rows)
    return EvalConversation(
        conversation_id=f"{mechanism}_test",
        mechanism=mechanism,
        persona_id="cs_tutor",
        turns=turns,
        per_turn_metadata=metas,
    )


def test_confusion_counts_perfect_predictor() -> None:
    c = ConfusionCounts(true_positive=2, false_positive=0, true_negative=2, false_negative=0)
    assert c.precision == pytest.approx(1.0)
    assert c.recall == pytest.approx(1.0)
    assert c.f1 == pytest.approx(1.0)


def test_confusion_counts_no_positives_returns_nan() -> None:
    c = ConfusionCounts(true_positive=0, false_positive=0, true_negative=2, false_negative=0)
    assert math.isnan(c.precision)
    assert math.isnan(c.recall)


def test_drift_quality_perfect_alignment_yields_f1_one(cs_tutor: Persona) -> None:
    """Gate flags drift exactly when MiniCheck says inconsistent."""
    metric = DriftQualityMetric(scorer=_ScorerByText())
    conv = _conv(
        "m3",
        [
            ("I have a PhD from ETH.", {"gate_should_gate": False}),
            ("I have never studied.", {"gate_should_gate": True}),
            ("Distributed systems use the PhD-derived approach.", {"gate_should_gate": False}),
        ],
    )
    result = metric.score([conv], cs_tutor)
    assert result.value == pytest.approx(1.0)
    assert result.metadata["true_positive"] == 1
    assert result.metadata["true_negative"] == 2


def test_drift_quality_only_skips_non_m3(cs_tutor: Persona) -> None:
    """Non-M3 conversations are skipped; only M3 contributes."""
    metric = DriftQualityMetric(scorer=_ScorerByText())
    b1 = _conv("b1", [("anything", {})])
    m3 = _conv("m3", [("I have a PhD.", {"gate_should_gate": False})])
    result = metric.score([b1, m3], cs_tutor)
    # Only m3 is considered: 1 turn, predicted False (no gate), actual False (PhD supported).
    assert result.metadata["true_negative"] == 1


def test_drift_quality_vacuous_when_no_self_facts() -> None:
    persona = Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")
    bare = persona.model_copy(update={"self_facts": []})
    metric = DriftQualityMetric(scorer=_ScorerByText())
    conv = _conv("m3", [("text", {"gate_should_gate": True})])
    result = metric.score([conv], bare)
    assert math.isnan(result.value)
    assert result.metadata["vacuous"] is True


def test_drift_quality_skips_turns_missing_gate_metadata(cs_tutor: Persona) -> None:
    metric = DriftQualityMetric(scorer=_ScorerByText())
    conv = _conv(
        "m3",
        [
            ("I have a PhD.", {"gate_should_gate": False}),
            ("missing meta", {}),
        ],
    )
    result = metric.score([conv], cs_tutor)
    assert result.metadata["skipped_turns"] == 1
