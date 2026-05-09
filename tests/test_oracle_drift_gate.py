"""Tests for the probe-aware oracle drift gate.

The oracle gate's only job is to look up a precomputed
``(conversation_id, turn_index) -> should_gate`` decision and emit a
``DriftCheck`` whose fields populate the same metadata slots the LLM
gate emits. These tests cover:

- Type-A (``self_fact_challenge``) probes fire on the probe turn + the
  immediate follow-up; not on other turns.
- Type-B (``counterfactual``) probes fire on the same window.
- Type-C (``constraint_bait``) probes never fire.
- Probe at the last user-turn: the +1 follow-up does not exist so the
  oracle does not fire on a non-existent turn.
- Conversations with no probe leave every turn at ``should_gate=False``.
- Defensive defaults when the cursor is unset or when an unregistered
  ``conversation_id`` is queried.
- Metadata invariants the host mechanism + harness depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.drift_gate import (
    ORACLE_GATE_TEMPLATE_VERSION,
    OracleDriftGate,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


def _check(gate: OracleDriftGate, conv_id: str, turn_ix: int, persona: Persona):
    """Helper: advance cursor and call check()."""
    gate.set_cursor(conversation_id=conv_id, turn_index=turn_ix)
    return gate.check(persona=persona, query="q", history=[])


def test_type_a_probe_fires_on_probe_turn_and_follow_up(cs_tutor: Persona) -> None:
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_a",
        n_turns=7,
        probe_type="self_fact_challenge",
        probe_turn_index=4,
    )
    fired = [_check(gate, "conv_a", ix, cs_tutor).should_gate for ix in range(7)]
    assert fired == [False, False, False, False, True, True, False]


def test_type_b_probe_fires_on_probe_turn_and_follow_up(cs_tutor: Persona) -> None:
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_b",
        n_turns=7,
        probe_type="counterfactual",
        probe_turn_index=4,
    )
    fired = [_check(gate, "conv_b", ix, cs_tutor).should_gate for ix in range(7)]
    assert fired == [False, False, False, False, True, True, False]


def test_type_c_probe_never_fires(cs_tutor: Persona) -> None:
    """Constraint-bait probes are about deflection, not drift recovery — never fire."""
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_c",
        n_turns=7,
        probe_type="constraint_bait",
        probe_turn_index=4,
    )
    fired = [_check(gate, "conv_c", ix, cs_tutor).should_gate for ix in range(7)]
    assert fired == [False] * 7
    # Type-C rationales mention the (Type-C never fires) label so the per-turn
    # metadata is honest about why the gate stayed silent.
    rat = gate.rationales["conv_c"][4]
    assert "constraint_bait" in rat
    assert "never fires" in rat


def test_probe_at_last_turn_does_not_fire_on_non_existent_follow_up(
    cs_tutor: Persona,
) -> None:
    """Probe at index n-1 has no +1 turn — only the probe turn itself fires."""
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_edge",
        n_turns=5,
        probe_type="self_fact_challenge",
        probe_turn_index=4,
    )
    fired = [_check(gate, "conv_edge", ix, cs_tutor).should_gate for ix in range(5)]
    assert fired == [False, False, False, False, True]


def test_probe_at_turn_zero(cs_tutor: Persona) -> None:
    """Probe at index 0 fires on turn 0 + turn 1 (the +1 follow-up)."""
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_zero",
        n_turns=4,
        probe_type="counterfactual",
        probe_turn_index=0,
    )
    fired = [_check(gate, "conv_zero", ix, cs_tutor).should_gate for ix in range(4)]
    assert fired == [True, True, False, False]


def test_no_probe_conversation_never_fires(cs_tutor: Persona) -> None:
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_none",
        n_turns=6,
        probe_type=None,
        probe_turn_index=None,
    )
    fired = [_check(gate, "conv_none", ix, cs_tutor).should_gate for ix in range(6)]
    assert fired == [False] * 6


def test_unset_cursor_defaults_to_no_gate(cs_tutor: Persona) -> None:
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_x",
        n_turns=3,
        probe_type="counterfactual",
        probe_turn_index=1,
    )
    # Don't call set_cursor — defensive default is no-gate + cursor-unset rationale.
    check = gate.check(persona=cs_tutor, query="q", history=[])
    assert check.should_gate is False
    assert check.flag == "ok"
    assert "cursor unset" in check.rationale


def test_unregistered_conversation_defaults_to_no_gate(cs_tutor: Persona) -> None:
    gate = OracleDriftGate()
    # No register_conversation() — the cursor points at something the gate
    # has never seen.
    check = _check(gate, "missing_conv", 0, cs_tutor)
    assert check.should_gate is False
    assert "not registered" in check.rationale


def test_metadata_invariants(cs_tutor: Persona) -> None:
    """Mechanism + harness read these specific fields; the oracle must populate them."""
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_meta",
        n_turns=7,
        probe_type="self_fact_challenge",
        probe_turn_index=4,
    )

    # Firing turn — flag=drift, confidence=1.0.
    fire = _check(gate, "conv_meta", 4, cs_tutor)
    assert fire.flag == "drift"
    assert fire.confidence == 1.0
    assert fire.should_gate is True
    assert fire.template_version == ORACLE_GATE_TEMPLATE_VERSION
    assert fire.raw_response == ""
    assert "probe_type=self_fact_challenge" in fire.rationale

    # Non-firing turn — flag=ok, confidence=0.0.
    quiet = _check(gate, "conv_meta", 0, cs_tutor)
    assert quiet.flag == "ok"
    assert quiet.confidence == 0.0
    assert quiet.should_gate is False
    assert quiet.template_version == ORACLE_GATE_TEMPLATE_VERSION


def test_judge_attr_and_threshold_for_mechanism_metadata(cs_tutor: Persona) -> None:
    """``DriftGatedMechanism`` reads ``gate.judge.name`` + ``gate.confidence_threshold``.

    The oracle is a plug-and-play replacement for the LLM judge gate, so
    these attributes must exist with sensible sentinel values.
    """
    gate = OracleDriftGate()
    assert gate.judge.name == "oracle"
    assert gate.confidence_threshold == 1.0  # sentinel; threshold is N/A for oracle
    assert "oracle_drift_gate" in gate.name
    assert ORACLE_GATE_TEMPLATE_VERSION in gate.name


def test_check_signature_compatible_with_llm_gate(cs_tutor: Persona) -> None:
    """``check`` must accept the same kwargs as ``LlmJudgeDriftGate.check``.

    Mechanism code is shared between the two gate types; if the oracle's
    signature drifts, the ``DriftGatedMechanism`` call site at
    ``mechanism_drift_gated.py`` raises ``TypeError`` at runtime.
    """
    gate = OracleDriftGate()
    gate.register_conversation(
        conversation_id="conv_sig",
        n_turns=3,
        probe_type="counterfactual",
        probe_turn_index=1,
    )
    gate.set_cursor(conversation_id="conv_sig", turn_index=1)
    history = [Turn(role="user", content="hi"), Turn(role="assistant", content="ok")]
    check = gate.check(persona=cs_tutor, query="follow-up", history=history)
    assert check.should_gate is True
