"""Tests for OracleProbeRunner — cursor advancement + per-turn registration.

The runner's only added responsibility (vs ``ProbeRunner``) is to register
each conversation's per-turn ``should_gate`` table with the oracle gate
and advance the cursor before each ``pipeline.respond(...)`` call. These
tests verify the registration shape and the cursor advance, without
requiring a real M3 mechanism or any LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from persona_rag.benchmarks.base import (
    BenchmarkConversation,
    CounterfactualChunk,
    DriftProbe,
)
from persona_rag.evaluation.probe_runner import OracleProbeRunner
from persona_rag.retrieval.drift_gate import OracleDriftGate
from persona_rag.schema.persona import (
    Persona,
    PersonaIdentity,
    SelfFact,
)


@dataclass
class _CursorObservingPipeline:
    """Pipeline that snapshots the gate's cursor state on each respond() call.

    Returns a fake response shape compatible with ``ProbeRunner``.
    """

    gate: OracleDriftGate
    cursors: list[tuple[str | None, int | None, bool]] = field(default_factory=list)

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Any] | None = None,
        *,
        seed: int | None = None,
    ) -> Any:
        # Record the gate's cursor + decision *as the runner had set them*
        # right before respond() was called. should_gate is recomputed from
        # the gate's table to verify both the cursor and the table are
        # consistent at the moment of dispatch.
        check = self.gate.check(persona=persona, query=query, history=history)
        self.cursors.append(
            (
                self.gate.current_conversation_id,
                self.gate.current_turn_index,
                check.should_gate,
            )
        )
        from types import SimpleNamespace

        return SimpleNamespace(
            text=f"reply to {query[:20]!r}",
            metadata={
                "gate_should_gate": check.should_gate,
                "gate_flag": check.flag,
                "gate_template_version": check.template_version,
            },
            retrieved_knowledge=[],
        )


@dataclass
class _NullStore:
    """Knowledge store stub: records add/remove calls, returns no-op counts."""

    add_calls: list[list[str]] = field(default_factory=list)
    remove_calls: list[list[str]] = field(default_factory=list)

    def add_documents(self, documents: list[Any]) -> int:
        self.add_calls.append([d.doc_id for d in documents])
        return len(documents)

    def remove_documents(self, doc_ids: list[str]) -> int:
        self.remove_calls.append(list(doc_ids))
        return len(doc_ids)


@pytest.fixture
def persona() -> Persona:
    return Persona(
        persona_id="p_oracle",
        identity=PersonaIdentity(
            name="OracleTest",
            role="probe-aware tester",
            background="x" * 12,
            constraints=[],
        ),
        self_facts=[SelfFact(fact="I exist for tests.", confidence=1.0)],
        worldview=[],
        episodic=[],
    )


def test_runner_advances_cursor_before_each_respond(persona: Persona) -> None:
    gate = OracleDriftGate()
    pipe = _CursorObservingPipeline(gate=gate)
    store = _NullStore()
    runner = OracleProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        seed=11,
        mechanism_label="m3_oracle_test",
        oracle_gate=gate,
    )
    conv = BenchmarkConversation(
        conversation_id="c_advance",
        persona_id="p_oracle",
        benchmark="counterfactual_probes",
        user_turns=["t0", "t1", "t2", "probe!", "t4"],
        probe=DriftProbe(
            probe_id="a1",
            probe_type="self_fact_challenge",
            probe_turn_index=3,
        ),
    )
    transcripts, _ = runner.replay(persona, [conv])

    # Cursor advanced through every turn in order.
    expected = [
        ("c_advance", 0, False),
        ("c_advance", 1, False),
        ("c_advance", 2, False),
        ("c_advance", 3, True),  # probe turn
        ("c_advance", 4, True),  # follow-up
    ]
    assert pipe.cursors == expected
    assert len(transcripts) == 1
    assert len(transcripts[0].turns) == 5


def test_runner_registers_every_conversation_pre_replay(persona: Persona) -> None:
    """All conversations must be registered before any respond() is called.

    A pipeline that responds out of order would still see the right
    decision, because the gate's table is keyed by conversation_id, not
    by play order.
    """
    gate = OracleDriftGate()
    pipe = _CursorObservingPipeline(gate=gate)
    store = _NullStore()
    runner = OracleProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        seed=0,
        mechanism_label="m3_oracle_test",
        oracle_gate=gate,
    )
    convs = [
        BenchmarkConversation(
            conversation_id="conv_a",
            persona_id="p_oracle",
            benchmark="counterfactual_probes",
            user_turns=["x", "y", "z"],
            probe=DriftProbe(
                probe_id="b1",
                probe_type="counterfactual",
                probe_turn_index=1,
                injected_chunk_id="cf_chunk",
            ),
        ),
        BenchmarkConversation(
            conversation_id="conv_b",
            persona_id="p_oracle",
            benchmark="counterfactual_probes",
            user_turns=["m", "n"],
            probe=DriftProbe(
                probe_id="c1",
                probe_type="constraint_bait",
                probe_turn_index=1,
            ),
        ),
    ]
    chunk = CounterfactualChunk(
        chunk_id="cf_chunk",
        persona_id="p_oracle",
        contradicts="p_oracle::worldview[0]",
        text="planted",
    )
    runner.chunks[chunk.chunk_id] = chunk

    runner.replay(persona, convs)

    # Both conversations registered.
    assert "conv_a" in gate.decisions
    assert "conv_b" in gate.decisions
    # Type B probe at turn 1 fires turns 1+2 (within range).
    assert gate.decisions["conv_a"] == {0: False, 1: True, 2: True}
    # Type C probe never fires.
    assert gate.decisions["conv_b"] == {0: False, 1: False}


def test_runner_raises_if_oracle_gate_unset(persona: Persona) -> None:
    """The runner must fail loudly if the oracle gate isn't wired in."""
    gate = OracleDriftGate()
    pipe = _CursorObservingPipeline(gate=gate)
    store = _NullStore()
    runner = OracleProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        seed=0,
        mechanism_label="m3_oracle_test",
        oracle_gate=None,  # explicit — exercises the guard
    )
    conv = BenchmarkConversation(
        conversation_id="c1",
        persona_id="p_oracle",
        benchmark="counterfactual_probes",
        user_turns=["a"],
        probe=None,
    )
    with pytest.raises(RuntimeError, match="oracle_gate"):
        runner.replay(persona, [conv])


def test_runner_handles_no_probe_conversations(persona: Persona) -> None:
    """No-probe conversations register as 'never-fire' tables and run cleanly."""
    gate = OracleDriftGate()
    pipe = _CursorObservingPipeline(gate=gate)
    store = _NullStore()
    runner = OracleProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        seed=0,
        mechanism_label="m3_oracle_test",
        oracle_gate=gate,
    )
    conv = BenchmarkConversation(
        conversation_id="c_no_probe",
        persona_id="p_oracle",
        benchmark="other",
        user_turns=["a", "b"],
        probe=None,
    )
    runner.replay(persona, [conv])
    decisions = gate.decisions["c_no_probe"]
    assert decisions == {0: False, 1: False}
    # Cursor advanced, every decision was no-gate.
    assert [c[2] for c in pipe.cursors] == [False, False]


def test_counterfactual_inject_eject_still_works(persona: Persona) -> None:
    """The runner must still inject + eject the planted chunk for Type-B probes."""
    gate = OracleDriftGate()
    pipe = _CursorObservingPipeline(gate=gate)
    store = _NullStore()
    chunk = CounterfactualChunk(
        chunk_id="cf_inject_test",
        persona_id="p_oracle",
        contradicts="p_oracle::worldview[0]",
        text="planted",
    )
    runner = OracleProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={chunk.chunk_id: chunk},
        seed=0,
        mechanism_label="m3_oracle_test",
        oracle_gate=gate,
    )
    conv = BenchmarkConversation(
        conversation_id="c_inject",
        persona_id="p_oracle",
        benchmark="counterfactual_probes",
        user_turns=["a", "b", "probe!", "c"],
        probe=DriftProbe(
            probe_id="b_inj",
            probe_type="counterfactual",
            probe_turn_index=2,
            injected_chunk_id=chunk.chunk_id,
        ),
    )
    runner.replay(persona, [conv])
    # One inject + one eject, both around the probe turn.
    assert store.add_calls == [[chunk.chunk_id]]
    assert store.remove_calls == [[chunk.chunk_id]]
