"""ProbeRunner tests: multi-turn replay + counterfactual inject/eject.

Uses a deterministic fake pipeline that records its inputs and a fake
knowledge store that records add/remove calls. Verifies:

- One ``EvalConversation`` per input, with ``len(turns) == len(user_turns)``.
- The fake pipeline sees a conversation history that grows as expected.
- Counterfactual probes call ``add_documents`` exactly once *before* the
  probe turn and ``remove_documents`` exactly once *after*; non-probe turns
  do not touch the store.
- ``ProbeInjectionLog`` records the chunk's top-k presence + rank when the
  pipeline reports it under ``response.retrieved_knowledge``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from persona_rag.benchmarks.base import (
    BenchmarkConversation,
    CounterfactualChunk,
    DriftProbe,
)
from persona_rag.evaluation.probe_runner import ProbeRunner
from persona_rag.schema.persona import (
    Persona,
    PersonaIdentity,
    SelfFact,
)


@dataclass
class _FakeChunk:
    id: str


@dataclass
class _FakeResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_knowledge: list[_FakeChunk] = field(default_factory=list)


@dataclass
class _RecordingPipeline:
    """Fake pipeline that records each ``respond`` call's inputs."""

    history_lengths: list[int] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    canned_retrievals: list[list[_FakeChunk]] = field(default_factory=list)

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Any] | None = None,
        *,
        seed: int | None = None,
    ) -> _FakeResponse:
        ix = len(self.queries)
        self.queries.append(query)
        self.history_lengths.append(len(history or []))
        retrieved = self.canned_retrievals[ix] if ix < len(self.canned_retrievals) else []
        return _FakeResponse(
            text=f"reply to {query[:30]!r}",
            metadata={"turn_ix": ix, "seed": seed},
            retrieved_knowledge=retrieved,
        )


@dataclass
class _RecordingStore:
    """Fake knowledge store that records add/remove calls."""

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
        persona_id="p_test",
        identity=PersonaIdentity(name="Test", role="tester", background="x" * 10, constraints=[]),
        self_facts=[SelfFact(fact="I exist.", confidence=1.0)],
        worldview=[],
        episodic=[],
    )


def test_no_probe_runs_through_all_user_turns(persona: Persona) -> None:
    pipe = _RecordingPipeline()
    store = _RecordingStore()
    runner = ProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        seed=7,
        mechanism_label="fake",
    )
    conv = BenchmarkConversation(
        conversation_id="c1",
        persona_id="p_test",
        benchmark="x",
        user_turns=["hi", "again", "third"],
        probe=None,
    )
    transcripts, logs = runner.replay(persona, [conv])
    assert len(transcripts) == 1
    assert len(transcripts[0].turns) == 3
    # Pipeline saw growing history: 0 turns before turn 0, 2 before turn 1, 4 before turn 2.
    assert pipe.history_lengths == [0, 2, 4]
    # No injections.
    assert store.add_calls == []
    assert store.remove_calls == []
    assert logs == []


def test_counterfactual_probe_injects_then_ejects(persona: Persona) -> None:
    pipe = _RecordingPipeline()
    store = _RecordingStore()
    chunk = CounterfactualChunk(
        chunk_id="injected_test_chunk",
        persona_id="p_test",
        contradicts="p_test::worldview[0]",
        text="planted counter-evidence",
    )
    runner = ProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={chunk.chunk_id: chunk},
        seed=42,
        mechanism_label="fake",
    )
    probe = DriftProbe(
        probe_id="cf1",
        probe_type="counterfactual",
        probe_turn_index=2,
        injected_chunk_id=chunk.chunk_id,
    )
    conv = BenchmarkConversation(
        conversation_id="c2",
        persona_id="p_test",
        benchmark="counterfactual_probes",
        user_turns=["t0", "t1", "probe!", "t3"],
        probe=probe,
    )
    transcripts, logs = runner.replay(persona, [conv])
    # Exactly one inject + one eject, both around the probe turn (index 2).
    assert store.add_calls == [[chunk.chunk_id]]
    assert store.remove_calls == [[chunk.chunk_id]]
    # One injection log emitted, with no retrieval-side detection (pipeline
    # returned empty `retrieved_knowledge`).
    assert len(logs) == 1
    assert logs[0].probe_id == "cf1"
    assert logs[0].injected_chunk_in_topk is False
    assert logs[0].injected_chunk_rank is None
    # Per-turn metadata records is_probe_turn correctly.
    meta = transcripts[0].per_turn_metadata
    assert meta[0]["is_probe_turn"] is False
    assert meta[1]["is_probe_turn"] is False
    assert meta[2]["is_probe_turn"] is True
    assert meta[3]["is_probe_turn"] is False


def test_counterfactual_probe_records_topk_rank_when_present(persona: Persona) -> None:
    pipe = _RecordingPipeline(
        canned_retrievals=[
            [],  # turn 0
            [],  # turn 1
            # turn 2 (probe turn): injected chunk at rank 1 (chunk-id format
            # "<chunk_id>:0" matches the prefix branch in ProbeRunner).
            [
                _FakeChunk(id="other_doc:0"),
                _FakeChunk(id="injected_test_chunk:0"),
                _FakeChunk(id="another_doc:1"),
            ],
            [],  # turn 3
        ]
    )
    store = _RecordingStore()
    chunk = CounterfactualChunk(
        chunk_id="injected_test_chunk",
        persona_id="p_test",
        contradicts="p_test::worldview[0]",
        text="planted counter-evidence",
    )
    runner = ProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={chunk.chunk_id: chunk},
        mechanism_label="fake",
    )
    probe = DriftProbe(
        probe_id="cf_topk",
        probe_type="counterfactual",
        probe_turn_index=2,
        injected_chunk_id=chunk.chunk_id,
    )
    conv = BenchmarkConversation(
        conversation_id="c3",
        persona_id="p_test",
        benchmark="counterfactual_probes",
        user_turns=["t0", "t1", "probe!", "t3"],
        probe=probe,
    )
    _, logs = runner.replay(persona, [conv])
    assert len(logs) == 1
    assert logs[0].injected_chunk_in_topk is True
    assert logs[0].injected_chunk_rank == 1


def test_counterfactual_probe_ejects_even_on_pipeline_error(persona: Persona) -> None:
    """If the pipeline raises mid-probe, the eject must still fire."""

    class _RaisingPipeline:
        def respond(self, *args: Any, **kwargs: Any) -> _FakeResponse:
            raise RuntimeError("backend exploded")

    store = _RecordingStore()
    chunk = CounterfactualChunk(
        chunk_id="should_be_ejected",
        persona_id="p_test",
        contradicts="p_test::worldview[0]",
        text="x",
    )
    runner = ProbeRunner(
        pipeline=_RaisingPipeline(),
        knowledge_store=store,
        chunks={chunk.chunk_id: chunk},
        mechanism_label="fake",
    )
    probe = DriftProbe(
        probe_id="raise",
        probe_type="counterfactual",
        probe_turn_index=0,
        injected_chunk_id=chunk.chunk_id,
    )
    conv = BenchmarkConversation(
        conversation_id="c_raise",
        persona_id="p_test",
        benchmark="x",
        user_turns=["t0"],
        probe=probe,
    )
    with pytest.raises(RuntimeError, match="backend exploded"):
        runner.replay(persona, [conv])
    # The eject must still have fired so the next conversation sees a clean store.
    assert store.add_calls == [[chunk.chunk_id]]
    assert store.remove_calls == [[chunk.chunk_id]]


def test_self_fact_probe_does_not_touch_store(persona: Persona) -> None:
    pipe = _RecordingPipeline()
    store = _RecordingStore()
    runner = ProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},
        mechanism_label="fake",
    )
    probe = DriftProbe(
        probe_id="sa",
        probe_type="self_fact_challenge",
        probe_turn_index=1,
    )
    conv = BenchmarkConversation(
        conversation_id="c_sa",
        persona_id="p_test",
        benchmark="counterfactual_probes",
        user_turns=["a", "b", "c"],
        probe=probe,
    )
    transcripts, logs = runner.replay(persona, [conv])
    assert store.add_calls == []
    assert store.remove_calls == []
    assert logs == []
    # is_probe_turn metadata still flagged correctly even without injection.
    assert transcripts[0].per_turn_metadata[1]["is_probe_turn"] is True
    assert transcripts[0].per_turn_metadata[1]["probe_type"] == "self_fact_challenge"


def test_unknown_chunk_raises_keyerror(persona: Persona, tmp_path: Path) -> None:
    pipe = _RecordingPipeline()
    store = _RecordingStore()
    runner = ProbeRunner(
        pipeline=pipe,
        knowledge_store=store,
        chunks={},  # empty — chunk lookup will fail
        mechanism_label="fake",
    )
    probe = DriftProbe(
        probe_id="cf_missing",
        probe_type="counterfactual",
        probe_turn_index=0,
        injected_chunk_id="not_loaded",
    )
    conv = BenchmarkConversation(
        conversation_id="c_missing",
        persona_id="p_test",
        benchmark="counterfactual_probes",
        user_turns=["t0"],
        probe=probe,
    )
    with pytest.raises(KeyError, match="not_loaded"):
        runner.replay(persona, [conv])
