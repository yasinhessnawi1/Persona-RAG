"""Tests for the drift-gated hybrid mechanism (M3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from persona_rag.retrieval import (
    DriftGatedMechanism,
    FakeCharacterRMScorer,
    HybridRanker,
    LlmJudgeDriftGate,
    TypedRetrievalRAG,
)
from persona_rag.retrieval.base import Turn
from persona_rag.schema.chunker import chunk_persona
from persona_rag.schema.persona import Persona
from persona_rag.stores import (
    EpisodicStore,
    IdentityStore,
    SelfFactsStore,
    WorldviewStore,
)
from persona_rag.stores.knowledge_store import KnowledgeDocument, KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


@pytest.fixture
def typed_stores(tmp_path, fake_embedder):
    path = tmp_path / "chroma_persona"
    return (
        IdentityStore(path, embedding_function=fake_embedder),
        SelfFactsStore(path, embedding_function=fake_embedder),
        WorldviewStore(path, embedding_function=fake_embedder),
        EpisodicStore(path, embedding_function=fake_embedder),
    )


@pytest.fixture
def knowledge_store(tmp_path, fake_embedder) -> KnowledgeStore:
    s = KnowledgeStore(
        persist_path=tmp_path / "ks",
        collection_name="ks_test_m3",
        embedding_function=fake_embedder,
        chunk_size=512,
        chunk_overlap=0,
    )
    s.index_corpus(
        [
            KnowledgeDocument(
                doc_id="raft",
                text="Raft is a consensus algorithm with leader election.",
                source="raft.md",
            ),
            KnowledgeDocument(
                doc_id="cap",
                text="CAP describes the consistency-availability-partition tradeoff.",
                source="cap.md",
            ),
        ]
    )
    return s


@pytest.fixture
def m1(fake_backend, knowledge_store, typed_stores, cs_tutor) -> TypedRetrievalRAG:
    identity, self_facts, worldview, episodic = typed_stores
    chunks = chunk_persona(cs_tutor)
    for store in typed_stores:
        store.index(chunks)
    return TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
    )


class _ScriptedBackend:
    """LLMBackend stub that maps prompt → response from a script.

    The script is a list of (substring_match, response) pairs. The first pair
    whose substring appears in the prompt fires; if none match, a default
    deterministic response is returned. Used to drive the gate, rerank judge,
    and extra-candidate generations from one place.
    """

    name = "scripted-backend"
    model_id = "fake/scripted-backend"
    num_layers = 0
    hidden_dim = 0

    def __init__(self, script: list[tuple[str, str]]) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []
        self.seeds_seen: list[int | None] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        self.seeds_seen.append(seed)
        for needle, response in self.script:
            if needle in prompt:
                self.calls.append((needle, response))
                return response
        # Fallback — make sure outputs are deterministic but varied so the
        # ranker doesn't see identical extra candidates.
        digest = hashlib.sha256(f"{prompt}|{temperature}|{seed}".encode()).hexdigest()[:8]
        out = f"[scripted-fallback temp={temperature:.2f} seed={seed}] {digest}"
        self.calls.append(("(fallback)", out))
        return out


# --------------------------------------------------------- cheap path


def test_cheap_path_when_gate_returns_ok(fake_backend, m1, cs_tutor) -> None:
    gate_judge = _ScriptedBackend(
        [("you are an evaluator", "flag: ok\nconfidence: 0.95\nrationale: in voice")]
    )
    rerank_judge = _ScriptedBackend([])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(
        character_rm=FakeCharacterRMScorer(),
        rerank_judge=rerank_judge,
    )
    m3 = DriftGatedMechanism(backend=fake_backend, m1=m1, drift_gate=gate, hybrid_ranker=ranker)
    history = [
        Turn(role="user", content="prior question"),
        Turn(role="assistant", content="prior answer"),
    ]
    response = m3.respond("a new question", cs_tutor, history=history)
    md = response.metadata
    assert md["mechanism"] == "drift_gated"
    assert md["path_taken"] == "cheap"
    assert md["drift_gated"] is False
    assert md["gate_flag"] == "ok"
    assert md["llm_call_count"]["total"] == 2
    assert md["n_candidates_generated"] == 1
    # Rerank judge was never called on the cheap path.
    assert rerank_judge.calls == []


def test_cheap_path_on_turn_zero(fake_backend, m1, cs_tutor) -> None:
    """Turn 0 always takes the cheap path; gate is not called."""
    gate_judge = _ScriptedBackend([])
    rerank_judge = _ScriptedBackend([])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(character_rm=FakeCharacterRMScorer(), rerank_judge=rerank_judge)
    m3 = DriftGatedMechanism(backend=fake_backend, m1=m1, drift_gate=gate, hybrid_ranker=ranker)
    response = m3.respond("first question", cs_tutor, history=[])
    assert response.metadata["path_taken"] == "cheap"
    # Gate-judge LLM call avoided on turn 0.
    assert gate_judge.calls == []


# --------------------------------------------------------- gated path


def test_gated_path_runs_full_pipeline(fake_backend, m1, cs_tutor) -> None:
    gate_judge = _ScriptedBackend(
        [("you are an evaluator", "flag: drift\nconfidence: 0.9\nrationale: off-persona")]
    )
    rerank_judge = _ScriptedBackend([("Evaluate how well", "[RESULT] 4")])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(character_rm=FakeCharacterRMScorer(), rerank_judge=rerank_judge)
    m3 = DriftGatedMechanism(
        backend=fake_backend,
        m1=m1,
        drift_gate=gate,
        hybrid_ranker=ranker,
        n_candidates=3,
        extra_candidate_temperatures=(0.7, 1.0),
    )
    history = [
        Turn(role="user", content="prior question"),
        Turn(role="assistant", content="prior off-voice answer"),
    ]
    response = m3.respond("a follow-up question", cs_tutor, history=history)
    md = response.metadata
    assert md["path_taken"] == "gated"
    assert md["drift_gated"] is True
    assert md["gate_flag"] == "drift"
    # Three candidates ranked: M1 baseline (1) + 2 extra at varied temps.
    assert md["n_candidates_generated"] == 3
    assert md["n_extra_candidates"] == 2
    # Cost: gate(1) + responder(1) + extra(2) + ranker_character_rm(3) + ranker_judge(3) = 10
    assert md["llm_call_count"] == {
        "gate": 1,
        "responder": 1,
        "candidates": 2,
        "ranker_character_rm": 3,
        "ranker_judge": 3,
        "total": 10,
    }
    # Per-candidate breakdown is logged.
    assert len(md["candidate_scores"]) == 3
    # M1's augmented retrieval shows up in M1's metadata.
    assert md["augmented_for_drift"] is True


def test_gated_path_falls_back_to_judge_only_when_character_rm_disabled(
    fake_backend, m1, cs_tutor
) -> None:
    """``enabled_signals=["judge"]`` runs the gated path cleanly.

    Verifies the ranker degrades to a 1-signal LLM-judge ranker without
    code change.
    """
    gate_judge = _ScriptedBackend(
        [("you are an evaluator", "flag: drift\nconfidence: 0.9\nrationale: x")]
    )
    rerank_judge = _ScriptedBackend([("Evaluate how well", "[RESULT] 5")])
    char_rm = FakeCharacterRMScorer()
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(
        character_rm=char_rm,
        rerank_judge=rerank_judge,
        enabled_signals=("judge",),
    )
    m3 = DriftGatedMechanism(
        backend=fake_backend,
        m1=m1,
        drift_gate=gate,
        hybrid_ranker=ranker,
    )
    history = [Turn(role="user", content="x"), Turn(role="assistant", content="y")]
    response = m3.respond("q", cs_tutor, history=history)
    md = response.metadata
    assert md["ranker_signals"] == ["judge"]
    assert md["llm_call_count"]["ranker_character_rm"] == 0
    assert md["llm_call_count"]["ranker_judge"] == 3
    # CharacterRM scorer's `score()` was never invoked.
    assert char_rm.calls == []


def test_n_candidates_one_collapses_to_baseline_only(fake_backend, m1, cs_tutor) -> None:
    """``n_candidates=1`` is allowed; ranker sorts a single candidate."""
    gate_judge = _ScriptedBackend(
        [("you are an evaluator", "flag: drift\nconfidence: 0.9\nrationale: x")]
    )
    rerank_judge = _ScriptedBackend([("Evaluate how well", "[RESULT] 3")])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(character_rm=FakeCharacterRMScorer(), rerank_judge=rerank_judge)
    m3 = DriftGatedMechanism(
        backend=fake_backend,
        m1=m1,
        drift_gate=gate,
        hybrid_ranker=ranker,
        n_candidates=1,
        extra_candidate_temperatures=(),
    )
    history = [Turn(role="user", content="x"), Turn(role="assistant", content="y")]
    response = m3.respond("q", cs_tutor, history=history)
    assert response.metadata["n_candidates_generated"] == 1
    assert response.metadata["n_extra_candidates"] == 0


def test_temperatures_must_match_n_candidates(fake_backend, m1) -> None:
    """Mismatched ``extra_candidate_temperatures`` length fails fast."""
    gate_judge = _ScriptedBackend([])
    rerank_judge = _ScriptedBackend([])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(character_rm=FakeCharacterRMScorer(), rerank_judge=rerank_judge)
    with pytest.raises(ValueError, match="extra_candidate_temperatures"):
        DriftGatedMechanism(
            backend=ranker.rerank_judge,
            m1=m1,
            drift_gate=gate,
            hybrid_ranker=ranker,
            n_candidates=3,
            extra_candidate_temperatures=(0.7,),  # one too few
        )


def test_seed_propagation_distinct_per_candidate(fake_backend, m1, cs_tutor) -> None:
    """Each non-baseline candidate gets a distinct derived seed."""
    gate_judge = _ScriptedBackend(
        [("you are an evaluator", "flag: drift\nconfidence: 0.9\nrationale: x")]
    )
    rerank_judge = _ScriptedBackend([("Evaluate how well", "[RESULT] 3")])
    extra_backend = _ScriptedBackend([])
    gate = LlmJudgeDriftGate(judge=gate_judge, confidence_threshold=0.5)
    ranker = HybridRanker(character_rm=FakeCharacterRMScorer(), rerank_judge=rerank_judge)
    m3 = DriftGatedMechanism(
        backend=extra_backend,
        m1=m1,
        drift_gate=gate,
        hybrid_ranker=ranker,
        n_candidates=3,
        extra_candidate_temperatures=(0.7, 1.0),
    )
    history = [Turn(role="user", content="x"), Turn(role="assistant", content="y")]
    m3.respond("q", cs_tutor, history=history, seed=42)
    # Two extra-candidate generations → two seeds, distinct from each other.
    assert len(extra_backend.seeds_seen) == 2
    assert extra_backend.seeds_seen[0] != extra_backend.seeds_seen[1]
    # Both derived from the input seed (offsets +1 and +2).
    assert extra_backend.seeds_seen == [43, 44]
