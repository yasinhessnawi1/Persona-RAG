"""Tests for the vanilla RAG pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.vanilla_rag import VanillaRAG
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_store import KnowledgeDocument, KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")


@pytest.fixture
def historian() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "historian.yaml")


@pytest.fixture
def store(tmp_path, fake_embedder) -> KnowledgeStore:
    s = KnowledgeStore(
        persist_path=tmp_path / "ks",
        collection_name="ks_test",
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


def test_b1_response_shape(fake_backend, store, cs_tutor) -> None:
    """B1 returns a `Response` with retrieved knowledge and empty persona."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert r.text.startswith("[fake]")
    assert r.retrieved_persona == {}
    assert len(r.retrieved_knowledge) > 0
    assert r.steering_applied is False
    assert r.drift_signal is None


def test_b1_prompt_contains_no_persona_text(fake_backend, store, cs_tutor) -> None:
    """The B1 prompt must NOT mention the persona's name or role — it's the floor baseline."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert cs_tutor.identity.name not in r.prompt_used
    assert cs_tutor.identity.role not in r.prompt_used
    # And the system block uses the literature-standard framing.
    assert "helpful assistant" in r.prompt_used


def test_b1_metadata_records_persona_ignored_and_baseline(fake_backend, store, cs_tutor) -> None:
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert r.metadata["persona_ignored"] is True
    assert r.metadata["baseline"] == "vanilla_rag"
    assert r.metadata["fusion_mode"] == "rrf"
    assert r.metadata["persona_id"] == cs_tutor.persona_id


def test_b1_output_is_persona_invariant(fake_backend, store, cs_tutor, historian) -> None:
    """Same query against different personas → identical generated prompt (and reply)."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r_tutor = pipeline.respond("Tell me about CAP.", cs_tutor)
    r_hist = pipeline.respond("Tell me about CAP.", historian)
    assert r_tutor.prompt_used == r_hist.prompt_used
    assert r_tutor.text == r_hist.text


def test_b1_deterministic_under_same_seed_and_query(fake_backend, store, cs_tutor) -> None:
    """Same query → identical reply (FakeBackend is hash-deterministic)."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r1 = pipeline.respond("What is Raft?", cs_tutor)
    r2 = pipeline.respond("What is Raft?", cs_tutor)
    assert r1.prompt_used == r2.prompt_used
    assert r1.text == r2.text


def test_b1_weighted_sum_alpha_is_recorded(fake_backend, store, cs_tutor) -> None:
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store, alpha=0.5)
    r = pipeline.respond("Raft", cs_tutor)
    assert "weighted_sum" in r.metadata["fusion_mode"]


def test_b1_seed_propagates_to_backend_and_metadata(fake_backend, store, cs_tutor) -> None:
    """`respond(..., seed=N)` forwards N to backend.generate and records it in metadata."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r = pipeline.respond("What is Raft?", cs_tutor, seed=1337)
    assert fake_backend.seeds_seen[-1] == 1337
    assert r.metadata["seed"] == 1337


def test_b1_seed_default_none(fake_backend, store, cs_tutor) -> None:
    """Without an explicit seed the backend gets None (its own default applies)."""
    pipeline = VanillaRAG(backend=fake_backend, knowledge_store=store)
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert fake_backend.seeds_seen[-1] is None
    assert r.metadata["seed"] is None
