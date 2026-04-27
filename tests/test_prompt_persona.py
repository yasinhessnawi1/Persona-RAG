"""Tests for the prompt-persona pipeline (structured persona block + few-shots)."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.prompt_persona import PromptPersonaRAG
from persona_rag.retrieval.prompt_templates import FewShotBundle
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_store import KnowledgeDocument, KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "personas" / "examples"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "cs_tutor.yaml")


@pytest.fixture
def historian() -> Persona:
    return Persona.from_yaml(REPO_ROOT / "personas" / "historian.yaml")


@pytest.fixture
def cs_tutor_few_shots() -> FewShotBundle:
    return FewShotBundle.from_yaml(EXAMPLES_DIR / "cs_tutor.yaml")


@pytest.fixture
def historian_few_shots() -> FewShotBundle:
    return FewShotBundle.from_yaml(EXAMPLES_DIR / "historian.yaml")


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


# -------------------------------------------------------------- B2 v03 (headline)


def test_b2_response_shape(fake_backend, store, cs_tutor, cs_tutor_few_shots) -> None:
    """B2 returns a `Response` with knowledge but persona via system block, not retrieval."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend,
        knowledge_store=store,
        few_shots=cs_tutor_few_shots,
    )
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert r.text.startswith("[fake]")
    assert r.retrieved_persona == {}  # B2 puts persona in system block
    assert len(r.retrieved_knowledge) > 0
    assert r.steering_applied is False


def test_b2_prompt_contains_persona_identity_and_constraint(
    fake_backend, store, cs_tutor, cs_tutor_few_shots
) -> None:
    """Persona identity (name + role) and every constraint must surface in the final prompt."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert cs_tutor.identity.name in r.prompt_used
    assert cs_tutor.identity.role in r.prompt_used
    # Every constraint must appear verbatim.
    for c in cs_tutor.identity.constraints:
        assert c in r.prompt_used


def test_b2_prompt_contains_few_shots(fake_backend, store, cs_tutor, cs_tutor_few_shots) -> None:
    """All 5 few-shot example titles surface in the system block."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    r = pipeline.respond("What is Raft?", cs_tutor)
    for ex in cs_tutor_few_shots.exchanges:
        assert ex.title in r.prompt_used


def test_b2_metadata_records_constraint_case_count(
    fake_backend, store, cs_tutor, cs_tutor_few_shots
) -> None:
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    r = pipeline.respond("Anything", cs_tutor)
    assert r.metadata["baseline"] == "prompt_persona"
    assert r.metadata["b2_variant"] == "v03"
    assert r.metadata["few_shot_count"] == 5
    assert r.metadata["constraint_case_count"] == 1
    assert r.metadata["persona_id"] == cs_tutor.persona_id


def test_b2_output_varies_across_personas(
    fake_backend, store, cs_tutor, cs_tutor_few_shots, historian, historian_few_shots
) -> None:
    """Same query, different personas → different generated prompt (and reply)."""
    p_tutor = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    p_hist = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=historian_few_shots
    )
    r_tutor = p_tutor.respond("Tell me about your background.", cs_tutor)
    r_hist = p_hist.respond("Tell me about your background.", historian)
    assert r_tutor.prompt_used != r_hist.prompt_used
    assert r_tutor.text != r_hist.text


def test_b2_rejects_mismatched_few_shot_bundle(
    fake_backend, store, cs_tutor, historian_few_shots
) -> None:
    """Pairing the wrong persona with the wrong few-shot bundle is a config bug — fail loudly."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=historian_few_shots
    )
    with pytest.raises(ValueError, match="does not match"):
        pipeline.respond("anything", cs_tutor)


# -------------------------------------------------------------- B2 v02 audit shim


def test_b2_v02_one_liner_does_not_include_few_shots(
    fake_backend, store, cs_tutor, cs_tutor_few_shots
) -> None:
    """The audit-only one-liner shim has no few-shots."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend,
        knowledge_store=store,
        few_shots=cs_tutor_few_shots,
        b2_variant="v02_one_liner",
    )
    r = pipeline.respond("What is Raft?", cs_tutor)
    assert cs_tutor.identity.name in r.prompt_used
    # No few-shot titles in the one-liner shim.
    for ex in cs_tutor_few_shots.exchanges:
        assert ex.title not in r.prompt_used
    # Constraints not in the simple shim.
    assert "You must NOT" not in r.prompt_used
    assert r.metadata["b2_variant"] == "v02_one_liner"


def test_b2_invalid_variant_raises(fake_backend, store, cs_tutor, cs_tutor_few_shots) -> None:
    pipeline = PromptPersonaRAG(
        backend=fake_backend,
        knowledge_store=store,
        few_shots=cs_tutor_few_shots,
        b2_variant="bogus",
    )
    with pytest.raises(ValueError, match="b2_variant"):
        pipeline.respond("anything", cs_tutor)


# -------------------------------------------------------------- determinism


def test_b2_deterministic_under_same_inputs(
    fake_backend, store, cs_tutor, cs_tutor_few_shots
) -> None:
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    r1 = pipeline.respond("What is Raft?", cs_tutor)
    r2 = pipeline.respond("What is Raft?", cs_tutor)
    assert r1.prompt_used == r2.prompt_used
    assert r1.text == r2.text


def test_b2_seed_propagates_to_backend_and_metadata(
    fake_backend, store, cs_tutor, cs_tutor_few_shots
) -> None:
    """Multi-seed dispatch from the entry-point script must reach the backend cleanly."""
    pipeline = PromptPersonaRAG(
        backend=fake_backend, knowledge_store=store, few_shots=cs_tutor_few_shots
    )
    for seed in (42, 1337, 2024):
        r = pipeline.respond("What is Raft?", cs_tutor, seed=seed)
        assert r.metadata["seed"] == seed
    assert fake_backend.seeds_seen[-3:] == [42, 1337, 2024]
