"""Tests for the typed-retrieval mechanism (per-turn typed-store retrieval + ID-RAG)."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval import TypedRetrievalRAG
from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.typed_retrieval import (
    DEFAULT_EPISTEMIC_ALLOWLIST,
    render_typed_system_block,
)
from persona_rag.schema.chunker import PersonaChunk, chunk_persona
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


# --------------------------------------------------------------- fixtures


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


@pytest.fixture
def historian() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "historian.yaml")


@pytest.fixture
def climate() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "climate_scientist.yaml")


@pytest.fixture
def typed_stores(tmp_path, fake_embedder):
    path = tmp_path / "chroma_persona"
    identity = IdentityStore(path, embedding_function=fake_embedder)
    self_facts = SelfFactsStore(path, embedding_function=fake_embedder)
    worldview = WorldviewStore(path, embedding_function=fake_embedder)
    episodic = EpisodicStore(path, embedding_function=fake_embedder)
    return identity, self_facts, worldview, episodic


@pytest.fixture
def knowledge_store(tmp_path, fake_embedder) -> KnowledgeStore:
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


def _index_persona(stores, persona: Persona) -> None:
    chunks = chunk_persona(persona)
    for store in stores:
        store.index(chunks)


@pytest.fixture
def m1(fake_backend, knowledge_store, typed_stores) -> TypedRetrievalRAG:
    identity, self_facts, worldview, episodic = typed_stores
    return TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
    )


# --------------------------------------------------------------- response shape


def test_response_shape(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("What is Raft?", cs_tutor)
    assert r.text.startswith("[fake]")
    # All five typed-memory keys are populated (some may be empty lists).
    for kind in ("identity", "constraint", "self_fact", "worldview", "episodic"):
        assert kind in r.retrieved_persona
    # Knowledge retrieval went through.
    assert len(r.retrieved_knowledge) > 0
    # No steering / drift signal on M1.
    assert r.steering_applied is False
    assert r.drift_signal is None


# --------------------------------------------------------------- ID-RAG


def test_identity_always_retrieved_on_first_turn(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("Anything", cs_tutor)
    assert any(c.kind == "identity" for c in r.retrieved_persona["identity"])
    assert r.metadata["id_rag_fired"] is True


def test_identity_present_in_prompt_on_first_turn(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("Anything", cs_tutor)
    # Persona name appears in the rendered prompt.
    assert cs_tutor.identity.name in r.prompt_used


def test_id_rag_fires_every_turn_when_enabled(
    fake_backend, knowledge_store, typed_stores, cs_tutor
):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        use_identity_every_turn=True,
    )
    history = [
        Turn(role="user", content="Earlier message."),
        Turn(role="assistant", content="Earlier reply."),
        Turn(role="user", content="Another message."),
        Turn(role="assistant", content="Another reply."),
    ]
    r = pipeline.respond("Now I have a question.", cs_tutor, history=history)
    assert r.metadata["id_rag_fired"] is True
    assert any(c.kind == "identity" for c in r.retrieved_persona["identity"])
    assert cs_tutor.identity.name in r.prompt_used


def test_id_rag_off_drops_identity_after_turn_zero(
    fake_backend, knowledge_store, typed_stores, cs_tutor
):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        use_identity_every_turn=False,
    )
    # Turn 0 — should still re-ground.
    r0 = pipeline.respond("First question.", cs_tutor)
    assert r0.metadata["id_rag_fired"] is True
    assert cs_tutor.identity.name in r0.prompt_used

    # Turn 1+ with use_identity_every_turn=False — identity dropped from prompt.
    history = [
        Turn(role="user", content="First question."),
        Turn(role="assistant", content="First answer."),
    ]
    r1 = pipeline.respond("Second question.", cs_tutor, history=history)
    assert r1.metadata["id_rag_fired"] is False
    assert r1.retrieved_persona["identity"] == []
    assert r1.retrieved_persona["constraint"] == []
    assert cs_tutor.identity.name not in r1.prompt_used


# --------------------------------------------------------------- epistemic tags


def test_epistemic_tags_render_in_prompt(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("What do you think about microservices?", cs_tutor)
    # cs_tutor's worldview includes belief, fact, and contested tags. Default
    # rendering surfaces them as parens after the claim text.
    rendered = r.prompt_used
    has_any_tag = any(f"({tag})" in rendered for tag in DEFAULT_EPISTEMIC_ALLOWLIST)
    assert has_any_tag, "Expected at least one (epistemic-tag) annotation in the prompt"


def test_epistemic_tags_off_strips_annotation(
    fake_backend, knowledge_store, typed_stores, cs_tutor
):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        use_epistemic_tags=False,
    )
    r = pipeline.respond("What do you think about microservices?", cs_tutor)
    rendered = r.prompt_used
    for tag in DEFAULT_EPISTEMIC_ALLOWLIST:
        assert f"({tag})" not in rendered


def test_epistemic_allowlist_filters_worldview(
    fake_backend, knowledge_store, typed_stores, cs_tutor
):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    # Restrict to a single tag — only chunks with that epistemic value pass through.
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        epistemic_allowlist=("contested",),
        top_k_worldview=10,
    )
    r = pipeline.respond("Anything", cs_tutor)
    for c in r.retrieved_persona["worldview"]:
        assert c.metadata.get("epistemic") == "contested"


# --------------------------------------------------------------- per-persona differentiation


def test_different_personas_retrieve_different_content(
    fake_backend, knowledge_store, typed_stores, cs_tutor, historian
):
    _index_persona(typed_stores, cs_tutor)
    _index_persona(typed_stores, historian)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
    )
    q = "Tell me about your background."
    r_tutor = pipeline.respond(q, cs_tutor)
    r_hist = pipeline.respond(q, historian)
    # Each persona's retrieved chunks belong to that persona.
    for kind in ("self_fact", "worldview"):
        for c in r_tutor.retrieved_persona[kind]:
            assert c.metadata.get("persona_id") == "cs_tutor"
        for c in r_hist.retrieved_persona[kind]:
            assert c.metadata.get("persona_id") == "historian"
    # Different prompts → different generated text under our deterministic stub.
    assert r_tutor.prompt_used != r_hist.prompt_used
    assert r_tutor.text != r_hist.text


# --------------------------------------------------------------- episodic switches


def test_episodic_off_by_default(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("Anything", cs_tutor)
    assert r.retrieved_persona["episodic"] == []


def test_episodic_write_back_when_enabled(fake_backend, knowledge_store, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        write_episodic=True,
    )
    before = episodic.count()
    pipeline.respond("Discuss Raft consensus.", cs_tutor)
    after = episodic.count()
    assert after == before + 1


def test_episodic_query_when_use_episodic_on(fake_backend, knowledge_store, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    # Plant one episodic chunk so the query has something to pull.
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    episodic.write(
        PersonaChunk(
            id="cs_tutor:episodic:planted",
            text="We discussed Rust borrow-checker pitfalls earlier.",
            kind="episodic",
            metadata={
                "persona_id": "cs_tutor",
                "kind": "episodic",
                "timestamp": now.isoformat(),
                "decay_t0": now.isoformat(),
                "turn_id": "0",
            },
        )
    )
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        use_episodic=True,
        top_k_episodic=3,
    )
    r = pipeline.respond("anything", cs_tutor)
    # At least one episodic chunk pulled from the store.
    assert any(c.metadata.get("persona_id") == "cs_tutor" for c in r.retrieved_persona["episodic"])


# --------------------------------------------------------------- metadata


def test_metadata_records_ablation_flags_and_chunk_ids(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r = m1.respond("Anything", cs_tutor)
    md = r.metadata
    assert md["mechanism"] == "typed_retrieval"
    assert md["persona_id"] == "cs_tutor"
    assert md["use_identity_every_turn"] is True
    assert md["use_epistemic_tags"] is True
    assert md["use_episodic"] is False
    assert md["epistemic_allowlist"] == list(DEFAULT_EPISTEMIC_ALLOWLIST)
    assert md["top_k_self_facts"] == 3
    assert md["top_k_worldview"] == 3
    assert md["top_k_knowledge"] == 5
    # Chunk ids are listed for traceability.
    assert isinstance(md["self_fact_chunk_ids"], list)
    assert isinstance(md["worldview_chunk_ids"], list)
    assert isinstance(md["knowledge_chunk_ids"], list)
    # Worldview epistemic mix is a dict of tag → count.
    assert isinstance(md["worldview_epistemic_mix"], dict)


def test_seed_propagates_through_to_backend(m1, typed_stores, cs_tutor, fake_backend):
    _index_persona(typed_stores, cs_tutor)
    for seed in (42, 1337, 2024):
        r = m1.respond("anything", cs_tutor, seed=seed)
        assert r.metadata["seed"] == seed
    assert fake_backend.seeds_seen[-3:] == [42, 1337, 2024]


# --------------------------------------------------------------- determinism


def test_deterministic_under_same_inputs(m1, typed_stores, cs_tutor):
    _index_persona(typed_stores, cs_tutor)
    r1 = m1.respond("What is Raft?", cs_tutor)
    r2 = m1.respond("What is Raft?", cs_tutor)
    assert r1.prompt_used == r2.prompt_used
    assert r1.text == r2.text


# --------------------------------------------------------------- prompt-budget trimming


def test_long_history_triggers_history_trim(fake_backend, knowledge_store, typed_stores, cs_tutor):
    """Conversation history that overflows the prompt budget gets FIFO-dropped."""
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    # Tight budget so even a small history blows it. max_input_tokens is the
    # hard cap; max_new_tokens is reserved for generation; the implementation
    # subtracts a 128-token safety margin and the system+query overhead.
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
        max_new_tokens=64,
        max_input_tokens=512,
    )
    # Build a 20-turn fake history of long-ish chat lines — guaranteed to
    # exceed any reasonable budget under max_input_tokens=512.
    long_text = "x " * 200  # ~400 chars per turn
    history = []
    for i in range(20):
        history.append(Turn(role="user", content=f"User turn {i}: {long_text}"))
        history.append(Turn(role="assistant", content=f"Assistant turn {i}: {long_text}"))

    r = pipeline.respond("a final question", cs_tutor, history=history)
    # Some history pairs were dropped to fit.
    assert r.metadata["trimmed_history_turns"] > 0
    # The dropped count is even (we drop in pairs of user+assistant).
    assert r.metadata["trimmed_history_turns"] % 2 == 0


def test_short_history_is_not_trimmed(fake_backend, knowledge_store, typed_stores, cs_tutor):
    """A short history under any reasonable budget passes through untouched."""
    _index_persona(typed_stores, cs_tutor)
    identity, self_facts, worldview, episodic = typed_stores
    pipeline = TypedRetrievalRAG(
        backend=fake_backend,
        knowledge_store=knowledge_store,
        identity_store=identity,
        self_facts_store=self_facts,
        worldview_store=worldview,
        episodic_store=episodic,
    )
    history = [
        Turn(role="user", content="A short question."),
        Turn(role="assistant", content="A short reply."),
    ]
    r = pipeline.respond("Another question.", cs_tutor, history=history)
    assert r.metadata["trimmed_history_turns"] == 0


# --------------------------------------------------------------- guards


def test_persona_id_required(m1, knowledge_store, typed_stores):
    # Construct a persona with no persona_id — should fail loudly.
    p = Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")
    p.persona_id = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="persona_id"):
        m1.respond("anything", p)


# --------------------------------------------------------------- renderer


def test_render_typed_system_block_concat_order():
    """Identity → constraints → self_facts → worldview → episodic → knowledge."""
    identity = [
        PersonaChunk(
            id="p:identity:0",
            text="Identity-line",
            kind="identity",
            metadata={"persona_id": "p", "kind": "identity"},
        )
    ]
    constraints = [
        PersonaChunk(
            id="p:constraint:0",
            text="Constraint-line",
            kind="constraint",
            metadata={"persona_id": "p", "kind": "constraint"},
        )
    ]
    self_facts = [
        PersonaChunk(
            id="p:self_fact:0",
            text="Self-fact-line",
            kind="self_fact",
            metadata={"persona_id": "p", "kind": "self_fact", "epistemic": "fact"},
        )
    ]
    worldview = [
        PersonaChunk(
            id="p:worldview:0",
            text="Worldview-line",
            kind="worldview",
            metadata={"persona_id": "p", "kind": "worldview", "epistemic": "belief"},
        )
    ]
    block = render_typed_system_block(
        identity_chunks=identity,
        constraint_chunks=constraints,
        self_fact_chunks=self_facts,
        worldview_chunks=worldview,
        episodic_chunks=[],
        knowledge_chunks=[],
    )
    # Each section appears, in the documented order.
    ix_identity = block.index("Identity-line")
    ix_constraint = block.index("Constraint-line")
    ix_self_fact = block.index("Self-fact-line")
    ix_worldview = block.index("Worldview-line")
    assert ix_identity < ix_constraint < ix_self_fact < ix_worldview
    # Epistemic tag rendered.
    assert "(belief)" in block
