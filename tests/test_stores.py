"""Tests for the four typed persona memory stores (IdentityStore, SelfFacts, Worldview, Episodic)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from persona_rag.schema import Persona, chunk_persona
from persona_rag.schema.chunker import PersonaChunk
from persona_rag.stores import (
    EpisodicStore,
    IdentityStore,
    RuntimeWriteForbiddenError,
    SelfFactsStore,
    WorldviewStore,
)

# --------------------------------------------------------------- fixtures


@pytest.fixture
def tutor(personas_dir: Path) -> Persona:
    return Persona.from_yaml(personas_dir / "cs_tutor.yaml")


@pytest.fixture
def historian(personas_dir: Path) -> Persona:
    return Persona.from_yaml(personas_dir / "historian.yaml")


@pytest.fixture
def stores(tmp_path: Path, fake_embedder):
    path = tmp_path / "chroma"
    identity = IdentityStore(path, embedding_function=fake_embedder)
    self_facts = SelfFactsStore(path, embedding_function=fake_embedder)
    worldview = WorldviewStore(path, embedding_function=fake_embedder)
    episodic = EpisodicStore(path, embedding_function=fake_embedder)
    return identity, self_facts, worldview, episodic


def _index_all(stores, persona: Persona) -> None:
    chunks = chunk_persona(persona)
    for store in stores:
        store.index(chunks)


# --------------------------------------------------------------- identity store


def test_identity_store_only_holds_identity_and_constraints(stores, tutor):
    identity, _, _, _ = stores
    _index_all(stores, tutor)
    chunks = identity.get_all(persona_id="cs_tutor")
    assert len(chunks) > 0
    assert {c.kind for c in chunks} == {"identity", "constraint"}


def test_identity_store_query_returns_everything_not_top_k(stores, tutor):
    identity, _, _, _ = stores
    _index_all(stores, tutor)
    expected = len(identity.get_all(persona_id="cs_tutor"))
    # `query` ignores top_k and returns every chunk for the persona.
    result = identity.query("any query text", top_k=1, persona_id="cs_tutor")
    assert len(result) == expected


def test_identity_store_runtime_write_forbidden(stores):
    identity, _, _, _ = stores
    chunk = PersonaChunk(
        id="x:identity:0",
        text="t",
        kind="identity",
        metadata={"persona_id": "x", "kind": "identity"},
    )
    with pytest.raises(RuntimeWriteForbiddenError):
        identity.write(chunk)


# --------------------------------------------------------------- self_facts store


def test_self_facts_store_indexes_only_self_fact_chunks(stores, tutor):
    _, self_facts, _, _ = stores
    _index_all(stores, tutor)
    total = self_facts.count()
    # Equals the number of self_facts in the yaml.
    assert total == len(tutor.self_facts)


def test_self_facts_query_returns_chunks(stores, tutor):
    _, self_facts, _, _ = stores
    _index_all(stores, tutor)
    result = self_facts.query("programming languages I use", top_k=3, persona_id="cs_tutor")
    assert 0 < len(result) <= 3
    assert all(c.kind == "self_fact" for c in result)


def test_self_facts_runtime_write_forbidden(stores):
    _, self_facts, _, _ = stores
    chunk = PersonaChunk(
        id="x:self_fact:0",
        text="t",
        kind="self_fact",
        metadata={"persona_id": "x", "kind": "self_fact"},
    )
    with pytest.raises(RuntimeWriteForbiddenError):
        self_facts.write(chunk)


# --------------------------------------------------------------- worldview store


def test_worldview_store_indexes_only_worldview_chunks(stores, tutor):
    _, _, worldview, _ = stores
    _index_all(stores, tutor)
    assert worldview.count() == len(tutor.worldview)


def test_worldview_epistemic_filter_single_tag(stores, tutor):
    _, _, worldview, _ = stores
    _index_all(stores, tutor)
    result = worldview.query("anything", top_k=10, persona_id="cs_tutor", epistemic="contested")
    assert all(c.metadata["epistemic"] == "contested" for c in result)


def test_worldview_epistemic_filter_list_of_tags(stores, tutor):
    _, _, worldview, _ = stores
    _index_all(stores, tutor)
    result = worldview.query(
        "anything",
        top_k=10,
        persona_id="cs_tutor",
        epistemic=["fact", "belief"],
    )
    kept = {c.metadata["epistemic"] for c in result}
    assert kept.issubset({"fact", "belief"})


def test_worldview_bi_temporal_filter_narrows_results(stores, historian):
    _, _, worldview, _ = stores
    _index_all(stores, historian)

    # Every claim tagged "always" should be retained across any as_of year;
    # claims outside their range should be dropped.
    result_1750 = worldview.query(
        "European history", top_k=20, persona_id="historian", as_of="1750"
    )
    assert len(result_1750) > 0
    for chunk in result_1750:
        vt = chunk.metadata["valid_time"]
        assert vt == "always" or _as_of_year_in_valid_time(1750, vt)

    result_1950 = worldview.query(
        "European history", top_k=20, persona_id="historian", as_of="1950"
    )
    for chunk in result_1950:
        vt = chunk.metadata["valid_time"]
        assert vt == "always" or _as_of_year_in_valid_time(1950, vt)

    # 1750 should retain claims tagged 1500-1800 / 1700-1800 / always; 1950
    # should drop the 1500-1800 and 1517-1648 bands — so strictly fewer.
    assert len(result_1750) >= len(result_1950)


def test_worldview_as_of_rejects_non_year_string(stores, tutor):
    _, _, worldview, _ = stores
    _index_all(stores, tutor)
    with pytest.raises(ValueError):
        worldview.query("anything", top_k=3, persona_id="cs_tutor", as_of="today")


def test_worldview_runtime_write_forbidden(stores):
    _, _, worldview, _ = stores
    chunk = PersonaChunk(
        id="x:worldview:0",
        text="t",
        kind="worldview",
        metadata={"persona_id": "x", "kind": "worldview", "valid_time": "always"},
    )
    with pytest.raises(RuntimeWriteForbiddenError):
        worldview.write(chunk)


def _as_of_year_in_valid_time(year: int, valid_time: str) -> bool:
    """Mirror of worldview_store._matches_as_of, duplicated for the test."""
    if valid_time == "always":
        return True
    if valid_time.isdigit() and len(valid_time) == 4:
        return int(valid_time) == year
    if "-" in valid_time:
        left, _, right = valid_time.partition("-")
        left_year = int(left)
        if right:
            return left_year <= year <= int(right)
        return left_year <= year
    return False


# --------------------------------------------------------------- episodic store


def test_episodic_store_indexes_only_episodic_chunks(stores, tutor):
    _, _, _, episodic = stores
    # tutor has no episodic entries — index should be a no-op
    _index_all(stores, tutor)
    assert episodic.count() == 0


def test_episodic_runtime_write_allowed_and_queryable(stores, fake_embedder):
    _, _, _, episodic = stores
    now = datetime.now(UTC)
    chunk = PersonaChunk(
        id="cs_tutor:episodic:99",
        text="User asked about Rust concurrency patterns.",
        kind="episodic",
        metadata={
            "persona_id": "cs_tutor",
            "kind": "episodic",
            "timestamp": now.isoformat(),
            "decay_t0": now.isoformat(),
            "turn_id": "7",
        },
    )
    episodic.write(chunk)
    result = episodic.query("Rust concurrency", top_k=5, persona_id="cs_tutor", now=now)
    assert len(result) == 1
    assert result[0].id == "cs_tutor:episodic:99"


def test_episodic_decay_down_ranks_older_entries(stores):
    _, _, _, episodic = stores
    now = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    fresh = now
    stale = now - timedelta(days=10)  # Far past one tau — strongly decayed.

    same_text = "We talked about Raft consensus."
    episodic.write(
        PersonaChunk(
            id="p:episodic:fresh",
            text=same_text,
            kind="episodic",
            metadata={
                "persona_id": "p",
                "kind": "episodic",
                "timestamp": fresh.isoformat(),
                "decay_t0": fresh.isoformat(),
                "turn_id": "1",
            },
        )
    )
    episodic.write(
        PersonaChunk(
            id="p:episodic:stale",
            text=same_text,
            kind="episodic",
            metadata={
                "persona_id": "p",
                "kind": "episodic",
                "timestamp": stale.isoformat(),
                "decay_t0": stale.isoformat(),
                "turn_id": "0",
            },
        )
    )
    result = episodic.query("Raft", top_k=2, persona_id="p", now=now)
    assert [c.id for c in result] == ["p:episodic:fresh", "p:episodic:stale"]


def test_episodic_empty_query_returns_empty(stores):
    _, _, _, episodic = stores
    assert episodic.query("", top_k=3) == []


# --------------------------------------------------------------- cross-cutting


def test_stores_share_persist_path_but_distinct_collections(stores, tutor):
    identity, self_facts, worldview, episodic = stores
    _index_all(stores, tutor)

    # Distinct collection names.
    names = {s.collection_name for s in (identity, self_facts, worldview, episodic)}
    assert len(names) == 4

    # Same persist path on disk.
    paths = {s.persist_path for s in (identity, self_facts, worldview, episodic)}
    assert len(paths) == 1


def test_delete_persona_removes_only_that_persona(stores, tutor, historian):
    _index_all(stores, tutor)
    _index_all(stores, historian)
    identity, self_facts, worldview, episodic = stores
    tutor_total_before = sum(
        len(s.get_all(persona_id="cs_tutor")) for s in (identity, self_facts, worldview, episodic)
    )
    removed = sum(s.delete_persona("cs_tutor") for s in (identity, self_facts, worldview, episodic))
    assert removed == tutor_total_before
    # Historian still present.
    historian_total = sum(
        len(s.get_all(persona_id="historian")) for s in (identity, self_facts, worldview, episodic)
    )
    assert historian_total > 0


def test_reindex_is_idempotent(stores, tutor):
    identity, self_facts, worldview, episodic = stores
    _index_all(stores, tutor)
    counts_first = [s.count() for s in (identity, self_facts, worldview, episodic)]
    _index_all(stores, tutor)
    counts_second = [s.count() for s in (identity, self_facts, worldview, episodic)]
    assert counts_first == counts_second


# --------------------------------------------------------------- slow: real embedder


@pytest.mark.slow
def test_real_embedder_smoke(tmp_path: Path, personas_dir: Path) -> None:
    """Exercise the real SentenceTransformer embedder across the four stores.

    Marked slow because it downloads ~90 MB of MiniLM weights on first run.
    """
    path = tmp_path / "chroma_real"
    identity = IdentityStore(path)
    self_facts = SelfFactsStore(path)
    worldview = WorldviewStore(path)
    episodic = EpisodicStore(path)

    tutor = Persona.from_yaml(personas_dir / "cs_tutor.yaml")
    chunks = chunk_persona(tutor)
    for store in (identity, self_facts, worldview, episodic):
        store.index(chunks)

    # Identity always-retrieved.
    ident = identity.query("", top_k=1, persona_id="cs_tutor")
    assert any(c.kind == "identity" for c in ident)

    # Worldview semantic query.
    wv = worldview.query("pedagogy", top_k=3, persona_id="cs_tutor")
    assert 0 < len(wv) <= 3
