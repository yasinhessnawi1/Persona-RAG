"""Tests for the v0.3 chunker (routes atomic items into kind-tagged chunks)."""

from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.schema import Persona, chunk_persona, chunks_by_kind


def _persona() -> Persona:
    ts = datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC)
    return Persona.model_validate(
        {
            "persona_id": "unit_test",
            "identity": {
                "name": "Alice",
                "role": "Tester",
                "background": "Writes tests all day.",
                "constraints": ["do not X", "do not Y"],
            },
            "self_facts": [
                {"fact": "fact one", "confidence": 1.0},
                {"fact": "fact two"},
                {"fact": "fact three"},
            ],
            "worldview": [
                {
                    "claim": "stance one",
                    "domain": "topic_a",
                    "epistemic": "belief",
                    "valid_time": "2000-",
                    "confidence": 0.75,
                },
                {
                    "claim": "stance two",
                    "domain": "topic_b",
                    "epistemic": "contested",
                },
            ],
            "episodic": [
                {"text": "ep one", "timestamp": ts, "turn_id": 0, "decay_t0": ts},
                {"text": "ep two", "timestamp": ts, "turn_id": 1, "decay_t0": ts},
            ],
        }
    )


def test_chunk_count_matches_items():
    chunks = chunk_persona(_persona())
    # 1 identity + 2 constraints + 3 self_facts + 2 worldview + 2 episodic = 10
    assert len(chunks) == 10


def test_chunk_kinds_are_tagged_correctly():
    chunks = chunk_persona(_persona())
    kinds = [c.kind for c in chunks]
    assert kinds.count("identity") == 1
    assert kinds.count("constraint") == 2
    assert kinds.count("self_fact") == 3
    assert kinds.count("worldview") == 2
    assert kinds.count("episodic") == 2


def test_chunks_by_kind_groups_all_kinds():
    grouped = chunks_by_kind(chunk_persona(_persona()))
    assert set(grouped.keys()) == {"identity", "constraint", "self_fact", "worldview", "episodic"}
    assert len(grouped["identity"]) == 1
    assert len(grouped["constraint"]) == 2
    assert len(grouped["self_fact"]) == 3
    assert len(grouped["worldview"]) == 2
    assert len(grouped["episodic"]) == 2


def test_self_fact_metadata_carries_epistemic_and_confidence():
    chunks = chunk_persona(_persona())
    sf = next(c for c in chunks if c.kind == "self_fact" and c.text == "fact one")
    assert sf.metadata["epistemic"] == "fact"
    assert float(sf.metadata["confidence"]) == 1.0


def test_worldview_metadata_carries_all_typed_fields():
    chunks = chunk_persona(_persona())
    wv = next(c for c in chunks if c.kind == "worldview" and c.text == "stance one")
    assert wv.metadata["domain"] == "topic_a"
    assert wv.metadata["epistemic"] == "belief"
    assert wv.metadata["valid_time"] == "2000-"
    assert float(wv.metadata["confidence"]) == 0.75


def test_episodic_metadata_carries_iso_timestamps():
    chunks = chunk_persona(_persona())
    episodic = [c for c in chunks if c.kind == "episodic"]
    assert len(episodic) == 2
    for c in episodic:
        assert "timestamp" in c.metadata
        assert "decay_t0" in c.metadata
        # Round-trip the stored string — must parse as ISO-8601.
        datetime.fromisoformat(c.metadata["timestamp"])
        datetime.fromisoformat(c.metadata["decay_t0"])


def test_identity_chunk_combines_name_role_background():
    chunks = chunk_persona(_persona())
    identity = next(c for c in chunks if c.kind == "identity")
    assert "Alice" in identity.text
    assert "Tester" in identity.text
    assert "Writes tests all day." in identity.text


def test_constraint_chunks_split_one_per_constraint():
    chunks = chunk_persona(_persona())
    constraints = [c for c in chunks if c.kind == "constraint"]
    texts = {c.text for c in constraints}
    assert texts == {"do not X", "do not Y"}


def test_chunk_ids_prefixed_with_persona_id_and_unique():
    chunks = chunk_persona(_persona())
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(c.id.startswith("unit_test:") for c in chunks)


def test_persona_id_required_for_chunking():
    p = Persona.model_validate(
        {
            "identity": {"name": "n", "role": "r", "background": "b", "constraints": []},
            "self_facts": [{"fact": "f"}],
            "worldview": [],
        }
    )
    import pytest

    with pytest.raises(ValueError):
        chunk_persona(p)


def test_empty_optional_lists_produce_only_identity_chunk():
    p = Persona.model_validate(
        {
            "persona_id": "empty",
            "identity": {"name": "n", "role": "r", "background": "b", "constraints": []},
            "self_facts": [],
            "worldview": [],
        }
    )
    chunks = chunk_persona(p)
    assert len(chunks) == 1
    assert chunks[0].kind == "identity"


def test_chunk_metadata_values_are_strings():
    for c in chunk_persona(_persona()):
        for value in c.metadata.values():
            assert isinstance(value, str)
