"""Tests for the persona schema (typed memory models + YAML IO)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from persona_rag.schema import (
    EPISTEMIC_TAGS,
    EpisodicEntry,
    Persona,
    PersonaIdentity,
    SelfFact,
    WorldviewClaim,
)

PERSONAS_DIR = Path(__file__).resolve().parents[1] / "personas"


def _minimal_valid_dict() -> dict:
    return {
        "identity": {
            "name": "Test User",
            "role": "Test role",
            "background": "Test background.",
            "constraints": ["Do not do X."],
        },
        "self_facts": [{"fact": "I am a self-fact."}],
        "worldview": [
            {
                "claim": "Testing is good.",
                "domain": "software_engineering",
            }
        ],
    }


# ----------------------------------------------------------------- positive cases


def test_minimal_valid_persona_parses():
    p = Persona.model_validate(_minimal_valid_dict())
    assert p.identity.name == "Test User"
    assert p.identity.constraints == ["Do not do X."]
    assert p.self_facts[0].fact == "I am a self-fact."
    assert p.self_facts[0].confidence == 1.0
    assert p.self_facts[0].epistemic == "fact"
    assert p.worldview[0].epistemic == "belief"  # default
    assert p.worldview[0].valid_time == "always"
    assert p.worldview[0].confidence == 0.85  # default
    assert p.episodic == []


def test_all_three_example_personas_load_and_validate():
    for stem in ("cs_tutor", "historian", "climate_scientist"):
        path = PERSONAS_DIR / f"{stem}.yaml"
        assert path.exists(), f"example persona missing: {path}"
        persona = Persona.from_yaml(path)
        assert persona.persona_id == stem
        assert len(persona.self_facts) >= 6
        assert len(persona.worldview) >= 5
        assert len(persona.identity.constraints) >= 3


def test_historian_uses_valid_time_meaningfully():
    persona = Persona.from_yaml(PERSONAS_DIR / "historian.yaml")
    valid_times = {w.valid_time for w in persona.worldview}
    # At least one claim should carry a real date range, not just "always".
    assert any(v != "always" for v in valid_times)


def test_from_yaml_derives_persona_id_from_filename(tmp_path: Path):
    path = tmp_path / "custom_name.yaml"
    path.write_text(yaml.safe_dump(_minimal_valid_dict()), encoding="utf-8")
    persona = Persona.from_yaml(path)
    assert persona.persona_id == "custom_name"


def test_from_yaml_explicit_persona_id_wins(tmp_path: Path):
    data = _minimal_valid_dict()
    data["persona_id"] = "explicit_id"
    path = tmp_path / "irrelevant_filename.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    persona = Persona.from_yaml(path)
    assert persona.persona_id == "explicit_id"


def test_from_yaml_unwraps_top_level_persona_key(tmp_path: Path):
    path = tmp_path / "wrapped.yaml"
    path.write_text(yaml.safe_dump({"persona": _minimal_valid_dict()}), encoding="utf-8")
    persona = Persona.from_yaml(path)
    assert persona.persona_id == "wrapped"


def test_yaml_round_trip_preserves_object(tmp_path: Path):
    original = Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")
    out = tmp_path / "roundtrip.yaml"
    original.to_yaml(out)
    reloaded = Persona.from_yaml(out)
    assert reloaded == original


# ----------------------------------------------------------------- negative cases


def test_missing_identity_rejected():
    data = _minimal_valid_dict()
    del data["identity"]
    with pytest.raises(ValidationError):
        Persona.model_validate(data)


def test_unknown_top_level_key_rejected():
    data = _minimal_valid_dict()
    data["emotion"] = "happy"
    with pytest.raises(ValidationError):
        Persona.model_validate(data)


def test_unknown_key_in_identity_rejected():
    data = _minimal_valid_dict()
    data["identity"]["age"] = 42
    with pytest.raises(ValidationError):
        Persona.model_validate(data)


def test_self_fact_requires_fact_field():
    data = _minimal_valid_dict()
    data["self_facts"] = [{"epistemic": "fact"}]  # missing `fact`
    with pytest.raises(ValidationError):
        Persona.model_validate(data)


def test_self_fact_rejects_non_fact_epistemic():
    # Literal["fact"] — assigning any other value must raise.
    with pytest.raises(ValidationError):
        SelfFact.model_validate({"fact": "foo", "epistemic": "belief"})


def test_self_fact_confidence_bounds():
    with pytest.raises(ValidationError):
        SelfFact.model_validate({"fact": "foo", "confidence": 1.1})
    with pytest.raises(ValidationError):
        SelfFact.model_validate({"fact": "foo", "confidence": -0.1})


def test_too_long_self_fact_rejected():
    data = _minimal_valid_dict()
    data["self_facts"] = [{"fact": "x" * 501}]
    with pytest.raises(ValidationError):
        Persona.model_validate(data)


def test_worldview_domain_pattern_rejects_hyphen():
    with pytest.raises(ValidationError):
        WorldviewClaim.model_validate({"claim": "c", "domain": "with-hyphen"})


def test_worldview_domain_pattern_rejects_uppercase():
    with pytest.raises(ValidationError):
        WorldviewClaim.model_validate({"claim": "c", "domain": "NotSnakeCase"})


def test_worldview_invalid_epistemic_rejected():
    with pytest.raises(ValidationError):
        WorldviewClaim.model_validate({"claim": "c", "domain": "d", "epistemic": "gossip"})


@pytest.mark.parametrize(
    "valid_time",
    ["always", "1492", "1500-1800", "1970-"],
)
def test_worldview_accepts_well_formed_valid_time(valid_time: str):
    wv = WorldviewClaim.model_validate({"claim": "c", "domain": "d", "valid_time": valid_time})
    assert wv.valid_time == valid_time


@pytest.mark.parametrize(
    "bad_valid_time",
    ["", "1800-1500", "today", "15", "1800-2100-"],
)
def test_worldview_rejects_bad_valid_time(bad_valid_time: str):
    with pytest.raises(ValidationError):
        WorldviewClaim.model_validate({"claim": "c", "domain": "d", "valid_time": bad_valid_time})


def test_identity_empty_constraint_rejected():
    with pytest.raises(ValidationError):
        PersonaIdentity.model_validate(
            {"name": "n", "role": "r", "background": "b", "constraints": [""]}
        )


def test_identity_too_long_constraint_rejected():
    with pytest.raises(ValidationError):
        PersonaIdentity.model_validate(
            {
                "name": "n",
                "role": "r",
                "background": "b",
                "constraints": ["x" * 201],
            }
        )


def test_non_mapping_yaml_raises():
    with pytest.raises(ValueError):
        Persona.model_validate("not a dict")  # type: ignore[arg-type]


# ----------------------------------------------------------------- episodic


def test_episodic_entry_naive_datetime_coerced_to_utc():
    now = datetime.now()
    e = EpisodicEntry.model_validate({"text": "t", "timestamp": now, "turn_id": 0, "decay_t0": now})
    assert e.timestamp.tzinfo is not None
    assert e.timestamp.tzinfo == UTC


def test_episodic_entry_requires_non_negative_turn_id():
    with pytest.raises(ValidationError):
        EpisodicEntry.model_validate(
            {
                "text": "t",
                "timestamp": datetime.now(UTC),
                "turn_id": -1,
                "decay_t0": datetime.now(UTC),
            }
        )


def test_episodic_chunks_survive_yaml_round_trip(tmp_path: Path):
    data = _minimal_valid_dict()
    ts = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    data["episodic"] = [
        {"text": "User mentioned they use Rust.", "timestamp": ts, "turn_id": 3, "decay_t0": ts}
    ]
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
    persona = Persona.from_yaml(path)
    out = tmp_path / "out.yaml"
    persona.to_yaml(out)
    reloaded = Persona.from_yaml(out)
    assert reloaded == persona


def test_epistemic_tags_constant_is_the_full_vocabulary():
    assert set(EPISTEMIC_TAGS) == {"fact", "belief", "hypothesis", "contested"}
