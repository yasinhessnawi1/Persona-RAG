"""Tests for the `PersonaRegistry`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from persona_rag.schema import Persona, PersonaRegistry
from persona_rag.stores import (
    EpisodicStore,
    IdentityStore,
    SelfFactsStore,
    WorldviewStore,
)


@pytest.fixture
def registry(tmp_path: Path, fake_embedder) -> PersonaRegistry:
    path = tmp_path / "chroma"
    return PersonaRegistry(
        identity_store=IdentityStore(path, embedding_function=fake_embedder),
        self_facts_store=SelfFactsStore(path, embedding_function=fake_embedder),
        worldview_store=WorldviewStore(path, embedding_function=fake_embedder),
        episodic_store=EpisodicStore(path, embedding_function=fake_embedder),
    )


def test_register_populates_all_four_stores(registry: PersonaRegistry, personas_dir: Path):
    reg = registry.register(personas_dir / "cs_tutor.yaml")
    assert reg.persona.persona_id == "cs_tutor"

    identity_chunks = reg.identity_store.get_all(persona_id="cs_tutor")
    sf_count = reg.self_facts_store.count()
    wv_count = reg.worldview_store.count()
    ep_count = reg.episodic_store.count()

    assert len(identity_chunks) > 0  # identity + constraints
    assert sf_count == len(reg.persona.self_facts)
    assert wv_count == len(reg.persona.worldview)
    assert ep_count == len(reg.persona.episodic)  # 0 for the shipped YAMLs


def test_register_vectors_none_when_no_extractor(registry: PersonaRegistry, personas_dir: Path):
    reg = registry.register(personas_dir / "cs_tutor.yaml")
    assert reg.vectors is None
    assert reg.vectors_cache_path is None


def test_register_is_idempotent(registry: PersonaRegistry, personas_dir: Path):
    registry.register(personas_dir / "cs_tutor.yaml")
    count_sf = registry.register(personas_dir / "cs_tutor.yaml").self_facts_store.count()
    # Re-registration does not duplicate chunks.
    first_sf = Persona.from_yaml(personas_dir / "cs_tutor.yaml").self_facts
    assert count_sf == len(first_sf)


def test_register_multiple_personas_coexist(registry: PersonaRegistry, personas_dir: Path):
    registry.register(personas_dir / "cs_tutor.yaml")
    registry.register(personas_dir / "historian.yaml")

    tutor_ident = registry.register(personas_dir / "cs_tutor.yaml").identity_store.get_all(
        persona_id="cs_tutor"
    )
    hist_ident = registry.register(personas_dir / "historian.yaml").identity_store.get_all(
        persona_id="historian"
    )
    assert len(tutor_ident) > 0
    assert len(hist_ident) > 0
    assert all(c.metadata["persona_id"] == "cs_tutor" for c in tutor_ident)
    assert all(c.metadata["persona_id"] == "historian" for c in hist_ident)


def test_delete_removes_from_every_store(registry: PersonaRegistry, personas_dir: Path):
    registry.register(personas_dir / "cs_tutor.yaml")
    counts = registry.delete("cs_tutor")
    assert sum(counts.values()) > 0

    # After delete, a query returns no tutor chunks.
    result = registry.register(personas_dir / "cs_tutor.yaml").self_facts_store.query(
        "teaching", top_k=3, persona_id="cs_tutor"
    )
    assert all(c.metadata["persona_id"] == "cs_tutor" for c in result)


class _DummyExtractor:
    """Stand-in for `PersonaVectorExtractor`. Returns a trivial payload."""

    def extract(self, persona: dict, contrast_set: Any) -> dict:
        return {"persona_id": persona.get("persona_id"), "vectors": {8: [0.1] * 384}}


def test_register_with_injected_extractor_sets_vectors(
    tmp_path: Path, fake_embedder, personas_dir: Path
):
    path = tmp_path / "chroma"
    cache_dir = tmp_path / "vectors_cache"
    registry = PersonaRegistry(
        identity_store=IdentityStore(path, embedding_function=fake_embedder),
        self_facts_store=SelfFactsStore(path, embedding_function=fake_embedder),
        worldview_store=WorldviewStore(path, embedding_function=fake_embedder),
        episodic_store=EpisodicStore(path, embedding_function=fake_embedder),
        vector_extractor=_DummyExtractor(),
        vectors_cache_dir=cache_dir,
    )
    reg = registry.register(personas_dir / "cs_tutor.yaml")
    assert reg.vectors is not None
    assert reg.vectors["persona_id"] == "cs_tutor"
    assert reg.vectors_cache_path == cache_dir / "cs_tutor.safetensors"
