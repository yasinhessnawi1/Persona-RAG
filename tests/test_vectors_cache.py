"""Tests for the safetensors + meta.json cache (bit-exact round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from persona_rag.vectors.cache import (
    load_persona_vectors,
    save_persona_vectors,
    update_best_layer,
)
from persona_rag.vectors.extractor import PersonaVectors


def _make_pv(persona_id: str = "test_persona") -> PersonaVectors:
    layers = [0, 1, 2]
    hidden_dim = 8
    g = torch.Generator().manual_seed(7)
    vectors = {layer: torch.randn(hidden_dim, generator=g) for layer in layers}
    in_c = {layer: torch.randn(hidden_dim, generator=g) for layer in layers}
    out_c = {layer: torch.randn(hidden_dim, generator=g) for layer in layers}
    in_s = {layer: torch.randn(10, hidden_dim, generator=g) for layer in layers}
    out_s = {layer: torch.randn(10, hidden_dim, generator=g) for layer in layers}
    metadata = {
        "persona_id": persona_id,
        "backend_name": "fake",
        "backend_model_id": "fake/0",
        "hidden_dim": hidden_dim,
        "num_layers_total": 4,
        "layers": layers,
        "pool": "last",
        "scope": "prompt",
        "n_pairs": 10,
        "seed": 42,
        "contrast_set_sha256": "a" * 64,
        "extractor_code_sha256": "b" * 64,
        "elapsed_seconds": 1.23,
        "timestamp_utc": "2026-04-24T00:00:00Z",
    }
    return PersonaVectors(
        persona_id=persona_id,
        vectors=vectors,
        in_persona_centroid=in_c,
        out_persona_centroid=out_c,
        in_states=in_s,
        out_states=out_s,
        metadata=metadata,
    )


def test_a19_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Saves then loads; every tensor must be bit-identical."""
    pv = _make_pv()
    save_persona_vectors(pv, tmp_path)
    loaded = load_persona_vectors(tmp_path, pv.persona_id)

    for layer in pv.layers:
        assert torch.equal(pv.vectors[layer], loaded.vectors[layer])
        assert torch.equal(pv.in_persona_centroid[layer], loaded.in_persona_centroid[layer])
        assert torch.equal(pv.out_persona_centroid[layer], loaded.out_persona_centroid[layer])
        assert torch.equal(pv.in_states[layer], loaded.in_states[layer])
        assert torch.equal(pv.out_states[layer], loaded.out_states[layer])


def test_meta_roundtrip_preserves_keys(tmp_path: Path) -> None:
    pv = _make_pv()
    _, meta_path = save_persona_vectors(pv, tmp_path, best_layer=1)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["persona_id"] == pv.persona_id
    assert meta["layers"] == pv.layers
    assert meta["pool"] == "last"
    assert meta["scope"] == "prompt"
    assert meta["n_pairs"] == 10
    assert meta["seed"] == 42
    assert meta["hidden_dim"] == 8
    assert meta["contrast_set_sha256"] == "a" * 64
    assert meta["extractor_code_sha256"] == "b" * 64
    assert meta["backend_name"] == "fake"
    assert meta["best_layer"] == 1
    assert meta["timestamp_utc"] == "2026-04-24T00:00:00Z"


def test_load_raises_when_safetensors_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"safetensors"):
        load_persona_vectors(tmp_path, "no_such_persona")


def test_update_best_layer_patches_meta_atomically(tmp_path: Path) -> None:
    pv = _make_pv()
    save_persona_vectors(pv, tmp_path)
    update_best_layer(tmp_path, pv.persona_id, best_layer=2)
    meta = json.loads((tmp_path / f"{pv.persona_id}.meta.json").read_text(encoding="utf-8"))
    assert meta["best_layer"] == 2


def test_update_best_layer_errors_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        update_best_layer(tmp_path, "ghost", 0)


def test_save_without_full_activations(tmp_path: Path) -> None:
    """store_full_activations=False → no in_states/out_states in the file."""
    pv = _make_pv()
    save_persona_vectors(pv, tmp_path, store_full_activations=False)
    loaded = load_persona_vectors(tmp_path, pv.persona_id)
    # Centroids + vectors still round-trip; activations are absent.
    assert loaded.in_states == {}
    assert loaded.out_states == {}
    for layer in pv.layers:
        assert torch.equal(pv.vectors[layer], loaded.vectors[layer])


def test_multiple_personas_coexist(tmp_path: Path) -> None:
    a = _make_pv("persona_a")
    b = _make_pv("persona_b")
    save_persona_vectors(a, tmp_path)
    save_persona_vectors(b, tmp_path)
    la = load_persona_vectors(tmp_path, "persona_a")
    lb = load_persona_vectors(tmp_path, "persona_b")
    assert la.persona_id == "persona_a"
    assert lb.persona_id == "persona_b"


# ----- Cache-invalidation contract: changing a contrast prompt → different hash --


def test_changing_contrast_prompt_changes_cached_sha256(tmp_path: Path) -> None:
    """If a single contrast prompt changes, the meta.json's
    ``contrast_set_sha256`` must change so callers (PersonaRegistry) detect
    a stale cache and force re-extraction.

    Without this guarantee, downstream code would silently reuse persona
    vectors built from a different contrast set than the current source
    defines.
    """
    from persona_rag.vectors.contrast_prompts import ContrastSet

    pv_baseline = _make_pv("p")
    cs_baseline = ContrastSet(
        in_persona=("in-prompt-1", "in-prompt-2"),
        out_persona=("out-prompt-1", "out-prompt-2"),
        topic_aligned=True,
        seed=0,
    )
    pv_baseline.metadata["contrast_set_sha256"] = cs_baseline.sha256()
    save_persona_vectors(pv_baseline, tmp_path)
    meta_baseline = json.loads((tmp_path / "p.meta.json").read_text(encoding="utf-8"))

    # Mutate exactly one in-persona prompt, re-hash, save again. The cached
    # contrast_set_sha256 must change. Note: we re-save as if a fresh
    # extraction had run with the modified contrast set — the cache writer is
    # the integration point that records the hash.
    cs_mutated = ContrastSet(
        in_persona=("in-prompt-1-MUTATED", "in-prompt-2"),
        out_persona=("out-prompt-1", "out-prompt-2"),
        topic_aligned=True,
        seed=0,
    )
    pv_mutated = _make_pv("p")
    pv_mutated.metadata["contrast_set_sha256"] = cs_mutated.sha256()
    save_persona_vectors(pv_mutated, tmp_path)
    meta_mutated = json.loads((tmp_path / "p.meta.json").read_text(encoding="utf-8"))

    assert cs_baseline.sha256() != cs_mutated.sha256(), (
        "ContrastSet.sha256() must reflect prompt content"
    )
    assert meta_baseline["contrast_set_sha256"] != meta_mutated["contrast_set_sha256"], (
        "cache writer must persist the new contrast_set_sha256 — without this, "
        "downstream consumers would silently reuse stale persona vectors"
    )


def test_changing_contrast_prompt_does_not_affect_round_trip_otherwise(
    tmp_path: Path,
) -> None:
    """Sanity: only the recorded hash changes — the rest of meta.json and
    the safetensors payload are unaffected by a contrast-prompt edit."""
    from persona_rag.vectors.contrast_prompts import ContrastSet

    pv = _make_pv("p")
    cs1 = ContrastSet(in_persona=("a",) * 2, out_persona=("b",) * 2)
    pv.metadata["contrast_set_sha256"] = cs1.sha256()
    save_persona_vectors(pv, tmp_path)
    loaded_first = load_persona_vectors(tmp_path, "p")

    cs2 = ContrastSet(in_persona=("a-edited",) * 2, out_persona=("b",) * 2)
    pv.metadata["contrast_set_sha256"] = cs2.sha256()
    save_persona_vectors(pv, tmp_path)
    loaded_second = load_persona_vectors(tmp_path, "p")

    # Tensors round-trip identically — only the meta hash differs.
    for layer in pv.layers:
        assert torch.equal(loaded_first.vectors[layer], loaded_second.vectors[layer])
    assert (
        loaded_first.metadata["contrast_set_sha256"]
        != loaded_second.metadata["contrast_set_sha256"]
    )
