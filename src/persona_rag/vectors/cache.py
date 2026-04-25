"""Safetensors + meta.json cache for persona vectors.

Layout on disk, given ``vectors_cache_dir`` and ``persona_id``:

    <cache_dir>/<persona_id>.safetensors   # flat key=tensor dump
    <cache_dir>/<persona_id>.meta.json     # sidecar: layers, hashes, config

Tensor keys in the safetensors file (flat — safetensors does not nest):

    persona_vector_layer_{N}         — the mass-mean persona direction.
    in_persona_centroid_layer_{N}    — in-persona centroid.
    out_persona_centroid_layer_{N}   — out-persona centroid.
    in_states_layer_{N}              — stacked (n_pairs, hidden_dim) train activations.
    out_states_layer_{N}             — stacked (n_pairs, hidden_dim) train activations.

All tensors are float32 on CPU by the time they reach this writer. The
safetensors format uses little-endian + row-major, so save→load is
bit-identical by construction.

Cache invalidation: the meta.json stores a ``contrast_set_sha256`` and an
``extractor_code_sha256``. If either changes, the in-memory check inside
:func:`load_persona_vectors` can force a re-extraction (the check is the
caller's responsibility — this module just stores the hashes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from safetensors.torch import load_file, save_file

from persona_rag.vectors.extractor import PersonaVectors


@dataclass(frozen=True, slots=True)
class PersonaVectorCacheMeta:
    """Sidecar metadata round-tripped through ``<persona_id>.meta.json``.

    Mirrors :attr:`PersonaVectors.metadata` but typed so callers can reason
    about fields without ``dict.get`` gymnastics. Extra fields in a
    future-written meta.json are preserved via the ``extra`` dict.
    """

    persona_id: str
    layers: list[int]
    pool: str
    scope: str
    n_pairs: int
    seed: int
    hidden_dim: int
    contrast_set_sha256: str
    extractor_code_sha256: str
    backend_name: str
    backend_model_id: str
    num_layers_total: int | None
    timestamp_utc: str
    elapsed_seconds: float
    best_layer: int | None = None  # filled in by the validation script
    extra: dict[str, Any] = field(default_factory=dict)


def save_persona_vectors(
    vectors: PersonaVectors,
    cache_dir: Path | str,
    *,
    best_layer: int | None = None,
    store_full_activations: bool = True,
) -> tuple[Path, Path]:
    """Write safetensors + meta.json for a :class:`PersonaVectors`.

    Parameters
    ----------
    vectors:
        Extractor output.
    cache_dir:
        Directory where ``<persona_id>.safetensors`` + ``<persona_id>.meta.json``
        are written. Created if missing.
    best_layer:
        Optional — the global best layer picked by the validation script.
        Stored in the sidecar for downstream code to read without re-running
        the probe. Can be set later via :func:`update_best_layer`.
    store_full_activations:
        If ``True`` (default), the full ``(n_pairs, hidden_dim)`` train
        activations are written alongside the centroids and vectors. This
        is required for bit-exact round-trip validation and for re-running
        the probe off-line. Set to ``False`` only if disk budget is tight —
        the centroids + vectors alone are enough for inference-time drift.

    Returns
    -------
    (safetensors_path, meta_path) — absolute paths to the two written files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    st_path = cache_dir / f"{vectors.persona_id}.safetensors"
    meta_path = cache_dir / f"{vectors.persona_id}.meta.json"

    flat: dict[str, Any] = {}
    for layer in vectors.layers:
        flat[f"persona_vector_layer_{layer}"] = vectors.vectors[layer].detach().contiguous()
        flat[f"in_persona_centroid_layer_{layer}"] = (
            vectors.in_persona_centroid[layer].detach().contiguous()
        )
        flat[f"out_persona_centroid_layer_{layer}"] = (
            vectors.out_persona_centroid[layer].detach().contiguous()
        )
        if store_full_activations:
            if layer in vectors.in_states:
                flat[f"in_states_layer_{layer}"] = vectors.in_states[layer].detach().contiguous()
            if layer in vectors.out_states:
                flat[f"out_states_layer_{layer}"] = vectors.out_states[layer].detach().contiguous()

    save_file(flat, str(st_path))

    md = vectors.metadata or {}
    meta = PersonaVectorCacheMeta(
        persona_id=vectors.persona_id,
        layers=list(vectors.layers),
        pool=str(md.get("pool", "last")),
        scope=str(md.get("scope", "prompt")),
        n_pairs=int(md.get("n_pairs", 0)),
        seed=int(md.get("seed", 0)),
        hidden_dim=int(md.get("hidden_dim", 0)),
        contrast_set_sha256=str(md.get("contrast_set_sha256", "")),
        extractor_code_sha256=str(md.get("extractor_code_sha256", "")),
        backend_name=str(md.get("backend_name", "")),
        backend_model_id=str(md.get("backend_model_id", "")),
        num_layers_total=md.get("num_layers_total"),
        timestamp_utc=str(md.get("timestamp_utc", "")),
        elapsed_seconds=float(md.get("elapsed_seconds", 0.0)),
        best_layer=best_layer,
        extra={k: v for k, v in md.items() if k not in _KNOWN_META_KEYS},
    )
    meta_path.write_text(
        json.dumps(asdict(meta), indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "cached persona vectors for {!r}: {} + {} "
        "({} layers, {} pairs per side, store_full_activations={})",
        vectors.persona_id,
        st_path.name,
        meta_path.name,
        len(vectors.layers),
        meta.n_pairs,
        store_full_activations,
    )
    return st_path, meta_path


def load_persona_vectors(cache_dir: Path | str, persona_id: str) -> PersonaVectors:
    """Reload :class:`PersonaVectors` from a previously-written cache.

    Raises ``FileNotFoundError`` if either file is missing. Does NOT validate
    the ``contrast_set_sha256`` / ``extractor_code_sha256`` — the caller
    compares those against the current hashes and decides whether to re-
    extract.

    The returned tensors are bit-identical to what was written, by the
    safetensors format contract. ``tests/test_vectors_cache.py`` asserts
    this directly.
    """
    cache_dir = Path(cache_dir)
    st_path = cache_dir / f"{persona_id}.safetensors"
    meta_path = cache_dir / f"{persona_id}.meta.json"
    if not st_path.exists():
        raise FileNotFoundError(f"persona-vector cache missing: {st_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"persona-vector meta missing: {meta_path}")

    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    layers: list[int] = list(meta_raw.get("layers", []))

    flat = load_file(str(st_path), device="cpu")

    vectors: dict[int, Any] = {}
    in_centroid: dict[int, Any] = {}
    out_centroid: dict[int, Any] = {}
    in_states: dict[int, Any] = {}
    out_states: dict[int, Any] = {}
    for layer in layers:
        vectors[layer] = flat[f"persona_vector_layer_{layer}"]
        in_centroid[layer] = flat[f"in_persona_centroid_layer_{layer}"]
        out_centroid[layer] = flat[f"out_persona_centroid_layer_{layer}"]
        k_in = f"in_states_layer_{layer}"
        k_out = f"out_states_layer_{layer}"
        if k_in in flat:
            in_states[layer] = flat[k_in]
        if k_out in flat:
            out_states[layer] = flat[k_out]

    return PersonaVectors(
        persona_id=persona_id,
        vectors=vectors,
        in_persona_centroid=in_centroid,
        out_persona_centroid=out_centroid,
        in_states=in_states,
        out_states=out_states,
        metadata=meta_raw,
    )


def update_best_layer(cache_dir: Path | str, persona_id: str, best_layer: int) -> Path:
    """Patch ``best_layer`` into an already-written meta.json sidecar.

    Used by the validation script after the probe picks the global best
    layer — avoids re-saving the heavy safetensors file just to record one
    integer. The patch is atomic (write to ``.tmp`` + rename).
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / f"{persona_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"persona-vector meta missing: {meta_path}")
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_raw["best_layer"] = int(best_layer)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(meta_raw, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(meta_path)
    return meta_path


# Keys the structured meta owns — anything else in extractor.metadata is
# dumped into meta.extra so forward-compat writers don't lose fields.
_KNOWN_META_KEYS: frozenset[str] = frozenset(
    {
        "persona_id",
        "layers",
        "pool",
        "scope",
        "n_pairs",
        "seed",
        "hidden_dim",
        "contrast_set_sha256",
        "extractor_code_sha256",
        "backend_name",
        "backend_model_id",
        "num_layers_total",
        "timestamp_utc",
        "elapsed_seconds",
    }
)
