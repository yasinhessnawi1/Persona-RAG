"""Tests for DriftSignal — sign convention is load-bearing."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from persona_rag.vectors.drift import DriftSignal
from persona_rag.vectors.extractor import PersonaVectors


def _make_pv_for_drift() -> PersonaVectors:
    """Two centroids pointing opposite directions on the first axis."""
    hidden_dim = 8
    in_c = torch.zeros(hidden_dim, dtype=torch.float32)
    in_c[0] = 1.0
    out_c = torch.zeros(hidden_dim, dtype=torch.float32)
    out_c[0] = -1.0
    return PersonaVectors(
        persona_id="p",
        vectors={0: in_c - out_c},
        in_persona_centroid={0: in_c},
        out_persona_centroid={0: out_c},
    )


def test_drift_signal_at_in_centroid_is_plus_one() -> None:
    """in_centroid → +1.000 — THE sign-convention gate."""
    pv = _make_pv_for_drift()
    drift = DriftSignal.from_persona_vectors(pv, 0)
    assert drift.compute(pv.in_persona_centroid[0]) == pytest.approx(1.0, abs=1e-5)


def test_drift_signal_at_out_centroid_is_minus_one() -> None:
    """out_centroid → -1.000 — THE sign-convention gate."""
    pv = _make_pv_for_drift()
    drift = DriftSignal.from_persona_vectors(pv, 0)
    assert drift.compute(pv.out_persona_centroid[0]) == pytest.approx(-1.0, abs=1e-5)


def test_drift_is_zero_at_orthogonal_state() -> None:
    """A hidden state orthogonal to the persona direction should read near 0."""
    pv = _make_pv_for_drift()
    orth = torch.zeros(8, dtype=torch.float32)
    orth[1] = 1.0  # orthogonal to axis 0 along which the centroids live
    drift = DriftSignal.from_persona_vectors(pv, 0)
    assert drift.compute(orth) == pytest.approx(0.0, abs=1e-5)


def test_drift_in_bounds() -> None:
    """Drift always ∈ [-1, 1]."""
    pv = _make_pv_for_drift()
    drift = DriftSignal.from_persona_vectors(pv, 0)
    g = torch.Generator().manual_seed(123)
    for _ in range(20):
        h = torch.randn(8, generator=g, dtype=torch.float32)
        v = drift.compute(h)
        assert -1.0 <= v <= 1.0


def test_drift_monotonic_along_direction() -> None:
    """Moving from out_centroid toward in_centroid increases drift monotonically."""
    pv = _make_pv_for_drift()
    drift = DriftSignal.from_persona_vectors(pv, 0)
    in_c = pv.in_persona_centroid[0]
    out_c = pv.out_persona_centroid[0]
    values = [drift.compute(out_c + t * (in_c - out_c)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values[0] == pytest.approx(-1.0, abs=1e-5)
    assert values[-1] == pytest.approx(1.0, abs=1e-5)
    for a, b in pairwise(values):
        assert a <= b + 1e-6


def test_from_persona_vectors_errors_on_missing_layer() -> None:
    pv = _make_pv_for_drift()
    with pytest.raises(KeyError, match="layer 99"):
        DriftSignal.from_persona_vectors(pv, 99)


def test_degenerate_centroids_give_zero_drift() -> None:
    """If c_in == c_out, drift signal is undefined — guard returns 0.0."""
    in_c = torch.ones(8, dtype=torch.float32)
    out_c = in_c.clone()
    pv = PersonaVectors(
        persona_id="p",
        vectors={0: in_c - out_c},
        in_persona_centroid={0: in_c},
        out_persona_centroid={0: out_c},
    )
    drift = DriftSignal.from_persona_vectors(pv, 0)
    assert drift.compute(torch.randn(8)) == 0.0
