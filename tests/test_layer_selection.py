"""Tests for pick_global_best_layer."""

from __future__ import annotations

import pytest

from persona_rag.vectors.layer_selection import pick_global_best_layer


def test_picks_mean_auroc_maximizer() -> None:
    per_persona = {
        "p1": {8: 0.70, 12: 0.85, 16: 0.80, 20: 0.75},
        "p2": {8: 0.72, 12: 0.88, 16: 0.78, 20: 0.74},
        "p3": {8: 0.68, 12: 0.84, 16: 0.82, 20: 0.76},
    }
    res = pick_global_best_layer(per_persona)
    assert res.best_layer == 12
    assert res.mean_auroc == pytest.approx((0.85 + 0.88 + 0.84) / 3)


def test_per_persona_best_reported() -> None:
    per_persona = {
        "p1": {8: 0.70, 12: 0.85, 16: 0.80},
        "p2": {8: 0.90, 12: 0.60, 16: 0.70},
    }
    res = pick_global_best_layer(per_persona)
    assert res.per_persona_best == {"p1": 12, "p2": 8}


def test_gap_warning_fires_on_wide_disagreement() -> None:
    per_persona = {
        "p1": {8: 0.90, 12: 0.60, 16: 0.55, 20: 0.50},  # best=8
        "p2": {8: 0.50, 12: 0.60, 16: 0.70, 20: 0.95},  # best=20
    }
    res = pick_global_best_layer(per_persona, diagnostic_gap_layers=4)
    assert res.gap_warning is not None
    assert "span" in res.gap_warning


def test_no_gap_warning_on_tight_agreement() -> None:
    per_persona = {
        "p1": {8: 0.70, 12: 0.90, 16: 0.80},
        "p2": {8: 0.75, 12: 0.88, 16: 0.82},
    }
    res = pick_global_best_layer(per_persona)
    assert res.gap_warning is None


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_global_best_layer({})


def test_layer_set_mismatch_raises() -> None:
    per_persona = {
        "p1": {8: 0.9, 12: 0.8},
        "p2": {8: 0.85, 16: 0.7},  # different layer set
    }
    with pytest.raises(ValueError, match="differ"):
        pick_global_best_layer(per_persona)


def test_tie_picks_smallest_index() -> None:
    """Deterministic tie-break: lowest-index layer wins."""
    per_persona = {
        "p1": {8: 0.80, 12: 0.80, 16: 0.80},
    }
    res = pick_global_best_layer(per_persona)
    assert res.best_layer == 8
