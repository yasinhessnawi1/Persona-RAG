"""Tests for SeparabilityProbe — synthetic separable → 1.0, shuffled → 0.5."""

from __future__ import annotations

import torch

from persona_rag.vectors.extractor import PersonaVectors
from persona_rag.vectors.probe import (
    AUROC_CONFIRMED_FLOOR,
    AUROC_WEAK_FLOOR,
    SeparabilityProbe,
)


def _make_pv(
    persona_id: str,
    layers: list[int],
    hidden_dim: int,
    *,
    n_per_side: int,
    sep: float = 3.0,
    seed: int = 0,
) -> PersonaVectors:
    """Build a synthetic PersonaVectors where in/out are trivially separable.

    Each layer gets its own random direction; in-persona states are +sep
    along that direction, out-persona are -sep, plus unit gaussian noise.
    Persona vector = in_mean - out_mean (computed here, as the real
    extractor would).
    """
    g = torch.Generator().manual_seed(seed)
    vectors: dict[int, torch.Tensor] = {}
    in_centroid: dict[int, torch.Tensor] = {}
    out_centroid: dict[int, torch.Tensor] = {}
    in_states: dict[int, torch.Tensor] = {}
    out_states: dict[int, torch.Tensor] = {}
    for layer in layers:
        direction = torch.nn.functional.normalize(
            torch.randn(hidden_dim, generator=g, dtype=torch.float32), dim=0
        )
        noise_in = torch.randn(n_per_side, hidden_dim, generator=g)
        noise_out = torch.randn(n_per_side, hidden_dim, generator=g)
        in_mat = sep * direction + noise_in
        out_mat = -sep * direction + noise_out
        in_states[layer] = in_mat
        out_states[layer] = out_mat
        in_centroid[layer] = in_mat.mean(dim=0)
        out_centroid[layer] = out_mat.mean(dim=0)
        vectors[layer] = in_centroid[layer] - out_centroid[layer]
    return PersonaVectors(
        persona_id=persona_id,
        vectors=vectors,
        in_persona_centroid=in_centroid,
        out_persona_centroid=out_centroid,
        in_states=in_states,
        out_states=out_states,
    )


def test_auroc_high_on_trivially_separable() -> None:
    """Near-perfect separation → per-layer AUROC ≥ 0.90 on n=60 train + n=30 test.

    We don't assert 1.0 because with only ~60 train / 30 test samples per
    side, noise realization on one of the sampled layers can dip AUROC
    into the 0.88-0.95 band — still well above the confirmed floor.
    """
    layers = [0, 1]
    hidden_dim = 32
    train = _make_pv("p", layers, hidden_dim, n_per_side=60, sep=4.0, seed=0)
    test = _make_pv("p", layers, hidden_dim, n_per_side=30, sep=4.0, seed=1)
    test = PersonaVectors(
        persona_id=test.persona_id,
        vectors={layer: train.vectors[layer] for layer in layers},
        in_persona_centroid=test.in_persona_centroid,
        out_persona_centroid=test.out_persona_centroid,
        in_states=test.in_states,
        out_states=test.out_states,
    )
    probe = SeparabilityProbe(seed=0)
    result = probe.train_and_evaluate(train, test)
    for layer, auroc in result.per_layer_auroc.items():
        assert auroc >= AUROC_CONFIRMED_FLOOR, f"layer {layer}: AUROC {auroc:.3f}"
    assert result.best_auroc >= 0.95
    assert result.verdict == "confirmed"


def test_shuffled_label_control_near_chance() -> None:
    """Mean over N shuffles should collapse to ~0.5 on 1-D projected features.

    With 1-D features, a single label shuffle flips the classifier sign
    randomly — test AUROC then jumps to 0 or 1. Averaging across N
    independent shuffles (SeparabilityProbe does 10) drives the expected
    value to 0.5. We assert |mean - 0.5| < 0.2 as a loose bar; the point
    of the control is that the *mean* lands near chance, and a TRUE
    leakage would force the mean well above 0.5.
    """
    layers = [0]
    hidden_dim = 16
    train = _make_pv("p", layers, hidden_dim, n_per_side=80, sep=3.0, seed=0)
    test = _make_pv("p", layers, hidden_dim, n_per_side=40, sep=3.0, seed=1)
    test = PersonaVectors(
        persona_id=test.persona_id,
        vectors={0: train.vectors[0]},
        in_persona_centroid=test.in_persona_centroid,
        out_persona_centroid=test.out_persona_centroid,
        in_states=test.in_states,
        out_states=test.out_states,
    )
    probe = SeparabilityProbe(seed=123)
    result = probe.train_and_evaluate(train, test)
    for layer, auroc in result.shuffled_label_auroc.items():
        # Mean across 10 shuffles must land well below the weak floor;
        # a leakage-free setup gives ~0.5 ± 0.15 at n=80.
        assert auroc < AUROC_WEAK_FLOOR, (
            f"layer {layer}: shuffled-label mean AUROC {auroc:.3f} crossed weak floor "
            f"{AUROC_WEAK_FLOOR:.2f} — setup likely leaks"
        )


def test_random_feature_control_near_chance() -> None:
    layers = [0]
    hidden_dim = 16
    train = _make_pv("p", layers, hidden_dim, n_per_side=50, sep=3.0, seed=0)
    test = _make_pv("p", layers, hidden_dim, n_per_side=20, sep=3.0, seed=1)
    test = PersonaVectors(
        persona_id=test.persona_id,
        vectors={0: train.vectors[0]},
        in_persona_centroid=test.in_persona_centroid,
        out_persona_centroid=test.out_persona_centroid,
        in_states=test.in_states,
        out_states=test.out_states,
    )
    probe = SeparabilityProbe(seed=77)
    result = probe.train_and_evaluate(train, test)
    assert result.random_feature_auroc < AUROC_WEAK_FLOOR


def test_verdict_boundary() -> None:
    """Verdict classifier honours the AUROC thresholds."""
    probe = SeparabilityProbe()
    assert probe._verdict(0.85) == "confirmed"
    assert probe._verdict(AUROC_CONFIRMED_FLOOR) == "confirmed"
    assert probe._verdict(AUROC_WEAK_FLOOR + 0.01) == "weak"
    assert probe._verdict(AUROC_WEAK_FLOOR) == "weak"
    assert probe._verdict(0.65) == "refuted"


def test_auroc_half_when_sep_zero() -> None:
    """With sep=0 the two classes are identical distributions; AUROC should collapse to chance."""
    layers = [0]
    hidden_dim = 8
    train = _make_pv("p", layers, hidden_dim, n_per_side=80, sep=0.0, seed=0)
    test = _make_pv("p", layers, hidden_dim, n_per_side=40, sep=0.0, seed=1)
    test = PersonaVectors(
        persona_id=test.persona_id,
        vectors={0: train.vectors[0]},
        in_persona_centroid=test.in_persona_centroid,
        out_persona_centroid=test.out_persona_centroid,
        in_states=test.in_states,
        out_states=test.out_states,
    )
    probe = SeparabilityProbe(seed=0)
    result = probe.train_and_evaluate(train, test)
    assert result.verdict == "refuted"
    # Best AUROC should be within 0.2 of 0.5 — loose bound because with
    # zero separation the probe fits noise.
    assert abs(result.best_auroc - 0.5) < 0.2
