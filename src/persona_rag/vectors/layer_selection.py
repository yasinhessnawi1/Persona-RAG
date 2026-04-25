"""Pick a single global best layer from per-persona AUROC tables.

Downstream code consumes a single integer layer for the drift signal, not a
persona→layer mapping. :func:`pick_global_best_layer` takes the validation
script's per-persona AUROC output and returns the layer that maximises mean
AUROC across personas. If per-persona best layers differ by more than
``diagnostic_gap_layers`` (default 4), an advisory string is returned
alongside to flag the finding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GlobalBestLayerResult:
    """Output of :func:`pick_global_best_layer`.

    Attributes
    ----------
    best_layer:
        Layer index with highest mean test AUROC across personas.
    mean_auroc:
        The mean AUROC at ``best_layer``.
    per_persona_best:
        ``{persona_id: layer_with_highest_auroc_for_that_persona}`` —
        the diagnostic breakdown.
    gap_warning:
        ``None`` if per-persona best layers agree closely; otherwise a
        short human-readable string for the research note.
    """

    best_layer: int
    mean_auroc: float
    per_persona_best: dict[str, int]
    gap_warning: str | None = None


def pick_global_best_layer(
    per_persona_auroc: dict[str, dict[int, float]],
    *,
    diagnostic_gap_layers: int = 4,
) -> GlobalBestLayerResult:
    """Choose the layer that maximises mean AUROC across personas.

    Parameters
    ----------
    per_persona_auroc:
        ``{persona_id: {layer: auroc}}``. Every persona must report the
        same layer set (same layer sweep).
    diagnostic_gap_layers:
        If the min and max per-persona best-layer indices differ by more
        than this, a ``gap_warning`` is returned (a wide gap is a finding
        about persona-vector universality).
    """
    if not per_persona_auroc:
        raise ValueError("per_persona_auroc is empty")

    # Sanity check: every persona lists the same layer sweep.
    layer_sets = [frozenset(d.keys()) for d in per_persona_auroc.values()]
    if len(set(layer_sets)) != 1:
        raise ValueError(f"per-persona layer sets differ: {[sorted(ls) for ls in layer_sets]}")
    layers = sorted(next(iter(layer_sets)))

    # Mean AUROC per layer.
    mean_per_layer: dict[int, float] = {}
    n_personas = len(per_persona_auroc)
    for layer in layers:
        total = sum(d[layer] for d in per_persona_auroc.values())
        mean_per_layer[layer] = total / n_personas

    best_layer = min(
        (layer for layer, mean in mean_per_layer.items() if mean == max(mean_per_layer.values())),
    )
    mean_auroc = mean_per_layer[best_layer]

    per_persona_best: dict[str, int] = {}
    for pid, table in per_persona_auroc.items():
        per_persona_best[pid] = min(
            (layer for layer, auroc in table.items() if auroc == max(table.values())),
        )

    gap_warning: str | None = None
    if per_persona_best:
        lo = min(per_persona_best.values())
        hi = max(per_persona_best.values())
        if hi - lo > diagnostic_gap_layers:
            gap_warning = (
                f"per-persona best layers span {hi - lo} layers "
                f"({lo}..{hi}) — may indicate persona-specific depth, "
                "flag in research note"
            )

    return GlobalBestLayerResult(
        best_layer=best_layer,
        mean_auroc=mean_auroc,
        per_persona_best=per_persona_best,
        gap_warning=gap_warning,
    )
