"""Drift signal: cosine similarity to in-persona centroid, scaled [-1, 1].

Used by drift-gated retrieval at inference and by monitoring / telemetry
throughout evaluation. The convention:

- ``compute(in_persona_centroid) == +1.0``
- ``compute(out_persona_centroid) == -1.0``
- decision boundary near ``0.0``

The sign convention IS a test (see ``tests/test_drift.py``) — if a future
refactor flips the sign, the test goes red before anything downstream reads
a corrupted threshold.

Math (for a single persona-vector layer):

    # Let c_in, c_out be the centroids along the persona direction.
    a_in  = cos(h, c_in)           ∈ [-1, 1]
    a_out = cos(h, c_out)          ∈ [-1, 1]
    s     = cos(c_in, c_out)       ∈ [-1, 1]  (typically negative)
    drift = (a_in - a_out) / (1 - s)   ∈ [-1, 1]

This reduces to the standard cosine-to-centroid map whenever c_in and
c_out are antipodal (s = -1 ⇒ drift = (a_in - a_out)/2). For the general
non-antipodal case it normalises so that the two centroids still map to
±1 exactly (verified by a test).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from persona_rag.vectors.extractor import PersonaVectors


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity on 1-D tensors; float32 for numerical stability."""
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError(f"expected 1-D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    denom = a32.norm() * b32.norm()
    if denom < 1e-12:
        return torch.tensor(0.0, dtype=torch.float32)
    return (a32 @ b32) / denom


@dataclass(frozen=True, slots=True)
class DriftSignal:
    """Cosine-to-centroid drift signal, scaled to [-1, 1].

    Constructed from a :class:`PersonaVectors` object + the best layer chosen
    either per-persona or globally.

    Use at inference by calling :meth:`compute` with a current hidden state
    at the same layer the signal was built from.

    Attributes
    ----------
    layer:
        Transformer layer the drift signal lives at.
    in_centroid:
        In-persona centroid at ``layer``.
    out_centroid:
        Out-persona centroid at ``layer``.
    """

    layer: int
    in_centroid: torch.Tensor
    out_centroid: torch.Tensor

    @classmethod
    def from_persona_vectors(cls, pv: PersonaVectors, layer: int) -> DriftSignal:
        """Pick the layer-specific centroids out of a :class:`PersonaVectors`."""
        if layer not in pv.in_persona_centroid:
            raise KeyError(
                f"layer {layer} not in persona vectors (have {sorted(pv.in_persona_centroid)})"
            )
        return cls(
            layer=layer,
            in_centroid=pv.in_persona_centroid[layer].detach().clone(),
            out_centroid=pv.out_persona_centroid[layer].detach().clone(),
        )

    def compute(self, hidden: torch.Tensor) -> float:
        """Return drift ∈ [-1, 1].

        Sign convention (ENFORCED BY TEST):
            - ``+1.0`` at the in-persona centroid.
            - ``-1.0`` at the out-persona centroid.
            - ``0.0`` at the decision boundary (equidistant-by-cosine).
        """
        a_in = _cosine(hidden, self.in_centroid)
        a_out = _cosine(hidden, self.out_centroid)
        s = _cosine(self.in_centroid, self.out_centroid)
        denom = 1.0 - s
        # ``1 - s`` is 0 only if c_in == c_out (degenerate persona); guard.
        if denom.abs() < 1e-6:
            return 0.0
        value = ((a_in - a_out) / denom).item()
        # Clamp against float32 round-off — the math says value ∈ [-1, 1]
        # exactly at the centroids, but a 1e-7 overshoot is possible.
        return max(-1.0, min(1.0, float(value)))
