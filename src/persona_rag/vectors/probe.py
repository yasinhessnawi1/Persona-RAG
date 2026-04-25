"""Linear-separability probe + spurious-correlation controls.

Projects hidden states onto the per-layer persona vector, fits a 1-D
logistic regression on the train split, and reports AUROC on the
prompt-disjoint test split. Computes two mandatory controls alongside the
main result:

1. **Shuffled-label control** — same features, labels permuted. Expected
   AUROC ~0.5. If a shuffled-label probe achieves >= 0.70, something is
   wrong with the evaluation setup (data leakage).
2. **Random-feature control** — labels intact, features replaced with random
   noise of the same shape. Expected AUROC ~0.5. If this approaches the
   main number, the persona direction is not carrying the separability.

AUROC verdict thresholds:

- ``AUROC >= 0.80`` → ``confirmed``.
- ``0.70 <= AUROC < 0.80`` → ``weak``.
- ``AUROC < 0.70`` → ``refuted``.

Uses scikit-learn. Logistic regression on 1-D features is equivalent to a
threshold-and-sigmoid on the projection; using ``LogisticRegression`` makes
the math legible to reviewers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from persona_rag.vectors.extractor import PersonaVectors

Verdict = Literal["confirmed", "weak", "refuted"]

# Verdict thresholds. Duplicated here so the test suite can import them.
AUROC_CONFIRMED_FLOOR: float = 0.80
AUROC_WEAK_FLOOR: float = 0.70


@dataclass(frozen=True, slots=True)
class SeparabilityResult:
    """Output of :meth:`SeparabilityProbe.train_and_evaluate`.

    Attributes
    ----------
    per_layer_auroc:
        ``{layer_idx: test_auroc}`` — the headline number, per layer.
    per_layer_train_auroc:
        ``{layer_idx: train_auroc}`` — reported for over-fit detection; if
        test ≪ train, the probe is memorising.
    best_layer:
        Layer index with highest test AUROC. When several layers tie, the
        smallest-index layer wins (arbitrary but deterministic).
    best_auroc:
        ``per_layer_auroc[best_layer]``.
    shuffled_label_auroc:
        ``{layer_idx: test_auroc}`` with train labels shuffled. Should be
        near 0.5.
    random_feature_auroc:
        Float — AUROC when projecting random noise onto the persona vector
        at ``best_layer`` (labels intact, features random). Should be near
        0.5.
    verdict:
        One of ``confirmed`` / ``weak`` / ``refuted`` based on
        ``best_auroc`` against the AUROC thresholds.
    notes:
        Free-form diagnostic strings the validator appends (e.g. per-layer
        over-fit warnings).
    """

    per_layer_auroc: dict[int, float]
    per_layer_train_auroc: dict[int, float]
    best_layer: int
    best_auroc: float
    shuffled_label_auroc: dict[int, float]
    random_feature_auroc: float
    verdict: Verdict
    notes: list[str] = field(default_factory=list)


class SeparabilityProbe:
    """Mass-mean persona-vector probe with controls.

    Parameters
    ----------
    seed:
        Seed for shuffled-label permutation, random-feature generation, and
        scikit-learn's RNG. Makes every probe run bit-reproducible.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        train_vectors: PersonaVectors,
        test_vectors: PersonaVectors,
    ) -> SeparabilityResult:
        """Compute per-layer AUROC + spurious-correlation controls.

        Takes two :class:`PersonaVectors` objects — one extracted from the
        train split of the contrast set, one from the test (prompt-disjoint)
        split. The persona direction is fit on train activations; AUROC is
        measured on test-split projections.

        This is the entry point the validation script calls.
        """
        layers = train_vectors.layers
        if test_vectors.layers != layers:
            raise ValueError(
                f"train/test layer mismatch: train={layers}, test={test_vectors.layers}"
            )

        per_layer_auroc: dict[int, float] = {}
        per_layer_train_auroc: dict[int, float] = {}
        shuffled: dict[int, float] = {}
        notes: list[str] = []

        for layer in layers:
            direction = train_vectors.vectors[layer]  # (hidden_dim,)
            # Main probe: project train + test activations onto the
            # train-fit persona vector; fit logistic regression on the
            # 1-D train projection; AUROC on the 1-D test projection.
            train_auroc, test_auroc = self._fit_and_score(
                direction,
                train_vectors.in_states[layer],
                train_vectors.out_states[layer],
                test_vectors.in_states[layer],
                test_vectors.out_states[layer],
            )
            per_layer_auroc[layer] = test_auroc
            per_layer_train_auroc[layer] = train_auroc

            if train_auroc - test_auroc > 0.1:
                notes.append(
                    f"layer {layer}: train AUROC ({train_auroc:.3f}) >> "
                    f"test AUROC ({test_auroc:.3f}) — possible over-fit"
                )

            # Shuffled-label control — same projections, labels permuted.
            shuffled[layer] = self._shuffled_label_auroc(
                direction,
                train_vectors.in_states[layer],
                train_vectors.out_states[layer],
                test_vectors.in_states[layer],
                test_vectors.out_states[layer],
            )
            if shuffled[layer] >= AUROC_WEAK_FLOOR:
                notes.append(
                    f"layer {layer}: SHUFFLED-label control reached "
                    f"{shuffled[layer]:.3f} — main result suspect "
                    "(possible leakage)"
                )

        best_layer = min(
            (
                layer
                for layer, auroc in per_layer_auroc.items()
                if auroc == max(per_layer_auroc.values())
            ),
        )
        best_auroc = per_layer_auroc[best_layer]
        verdict: Verdict = self._verdict(best_auroc)

        # Random-feature control at best_layer.
        random_feature_auroc = self._random_feature_auroc(
            train_vectors.in_states[best_layer],
            train_vectors.out_states[best_layer],
            test_vectors.in_states[best_layer],
            test_vectors.out_states[best_layer],
        )
        if random_feature_auroc >= AUROC_WEAK_FLOOR:
            notes.append(
                f"random-feature control at best_layer={best_layer} reached "
                f"{random_feature_auroc:.3f} — main result suspect"
            )

        logger.info(
            "probe complete: best_layer={} best_auroc={:.3f} verdict={} "
            "shuffled_min-max={:.3f}-{:.3f} random_feature={:.3f}",
            best_layer,
            best_auroc,
            verdict,
            min(shuffled.values()),
            max(shuffled.values()),
            random_feature_auroc,
        )

        return SeparabilityResult(
            per_layer_auroc=per_layer_auroc,
            per_layer_train_auroc=per_layer_train_auroc,
            best_layer=best_layer,
            best_auroc=best_auroc,
            shuffled_label_auroc=shuffled,
            random_feature_auroc=random_feature_auroc,
            verdict=verdict,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _project(states: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
        """Raw scalar projection of each row of ``states`` onto ``direction``.

        Returns shape ``(n,)``. Computed in float32.
        """
        if states.dim() != 2:
            raise ValueError(f"expected (n, hidden_dim), got {tuple(states.shape)}")
        if direction.dim() != 1:
            raise ValueError(f"expected (hidden_dim,), got {tuple(direction.shape)}")
        if states.shape[1] != direction.shape[0]:
            raise ValueError(
                f"hidden_dim mismatch: states={states.shape[1]}, direction={direction.shape[0]}"
            )
        return (states.to(torch.float32) @ direction.to(torch.float32)).numpy()

    def _fit_and_score(
        self,
        direction: torch.Tensor,
        train_in: torch.Tensor,
        train_out: torch.Tensor,
        test_in: torch.Tensor,
        test_out: torch.Tensor,
    ) -> tuple[float, float]:
        """Fit 1-D logistic regression on train projection; return (train, test) AUROC."""
        train_proj_in = self._project(train_in, direction)
        train_proj_out = self._project(train_out, direction)
        test_proj_in = self._project(test_in, direction)
        test_proj_out = self._project(test_out, direction)

        x_train = np.concatenate([train_proj_in, train_proj_out]).reshape(-1, 1)
        y_train = np.concatenate([np.ones_like(train_proj_in), np.zeros_like(train_proj_out)])
        x_test = np.concatenate([test_proj_in, test_proj_out]).reshape(-1, 1)
        y_test = np.concatenate([np.ones_like(test_proj_in), np.zeros_like(test_proj_out)])

        clf = LogisticRegression(solver="lbfgs", random_state=self._seed, max_iter=500)
        clf.fit(x_train, y_train)

        train_scores = clf.predict_proba(x_train)[:, 1]
        test_scores = clf.predict_proba(x_test)[:, 1]
        return (
            float(roc_auc_score(y_train, train_scores)),
            float(roc_auc_score(y_test, test_scores)),
        )

    def _shuffled_label_auroc(
        self,
        direction: torch.Tensor,
        train_in: torch.Tensor,
        train_out: torch.Tensor,
        test_in: torch.Tensor,
        test_out: torch.Tensor,
        n_shuffles: int = 10,
    ) -> float:
        """Mean AUROC over `n_shuffles` independent shuffles of train labels.

        On 1-D projected features, a single shuffle is high-variance: the
        logistic sigmoid direction is either "right" or "wrong" by pure
        chance, pushing test AUROC to ~1 or ~0. Averaging across shuffles
        drives the expected value to 0.5 (no signal). Reviewers compare
        this mean to the main AUROC — a main probe materially above the
        shuffled-label mean is the evidence the gate needs.
        """
        train_proj_in = self._project(train_in, direction)
        train_proj_out = self._project(train_out, direction)
        test_proj_in = self._project(test_in, direction)
        test_proj_out = self._project(test_out, direction)

        x_train = np.concatenate([train_proj_in, train_proj_out]).reshape(-1, 1)
        y_train = np.concatenate([np.ones_like(train_proj_in), np.zeros_like(train_proj_out)])
        x_test = np.concatenate([test_proj_in, test_proj_out]).reshape(-1, 1)
        y_test = np.concatenate([np.ones_like(test_proj_in), np.zeros_like(test_proj_out)])

        aurocs: list[float] = []
        for i in range(n_shuffles):
            y_shuffled = y_train.copy()
            # Use a fresh RNG per shuffle so we're not reusing state.
            np.random.default_rng(self._seed + i + 1).shuffle(y_shuffled)
            # Every shuffle-that-leaves-only-one-class gives roc_auc_score a
            # single-class y_train; skip it (logistic regression would also
            # refuse to fit).
            if len(set(y_shuffled)) < 2:
                continue
            clf = LogisticRegression(solver="lbfgs", random_state=self._seed + i, max_iter=500)
            clf.fit(x_train, y_shuffled)
            test_scores = clf.predict_proba(x_test)[:, 1]
            aurocs.append(float(roc_auc_score(y_test, test_scores)))
        if not aurocs:
            return 0.5
        return float(np.mean(aurocs))

    def _random_feature_auroc(
        self,
        train_in: torch.Tensor,
        train_out: torch.Tensor,
        test_in: torch.Tensor,
        test_out: torch.Tensor,
        n_random_dirs: int = 10,
    ) -> float:
        """Mean AUROC over `n_random_dirs` random probe directions.

        Labels are intact, but the probe direction is a random unit vector
        in hidden space. Any separability has to come from a lucky
        random projection lining up with a class-carrying axis; averaging
        across many random directions collapses this to chance (~0.5 for
        labels orthogonal to a random direction on average). A main AUROC
        materially above this mean means the persona direction is doing
        the work, not generic feature geometry.
        """
        hidden_dim = train_in.shape[1]
        aurocs: list[float] = []
        for i in range(n_random_dirs):
            g = torch.Generator().manual_seed(self._seed + 101 + i)
            direction = torch.randn(hidden_dim, generator=g, dtype=torch.float32)
            direction = direction / direction.norm()
            _train_auroc, test_auroc = self._fit_and_score(
                direction, train_in, train_out, test_in, test_out
            )
            aurocs.append(test_auroc)
        return float(np.mean(aurocs))

    @staticmethod
    def _verdict(auroc: float) -> Verdict:
        """Translate AUROC to a verdict label."""
        if auroc >= AUROC_CONFIRMED_FLOOR:
            return "confirmed"
        if auroc >= AUROC_WEAK_FLOOR:
            return "weak"
        return "refuted"
