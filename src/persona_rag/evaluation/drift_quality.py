"""M3 drift-gate quality metric: precision / recall against MiniCheck labels.

For each turn in an M3 conversation, we have:

- The gate's decision (``gate_should_gate``: True / False), pulled from
  per-turn metadata.
- A "ground-truth" inconsistency label derived from the MiniCheck
  metric: a turn is *inconsistent* if its per-sentence supported
  fraction falls below a threshold (default 0.5 -> at least half the
  sentences are not supported by any self-fact).

The metric reports precision / recall / F1 of the gate's "should_gate"
trigger against the MiniCheck label. The headline ``value`` is the F1
score (0-1, higher is better -- gate triggers correlate with measured
inconsistency).

Vacuous cases:

- Persona has no self-facts -> the MiniCheck label is undefined; metric
  returns NaN with ``vacuous=True`` in metadata.
- Conversation set contains no M3 turns with ``gate_should_gate``
  metadata -> metric returns NaN with ``no_gate_metadata=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation, MetricResult
from persona_rag.evaluation.minicheck_metric import (
    MiniCheckScorer,
    split_sentences,
)
from persona_rag.schema.persona import Persona


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Standard 2x2 confusion-matrix counts."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


def _turn_is_inconsistent(
    scorer: MiniCheckScorer,
    self_facts: list[str],
    assistant_text: str,
    contradiction_threshold: float,
    self_fact_support_threshold: float,
) -> bool:
    """Score one assistant turn; return True if MiniCheck flags ≥ threshold contradictions.

    A sentence is "contradicted" iff no self-fact supports it at
    ``self_fact_support_threshold``. The turn is inconsistent if the
    fraction of contradicted sentences exceeds
    ``contradiction_threshold``.
    """
    if not self_facts:
        return False
    sentences = split_sentences(assistant_text)
    if not sentences:
        return False
    pairs = [(sf, sentence) for sentence in sentences for sf in self_facts]
    probs = scorer.score_batch(pairs)
    n_facts = len(self_facts)
    contradicted = 0
    for i in range(len(sentences)):
        fact_probs = probs[i * n_facts : (i + 1) * n_facts]
        if max(fact_probs) < self_fact_support_threshold:
            contradicted += 1
    return (contradicted / len(sentences)) > contradiction_threshold


@dataclass
class DriftQualityMetric:
    """Gate precision / recall / F1 against MiniCheck-derived inconsistency labels.

    Only conversations whose ``mechanism`` is ``"m3"`` contribute. For
    each M3 turn we read ``gate_should_gate`` from per-turn metadata
    and the assistant text from ``ScoredTurn``. Turns missing the
    gate metadata are skipped (with a warning).
    """

    scorer: MiniCheckScorer
    contradiction_threshold: float = 0.0  # > 0 of sentences contradicted = inconsistent
    self_fact_support_threshold: float = 0.5
    name: str = "drift_quality_f1"

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        """Compute precision / recall / F1 of the gate vs MiniCheck labels."""
        self_facts = [sf.fact for sf in persona.self_facts]
        if not self_facts:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                per_persona={persona.persona_id or "<unknown>": float("nan")},
                metadata={"vacuous": True, "reason": "persona has no self_facts"},
            )

        m3_convs = [c for c in conversations if c.mechanism.lower() == "m3"]
        if not m3_convs:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                per_persona={persona.persona_id or "<unknown>": float("nan")},
                metadata={"vacuous": True, "reason": "no m3 conversations in input"},
            )

        tp = fp = tn = fn = 0
        skipped = 0
        per_conv_f1: list[float] = []
        per_conv_ids: list[str] = []

        for conv in m3_convs:
            conv_tp = conv_fp = conv_tn = conv_fn = 0
            for i, turn in enumerate(conv.turns):
                meta = conv.per_turn_metadata[i] if i < len(conv.per_turn_metadata) else {}
                # Pipelines may nest the gate fields inside ``metadata``;
                # flatten if so.
                if "metadata" in meta and isinstance(meta["metadata"], dict):
                    nested = meta["metadata"]
                    if "gate_should_gate" in nested and "gate_should_gate" not in meta:
                        meta = {**meta, "gate_should_gate": nested["gate_should_gate"]}
                if "gate_should_gate" not in meta:
                    skipped += 1
                    continue
                predicted_drift = bool(meta["gate_should_gate"])
                actual_drift = _turn_is_inconsistent(
                    self.scorer,
                    self_facts,
                    turn.assistant_text,
                    self.contradiction_threshold,
                    self.self_fact_support_threshold,
                )
                if predicted_drift and actual_drift:
                    conv_tp += 1
                elif predicted_drift and not actual_drift:
                    conv_fp += 1
                elif not predicted_drift and actual_drift:
                    conv_fn += 1
                else:
                    conv_tn += 1
            conv_counts = ConfusionCounts(conv_tp, conv_fp, conv_tn, conv_fn)
            per_conv_f1.append(conv_counts.f1 if not math.isnan(conv_counts.f1) else 0.0)
            per_conv_ids.append(conv.conversation_id)
            tp += conv_tp
            fp += conv_fp
            tn += conv_tn
            fn += conv_fn

        if skipped:
            logger.warning(
                "drift_quality: skipped {} M3 turns missing gate_should_gate metadata",
                skipped,
            )

        overall = ConfusionCounts(tp, fp, tn, fn)
        f1 = overall.f1 if not math.isnan(overall.f1) else 0.0
        meta_stats: dict[str, Any] = {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": f1,
            "skipped_turns": skipped,
            "scorer": self.scorer.name,
            "self_fact_support_threshold": self.self_fact_support_threshold,
            "contradiction_threshold": self.contradiction_threshold,
        }
        return MetricResult(
            name=self.name,
            value=f1,
            per_conversation=per_conv_f1,
            per_conversation_ids=per_conv_ids,
            per_persona={persona.persona_id or "<unknown>": f1},
            metadata=meta_stats,
        )


__all__ = ["ConfusionCounts", "DriftQualityMetric"]
