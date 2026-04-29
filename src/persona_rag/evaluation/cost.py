"""Per-mechanism cost tracker.

Reads per-turn metadata that the existing pipelines already record
(``metadata["gate_*"]``, ``metadata["candidates_n"]``, etc.) and turns
it into a per-mechanism cost summary.

What we report per turn:

- ``llm_calls`` -- count of LLM forward passes for this turn:
  - B1 / B2 / M1: 1 (responder).
  - M3 cheap path: 2 (gate + responder).
  - M3 gated path: ``2 + n_candidates + 2`` (gate + baseline + extras + ranker).
- ``judge_calls`` -- subset of ``llm_calls`` that hit a judge model
  (gate + ranker for M3; 0 elsewhere).
- ``latency_seconds`` -- wall-clock if the upstream pipeline recorded
  it; otherwise NaN.
- ``prompt_tokens`` / ``output_tokens`` -- if recorded; otherwise NaN.

The metric reads everything from the existing ``per_turn_metadata`` on
``EvalConversation``, so it does not re-instrument the responders. New
turn-level fields recorded by future pipelines flow through
automatically as long as they live under
``EvalConversation.per_turn_metadata``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from persona_rag.evaluation.metrics import EvalConversation, MetricResult
from persona_rag.schema.persona import Persona


def _llm_calls_for_turn(mechanism: str, turn_meta: dict[str, Any]) -> int:
    """Compute LLM-call count for one turn from its metadata."""
    mechanism = mechanism.lower()
    if mechanism in ("b1", "b2", "m1"):
        return 1
    if mechanism == "m3":
        # Gate is always 1. Cheap path: + 1 responder. Gated path:
        # + (n_candidates + 1) responders + 2 ranker calls (per the
        # mechanism's existing accounting).
        gate = 1
        if turn_meta.get("gate_should_gate", False):
            n_candidates = int(turn_meta.get("candidates_n", 1))
            ranker_calls = int(turn_meta.get("ranker_judge_calls", 2))
            return gate + n_candidates + ranker_calls
        return gate + 1
    # Unknown mechanism: best-effort = 1.
    return int(turn_meta.get("llm_calls", 1))


def _judge_calls_for_turn(mechanism: str, turn_meta: dict[str, Any]) -> int:
    """Calls that hit a judge model (for M3, ranking + gating; 0 elsewhere)."""
    mechanism = mechanism.lower()
    if mechanism != "m3":
        return 0
    judge_calls = 1  # gate is always called
    if turn_meta.get("gate_should_gate", False):
        judge_calls += int(turn_meta.get("ranker_judge_calls", 2))
    return judge_calls


def _maybe_float(value: Any) -> float:
    """NaN for missing / non-numeric, else float(value)."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass
class CostTracker:
    """Aggregate per-mechanism cost from per-turn pipeline metadata.

    Each metric instance is single-mechanism (the runner instantiates
    one tracker per mechanism). The headline ``value`` is the mean
    ``llm_calls`` per turn across all conversations, exposing the
    operational cost the M3 paper claims to reduce vs always-on
    retrieval.
    """

    mechanism: str
    name: str = "cost_llm_calls_per_turn"

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        """Compute per-mechanism per-turn cost."""
        per_conv_means: list[float] = []
        per_conv_ids: list[str] = []
        total_calls = 0
        total_judge_calls = 0
        total_turns = 0
        gated_turns = 0

        latencies: list[float] = []
        prompt_tokens: list[float] = []
        output_tokens: list[float] = []

        for conv in conversations:
            turn_calls: list[int] = []
            for i, _turn in enumerate(conv.turns):
                meta = conv.per_turn_metadata[i] if i < len(conv.per_turn_metadata) else {}
                # ``run_baseline.py`` writes a ``metadata`` sub-dict; flatten if present.
                if "metadata" in meta and isinstance(meta["metadata"], dict):
                    flat = {**meta["metadata"]}
                    flat.update({k: v for k, v in meta.items() if k != "metadata"})
                    meta = flat
                calls = _llm_calls_for_turn(self.mechanism, meta)
                judge_calls = _judge_calls_for_turn(self.mechanism, meta)
                turn_calls.append(calls)
                total_calls += calls
                total_judge_calls += judge_calls
                total_turns += 1
                if meta.get("gate_should_gate"):
                    gated_turns += 1

                lat = _maybe_float(meta.get("latency_seconds"))
                if not math.isnan(lat):
                    latencies.append(lat)
                pt = _maybe_float(meta.get("prompt_tokens"))
                if not math.isnan(pt):
                    prompt_tokens.append(pt)
                ot = _maybe_float(meta.get("output_tokens"))
                if not math.isnan(ot):
                    output_tokens.append(ot)
            per_conv_means.append(statistics.fmean(turn_calls) if turn_calls else 0.0)
            per_conv_ids.append(conv.conversation_id)

        aggregate = total_calls / total_turns if total_turns else 0.0
        gate_trigger_rate = gated_turns / total_turns if total_turns else 0.0
        meta_stats: dict[str, Any] = {
            "mechanism": self.mechanism,
            "n_conversations": len(conversations),
            "total_turns": total_turns,
            "total_llm_calls": total_calls,
            "total_judge_calls": total_judge_calls,
            "mean_llm_calls_per_turn": aggregate,
            "mean_judge_calls_per_turn": (total_judge_calls / total_turns if total_turns else 0.0),
            "gated_turns": gated_turns,
            "gate_trigger_rate": gate_trigger_rate,
        }
        if latencies:
            meta_stats["mean_latency_seconds"] = statistics.fmean(latencies)
            meta_stats["p95_latency_seconds"] = sorted(latencies)[
                max(0, int(0.95 * len(latencies)) - 1)
            ]
        if prompt_tokens:
            meta_stats["mean_prompt_tokens"] = statistics.fmean(prompt_tokens)
        if output_tokens:
            meta_stats["mean_output_tokens"] = statistics.fmean(output_tokens)

        return MetricResult(
            name=self.name,
            value=aggregate,
            per_conversation=per_conv_means,
            per_conversation_ids=per_conv_ids,
            per_persona={persona.persona_id or "<unknown>": aggregate},
            metadata=meta_stats,
        )


__all__ = ["CostTracker"]
