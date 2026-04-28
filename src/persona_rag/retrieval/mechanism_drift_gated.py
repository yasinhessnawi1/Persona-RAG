"""Drift-gated hybrid retrieval mechanism.

Per turn:

1. Drift gate (LLM judge) decides whether the assistant's most-recent turn
   drifted. One LLM call.
2. If ``should_gate=False`` — cheap path: typed retrieval + responder.
3. If ``should_gate=True``  — gated path:

   a. Typed retrieval with ``augment_for_drift=True`` — retrieval re-keys on
      the drifted-turn content. The responder produces one baseline candidate
      at greedy.
   b. ``n_candidates - 1`` additional candidates are generated from the
      same prompt at varied temperatures.
   c. Hybrid ranker (reward-model + cross-family LLM judge) selects the
      best candidate. The ranker can fall back to a 1-signal LLM-judge
      ranker via ``enabled_signals: ["judge"]``.

The cheap path runs no post-hoc check on the responder's output — the
gate is the only drift detector in this architecture. Per-turn cost:

- Cheap path: gate(1) + responder(1) = 2 LLM calls.
- Gated path: gate(1) + responder(1, augmented; baseline) + (n-1) extra
  candidates + ranker(2) = ``2 + n_candidates`` LLM calls.

Per-turn metadata records each component for cost auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Response, Turn
from persona_rag.retrieval.drift_gate import LlmJudgeDriftGate
from persona_rag.retrieval.hybrid_ranker import HybridRanker, RankedCandidate
from persona_rag.retrieval.typed_retrieval import TypedRetrievalRAG
from persona_rag.schema.persona import Persona

# Default extra-candidate temperatures. The baseline candidate is M1's
# greedy generation at temperature=0; these two extras span deterministic-
# to-creative without paying for >2x the cheap-path cost per turn.
DEFAULT_EXTRA_CANDIDATE_TEMPERATURES: tuple[float, ...] = (0.7, 1.0)


@dataclass
class DriftGatedMechanism:
    """LLM-judge-gated hybrid mechanism.

    Composes the typed-retrieval pipeline (cheap-path responder +
    gated-path augmented retrieval), the LLM-judge drift gate, and the
    2-signal hybrid ranker. Cheap-path branches stay shape-equivalent to
    the underlying typed-retrieval pipeline (the gate is a no-op wrapper
    around an unchanged call); gated-path branches add candidate
    generation and ranking on top of the augmented response.
    """

    backend: LLMBackend
    m1: TypedRetrievalRAG
    drift_gate: LlmJudgeDriftGate
    hybrid_ranker: HybridRanker
    n_candidates: int = 3
    extra_candidate_temperatures: tuple[float, ...] = DEFAULT_EXTRA_CANDIDATE_TEMPERATURES
    extra_candidate_max_new_tokens: int | None = None  # None → reuse M1's max_new_tokens
    name: str = "drift_gated"

    # Internal: bumps every gated-path turn so the candidate seeds are
    # distinct across calls within the same process.
    _gated_call_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_candidates < 1:
            raise ValueError(
                f"n_candidates must be >= 1, got {self.n_candidates}. "
                "n_candidates=1 collapses to baseline+ranker (no diversity)."
            )
        if len(self.extra_candidate_temperatures) != self.n_candidates - 1:
            raise ValueError(
                f"extra_candidate_temperatures has {len(self.extra_candidate_temperatures)} "
                f"entries; needs n_candidates - 1 = {self.n_candidates - 1}. "
                "Each non-baseline candidate needs one explicit temperature."
            )

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Turn] | None = None,
        *,
        seed: int | None = None,
    ) -> Response:
        """Single-turn entry point. Branches on the gate's decision."""
        history = list(history) if history else []

        # 1. Drift gate.
        drift_check = self.drift_gate.check(persona=persona, query=query, history=history)
        gate_metadata = {
            "gate_flag": drift_check.flag,
            "gate_confidence": drift_check.confidence,
            "gate_rationale": drift_check.rationale,
            "gate_should_gate": drift_check.should_gate,
            "gate_template_version": drift_check.template_version,
            "gate_threshold": self.drift_gate.confidence_threshold,
            "gate_judge_model": self.drift_gate.judge.name,
        }

        if not drift_check.should_gate:
            return self._cheap_path(query, persona, history, seed=seed, gate_metadata=gate_metadata)
        return self._gated_path(query, persona, history, seed=seed, gate_metadata=gate_metadata)

    # --------------------------------------------------------- cheap path

    def _cheap_path(
        self,
        query: str,
        persona: Persona,
        history: list[Turn],
        *,
        seed: int | None,
        gate_metadata: dict[str, Any],
    ) -> Response:
        """Typed retrieval + responder. 1 LLM call beyond the gate."""
        m1_response = self.m1.respond(query, persona, history, seed=seed, augment_for_drift=False)
        meta = dict(m1_response.metadata)
        meta.update(gate_metadata)
        meta.update(
            {
                "mechanism": self.name,
                "path_taken": "cheap",
                "drift_gated": False,
                "n_candidates_generated": 1,
                "selected_candidate_index": 0,
                "candidate_scores": [],
                "llm_call_count": {
                    "gate": 1,
                    "responder": 1,
                    "candidates": 0,
                    "ranker_character_rm": 0,
                    "ranker_judge": 0,
                    "total": 2,
                },
            }
        )
        logger.info(
            "drift_gated[{}] turn cheap path: gate_conf={:.2f} responder={}",
            persona.persona_id,
            gate_metadata["gate_confidence"],
            self.backend.name,
        )
        return Response(
            text=m1_response.text,
            retrieved_knowledge=m1_response.retrieved_knowledge,
            retrieved_persona=m1_response.retrieved_persona,
            prompt_used=m1_response.prompt_used,
            steering_applied=False,
            drift_signal=None,
            metadata=meta,
        )

    # --------------------------------------------------------- gated path

    def _gated_path(
        self,
        query: str,
        persona: Persona,
        history: list[Turn],
        *,
        seed: int | None,
        gate_metadata: dict[str, Any],
    ) -> Response:
        """Augmented retrieval + N candidates + 2-signal hybrid rerank."""
        # 2. Augmented retrieval. Produces baseline candidate (#0).
        baseline = self.m1.respond(query, persona, history, seed=seed, augment_for_drift=True)

        # 3. Generate (n_candidates - 1) additional candidates from the
        #    augmented prompt at varied temperatures. Seeds are derived from
        #    `seed` so a fixed seed reproduces the candidate set, while each
        #    call's seeds are distinct (avoiding silent collisions when k>1
        #    candidates share the same temperature elsewhere).
        self._gated_call_counter += 1
        prompt = baseline.prompt_used
        max_new_tokens = self.extra_candidate_max_new_tokens or self.m1.max_new_tokens
        extra_candidates: list[str] = []
        for i, temp in enumerate(self.extra_candidate_temperatures):
            cand_seed = self._seed_for_candidate(seed, i)
            text = self.backend.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=float(temp),
                seed=cand_seed,
            )
            extra_candidates.append(text)

        candidates = [baseline.text, *extra_candidates]

        # 4. Hybrid rerank.
        ranked: list[RankedCandidate] = self.hybrid_ranker.rank(
            persona=persona, query=query, candidates=candidates
        )
        best = ranked[0]

        ranker_character_rm_calls = (
            len(candidates) if "character_rm" in self.hybrid_ranker.enabled_signals else 0
        )
        ranker_judge_calls = len(candidates) if "judge" in self.hybrid_ranker.enabled_signals else 0
        # Per-turn cost: gate(1) + baseline(1) + (n-1) extra + ranker scoring per candidate.
        total_calls = (
            1  # gate
            + 1  # baseline
            + len(extra_candidates)
            + ranker_character_rm_calls
            + ranker_judge_calls
        )

        meta = dict(baseline.metadata)
        meta.update(gate_metadata)
        meta.update(
            {
                "mechanism": self.name,
                "path_taken": "gated",
                "drift_gated": True,
                "n_candidates_generated": len(candidates),
                "n_extra_candidates": len(extra_candidates),
                "extra_candidate_temperatures": list(self.extra_candidate_temperatures),
                "selected_candidate_index": best.candidate_ix,
                "candidate_scores": [
                    {
                        "candidate_ix": rc.candidate_ix,
                        "rank_ix": rc.rank_ix,
                        "weighted": rc.weighted_score,
                        "raw": dict(rc.raw_scores),
                        "normalised": dict(rc.normalised_scores),
                    }
                    for rc in ranked
                ],
                "all_candidates": list(candidates),
                "ranker_signals": list(self.hybrid_ranker.enabled_signals),
                "ranker_weights": dict(self.hybrid_ranker.weights),
                "llm_call_count": {
                    "gate": 1,
                    "responder": 1,
                    "candidates": len(extra_candidates),
                    "ranker_character_rm": ranker_character_rm_calls,
                    "ranker_judge": ranker_judge_calls,
                    "total": total_calls,
                },
            }
        )
        logger.info(
            "drift_gated[{}] turn gated path: gate_conf={:.2f} n_cand={} selected={} "
            "weighted={:.3f}",
            persona.persona_id,
            gate_metadata["gate_confidence"],
            len(candidates),
            best.candidate_ix,
            best.weighted_score,
        )
        return Response(
            text=best.text,
            retrieved_knowledge=baseline.retrieved_knowledge,
            retrieved_persona=baseline.retrieved_persona,
            prompt_used=prompt,
            steering_applied=False,
            drift_signal=None,
            metadata=meta,
        )

    @staticmethod
    def _seed_for_candidate(seed: int | None, candidate_ix: int) -> int | None:
        """Derive a distinct seed per non-baseline candidate.

        Returning ``None`` (when the caller passes ``seed=None``) lets the
        backend pick its own. Otherwise we offset by ``candidate_ix + 1``
        so candidate ``#1`` differs from the baseline at ``#0``,
        candidate ``#2`` differs from both, etc.
        """
        if seed is None:
            return None
        return seed + candidate_ix + 1


__all__ = [
    "DEFAULT_EXTRA_CANDIDATE_TEMPERATURES",
    "DriftGatedMechanism",
]
