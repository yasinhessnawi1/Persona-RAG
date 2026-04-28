"""Two-signal hybrid ranker for the gated path.

Signals:

- ``character_rm``: trained reward-model score (a ``CharacterScorer``).
  Higher means more on-persona.
- ``judge``: cross-family LLM-judge score from a purpose-built evaluator.
  Different model family from the drift gate's judge so a single model's
  preferences do not appear in both gating and ranking.

Per-signal scores use different scales (the reward model returns a
logit-shaped real, the LLM judge returns a small ordinal score). The
ranker normalises each signal across the candidates of one ``rank()``
call via min-max normalisation, then takes a weighted sum. Both raw and
normalised scores are recorded on each ``RankedCandidate`` so downstream
analysis can audit how the weighted sum was reached.

The ``enabled_signals`` config field is first-class. The default
``["character_rm", "judge"]`` ships the 2-signal hybrid; flipping to
``["judge"]`` degenerates the ranker cleanly to a 1-signal LLM-judge
re-rank without code change. This is the documented fallback path if
the reward model's English competency is found wanting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.character_rm import CharacterScorer
from persona_rag.schema.persona import Persona

# Default per-signal weights; sensitivity-swept at evaluation time.
DEFAULT_HYBRID_WEIGHTS: dict[str, float] = {"character_rm": 0.5, "judge": 0.5}
DEFAULT_ENABLED_SIGNALS: tuple[str, ...] = ("character_rm", "judge")


@dataclass(frozen=True)
class RankedCandidate:
    """One ranked candidate plus per-signal score breakdown."""

    text: str
    weighted_score: float
    raw_scores: dict[str, float]
    normalised_scores: dict[str, float]
    rank_ix: int
    candidate_ix: int


@dataclass
class HybridRanker:
    """2-signal hybrid ranker with min-max normalisation and per-signal logging.

    Attribute fields:

    - ``character_rm``: a ``CharacterScorer`` (real wrapper or fake stub).
      Required even when disabled — keeping the field non-optional avoids
      an ``Optional`` check on every call site; ``score()`` is simply
      never invoked when ``"character_rm"`` is not in ``enabled_signals``.
    - ``rerank_judge``: an LLM backend used as the second scoring signal.
      The judge prompt asks for a small ordinal score (1-5) from which a
      scalar is parsed.
    - ``weights``: per-signal weight in the final sum. Weights for
      disabled signals are ignored.
    - ``enabled_signals``: which signals participate in this rank() call.
    """

    character_rm: CharacterScorer
    rerank_judge: LLMBackend
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HYBRID_WEIGHTS))
    enabled_signals: tuple[str, ...] = DEFAULT_ENABLED_SIGNALS
    judge_max_new_tokens: int = 64
    judge_temperature: float = 0.0

    @property
    def name(self) -> str:
        return f"hybrid_ranker(signals={list(self.enabled_signals)})"

    def rank(
        self,
        *,
        persona: Persona,
        query: str,
        candidates: list[str],
    ) -> list[RankedCandidate]:
        """Score and sort candidates by the weighted sum of enabled signals.

        Returns the candidates ranked best-first. ``rank_ix`` is the
        position in the sorted output (0 = best); ``candidate_ix`` is the
        original index in ``candidates``. ``raw_scores`` and
        ``normalised_scores`` carry per-signal breakdowns for diagnostics.
        """
        if not candidates:
            raise ValueError("hybrid_ranker.rank: candidates must not be empty")
        if not self.enabled_signals:
            raise ValueError(
                "hybrid_ranker.rank: enabled_signals must contain at least one of "
                f"{list(DEFAULT_ENABLED_SIGNALS)}"
            )

        n = len(candidates)
        raw: dict[str, list[float]] = {}
        if "character_rm" in self.enabled_signals:
            raw["character_rm"] = [
                float(self.character_rm.score(persona=persona, query=query, response=cand))
                for cand in candidates
            ]
        if "judge" in self.enabled_signals:
            raw["judge"] = [
                self._judge_score(persona=persona, query=query, response=cand)
                for cand in candidates
            ]

        normalised = {sig: _min_max_normalise(scores) for sig, scores in raw.items()}

        weighted: list[float] = []
        for i in range(n):
            score = 0.0
            weight_total = 0.0
            for sig in self.enabled_signals:
                w = float(self.weights.get(sig, 0.0))
                if w == 0.0:
                    continue
                score += w * normalised[sig][i]
                weight_total += w
            # Re-normalise weights so disabled-but-weighted signals do not
            # silently shrink the combined score.
            weighted.append(score / weight_total if weight_total > 0 else 0.0)

        order = sorted(range(n), key=lambda i: weighted[i], reverse=True)
        ranked: list[RankedCandidate] = []
        for rank_ix, cand_ix in enumerate(order):
            ranked.append(
                RankedCandidate(
                    text=candidates[cand_ix],
                    weighted_score=weighted[cand_ix],
                    raw_scores={sig: raw[sig][cand_ix] for sig in raw},
                    normalised_scores={sig: normalised[sig][cand_ix] for sig in normalised},
                    rank_ix=rank_ix,
                    candidate_ix=cand_ix,
                )
            )
        logger.info(
            "hybrid_ranker.rank: n={} signals={} top_weighted={:.3f}",
            n,
            list(self.enabled_signals),
            ranked[0].weighted_score,
        )
        return ranked

    def _judge_score(self, *, persona: Persona, query: str, response: str) -> float:
        """Ask the rerank judge for a 1-5 score on persona consistency."""
        prompt = self._render_judge_prompt(persona=persona, query=query, response=response)
        raw = self.rerank_judge.generate(
            prompt,
            max_new_tokens=self.judge_max_new_tokens,
            temperature=self.judge_temperature,
        )
        return _parse_judge_score(raw)

    @staticmethod
    def _render_judge_prompt(*, persona: Persona, query: str, response: str) -> str:
        """Render the rerank-judge prompt.

        Mirrors the direct-assessment shape used by purpose-built
        evaluators (instruction + response + criteria → ``[RESULT] X``) on
        a 1-5 scale. Kept as a Python concat so changes are diffable and
        testable.
        """
        constraints = (
            "; ".join(persona.identity.constraints) if persona.identity.constraints else "(none)"
        )
        return (
            f"Evaluate how well the following assistant response embodies the stated persona.\n\n"
            f"Persona: {persona.identity.name}, {persona.identity.role}. "
            f"{persona.identity.background.strip()}\n"
            f"Persona constraints (the assistant must NOT violate these): {constraints}\n\n"
            f"User question: {query.strip()}\n\n"
            f"Assistant response:\n{response.strip()}\n\n"
            "Score the response from 1 (off-persona, ignores constraints, generic) to 5 "
            "(strong persona voice, honors constraints, content-appropriate).\n"
            "Answer in exactly this format:\n"
            "[RESULT] <integer between 1 and 5>"
        )


_RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])\b", re.IGNORECASE)
_FALLBACK_INT_RE = re.compile(r"\b([1-5])\b")


def _parse_judge_score(raw: str) -> float:
    """Extract a 1-5 integer score from the judge's response.

    Tolerates surrounding text. Defaults to ``3.0`` (neutral middle of the
    rubric) when the response is malformed — keeps the candidate in the
    ranking pool rather than zeroing it out, which would let one
    parser-glitched candidate dominate the rerank.
    """
    match = _RESULT_RE.search(raw)
    if match is None:
        match = _FALLBACK_INT_RE.search(raw)
    if match is None:
        return 3.0
    try:
        return float(match.group(1))
    except (ValueError, IndexError):
        return 3.0


def _min_max_normalise(values: list[float]) -> list[float]:
    """Min-max normalise to ``[0, 1]``; identical inputs map to all-0.5.

    Identical inputs produce a degenerate range; mapping them to 0.5 keeps
    the candidate set neutral on that signal rather than artificially
    favouring the first or last entry. The ranking is then invariant
    under signal-level scale shifts.
    """
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


__all__ = [
    "DEFAULT_ENABLED_SIGNALS",
    "DEFAULT_HYBRID_WEIGHTS",
    "HybridRanker",
    "RankedCandidate",
]
