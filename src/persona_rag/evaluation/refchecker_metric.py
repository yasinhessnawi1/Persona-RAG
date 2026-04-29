"""RefChecker triplet contradiction metric (soft-optional).

RefChecker (https://github.com/amazon-science/RefChecker) was archived
2026-04-08; the last release is v0.2.17. The repo is no longer
actively maintained. We treat it as a *secondary* metric -- MiniCheck
is the primary self-fact contradiction column -- and gate the import
behind a try/except so the harness still runs cleanly when RefChecker
is not installed.

Install (server-only, optional):

    uv pip install refchecker spacy
    uv run python -m spacy download en_core_web_sm

If installed, the metric:

1. Decomposes each assistant turn into (subject, predicate, object)
   triplets via RefChecker's ``LLMExtractor`` (LLM-backed, can use any
   OpenAI-compatible local model).
2. Checks each triplet against the persona's self_facts + worldview
   via the local ``NLIChecker`` (no API key required).
3. Reports per-triplet contradiction rate.

If not installed, the metric is a no-op that returns NaN with
``available=False`` in metadata. The runner can skip RefChecker rows
without crashing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation, MetricResult
from persona_rag.schema.persona import Persona


def is_refchecker_available() -> bool:
    """Return True iff the ``refchecker`` package can be imported."""
    try:
        import refchecker  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class RefCheckerMetric:
    """Triplet-level contradiction metric.

    Soft-optional: returns NaN with ``available=False`` metadata when
    the package isn't installed. Construction does not import
    ``refchecker``; only ``score`` does, so importing this module on a
    minimal install is safe.

    ``extractor_factory`` and ``checker_factory`` are zero-arg builders
    for RefChecker's extractor / checker. Defaults use the local
    NLIChecker (no API keys); callers can substitute LLM-backed
    extractors at runtime.
    """

    extractor_factory: Any | None = None
    checker_factory: Any | None = None
    contradiction_label: str = "Contradiction"
    name: str = "refchecker_triplet_contradiction"
    persona_reference_text: str = field(init=False, default="")

    def _build_persona_reference(self, persona: Persona) -> str:
        """One reference text covering self_facts + worldview claims."""
        lines = [f"This persona is named {persona.identity.name}, {persona.identity.role}."]
        for sf in persona.self_facts:
            lines.append(sf.fact)
        for w in persona.worldview:
            lines.append(w.claim)
        return "\n".join(lines)

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        """Run RefChecker if installed, else return NaN with available=False."""
        if not is_refchecker_available():
            logger.warning(
                "refchecker not installed -- RefCheckerMetric returning NaN. "
                "Install with: uv pip install refchecker spacy"
            )
            return MetricResult(
                name=self.name,
                value=float("nan"),
                per_persona={persona.persona_id or "<unknown>": float("nan")},
                metadata={"available": False, "reason": "refchecker not installed"},
            )

        try:
            from refchecker.checker import NLIChecker
            from refchecker.extractor import LLMExtractor
        except ImportError as exc:
            logger.warning("refchecker imports failed: {}", exc)
            return MetricResult(
                name=self.name,
                value=float("nan"),
                per_persona={persona.persona_id or "<unknown>": float("nan")},
                metadata={"available": False, "reason": str(exc)},
            )

        extractor = self.extractor_factory() if self.extractor_factory else LLMExtractor()
        checker = self.checker_factory() if self.checker_factory else NLIChecker()
        reference = self._build_persona_reference(persona)

        per_conv_rates: list[float] = []
        per_conv_ids: list[str] = []
        total_triplets = 0
        contradicted = 0

        for conv in conversations:
            conv_triplets = 0
            conv_contradicted = 0
            for turn in conv.turns:
                # ``extract`` API: returns a list of claim triplets per response.
                ext = extractor.extract(
                    batch_responses=[turn.assistant_text],
                    batch_questions=[turn.user_text],
                )
                # RefChecker returns extraction objects; pull the triplets out.
                turn_triplets = self._flatten_triplets(ext)
                if not turn_triplets:
                    continue
                labels = checker.check(
                    batch_claims=[turn_triplets],
                    batch_references=[[reference] * len(turn_triplets)],
                )
                conv_triplets += len(turn_triplets)
                for label_set in labels:
                    for label in label_set:
                        if str(label).lower() == self.contradiction_label.lower():
                            conv_contradicted += 1
            total_triplets += conv_triplets
            contradicted += conv_contradicted
            rate = (conv_contradicted / conv_triplets) if conv_triplets else 0.0
            per_conv_rates.append(rate)
            per_conv_ids.append(conv.conversation_id)

        aggregate = statistics.fmean(per_conv_rates) if per_conv_rates else 0.0
        return MetricResult(
            name=self.name,
            value=aggregate,
            per_conversation=per_conv_rates,
            per_conversation_ids=per_conv_ids,
            per_persona={persona.persona_id or "<unknown>": aggregate},
            metadata={
                "available": True,
                "n_conversations": len(conversations),
                "total_triplets": total_triplets,
                "contradicted_triplets": contradicted,
            },
        )

    @staticmethod
    def _flatten_triplets(extraction_result: Any) -> list[str]:
        """Pull ``[subject, predicate, object]`` triplets out of RefChecker's extraction shape.

        RefChecker's API has shifted across versions; we accept a few
        common shapes (list of claim objects with ``.content``,
        list-of-lists, or list of dicts).
        """
        triplets: list[str] = []
        if extraction_result is None:
            return triplets
        for batch_item in extraction_result:
            claims = getattr(batch_item, "claims", batch_item)
            if not claims:
                continue
            for claim in claims:
                if hasattr(claim, "content"):
                    triplets.append(str(claim.content))
                elif isinstance(claim, dict) and "content" in claim:
                    triplets.append(str(claim["content"]))
                elif isinstance(claim, list) and len(claim) == 3:
                    triplets.append(" ".join(str(x) for x in claim))
                else:
                    triplets.append(str(claim))
        return triplets


__all__ = ["RefCheckerMetric", "is_refchecker_available"]
