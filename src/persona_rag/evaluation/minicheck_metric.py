"""Sentence-level supported / unsupported scoring against persona self-facts.

Uses the ``lytang/MiniCheck-Flan-T5-Large`` checkpoint loaded directly via
``AutoModelForSeq2SeqLM`` rather than the official ``minicheck`` wrapper.
The wrapper pulls ``vllm`` and ``nltk``; the model itself just needs
``transformers`` + ``torch`` plus a regex sentence splitter.

Input format the model was trained on (verified from the upstream
inference module):

    "predict: " + tokenizer.eos_token.join([document, claim])

Output: the first decoder logit. The two relevant token ids are 209
("supported") and 3 ("unsupported"). Softmax over those two indices
gives ``p(supported)``; the binary label is ``p >= 0.5``.

Per assistant turn, every sentence is scored against every self-fact
of the persona; the per-turn score is the fraction of sentences that
are *not* contradicted by *any* self-fact (``min p(supported)`` across
self-facts must clear 0.5). Higher is better.

Edge cases:

- Empty self-facts -> per-turn score 1.0 (nothing to contradict).
- Very short turn (< 10 chars after strip) -> still scored, flagged in
  metadata as ``short_turn=True``.
- No alphabetic content -> score 1.0 with ``empty_turn=True``.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation, MetricResult
from persona_rag.schema.persona import Persona

DEFAULT_MINICHECK_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"

# Decoder-token ids the upstream MiniCheck inference uses.
_TOKEN_ID_UNSUPPORTED = 3
_TOKEN_ID_SUPPORTED = 209

# Sentence splitter: greedy on terminal punctuation, dot-included. Permissive
# enough that turns without sentence punctuation still produce one "sentence"
# (the whole turn). Avoids spaCy / nltk for a tiny dependency footprint.
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
_MIN_SENTENCE_CHARS = 4
_SHORT_TURN_THRESHOLD = 10


def split_sentences(text: str) -> list[str]:
    """Split a turn into sentence-level claims.

    The MiniCheck model expects one claim per call; passing a multi-
    sentence turn confuses the supported/unsupported readout. Splits
    on terminal punctuation and discards fragments under
    ``_MIN_SENTENCE_CHARS`` (whitespace, stray punctuation).

    Returns at least one sentence -- for a turn with no terminal
    punctuation, the whole turn is one sentence.
    """
    candidates = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)]
    sentences = [s for s in candidates if len(s) >= _MIN_SENTENCE_CHARS]
    if not sentences:
        stripped = text.strip()
        if stripped:
            return [stripped]
        return []
    return sentences


# ---------------------------------------------------------------------------
# Scorer protocol + concrete HuggingFace implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class MiniCheckScorer(Protocol):
    """Score one (document, claim) pair -> ``p(supported)`` in [0, 1]."""

    name: str

    def score(self, document: str, claim: str) -> float:
        """Return ``p(supported)`` for this (document, claim) pair."""

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Batched variant for throughput. Default impl is the sequential loop."""


@dataclass
class HFMiniCheckScorer:
    """Direct ``AutoModelForSeq2SeqLM`` MiniCheck scorer.

    Lazy-loads on first ``score`` call so importing this module on a
    CPU-only Mac does not trigger model download. Pass ``device="cpu"``
    explicitly to force CPU inference.
    """

    model_id: str = DEFAULT_MINICHECK_MODEL_ID
    device: str = "cpu"
    batch_size: int = 8
    max_input_tokens: int = 1024
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"minicheck:{self.model_id.split('/')[-1].lower()}"
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Local import -- keeps the module importable on machines where
        # transformers is absent (the metric is then unusable, but the
        # rest of the evaluation harness still loads).
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("MiniCheck loading {} on {}", self.model_id, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def _format(self, document: str, claim: str) -> str:
        # Mirrors upstream: "predict: <doc><eos><claim>". The Flan-T5 base
        # was instruction-tuned with the "predict:" prefix.
        eos = self._tokenizer.eos_token or "</s>"
        return "predict: " + eos.join([document, claim])

    def score(self, document: str, claim: str) -> float:
        """Single-pair scoring. Uses the batched path with batch=1."""
        return self.score_batch([(document, claim)])[0]

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Batch-scoring entry point. Returns ``p(supported)`` per pair."""
        self._ensure_loaded()
        torch = self._torch
        tok = self._tokenizer
        model = self._model

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = tok(
                [self._format(d, c) for d, c in batch],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
            ).to(self.device)
            decoder_input_ids = torch.full(
                (len(batch), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                out = model(
                    **inputs,
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True,
                )
            # First decoder position logit, restricted to the two relevant ids.
            two = out.logits[:, 0, [_TOKEN_ID_UNSUPPORTED, _TOKEN_ID_SUPPORTED]]
            probs = torch.softmax(two, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(float(p) for p in probs)
        return scores


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


@dataclass
class MiniCheckMetric:
    """Self-fact contradiction-rate metric.

    Per assistant turn:

    1. Split the turn into sentences.
    2. For each sentence cross each self-fact, get ``p(supported)``.
    3. Sentence is "contradicted" if ``max p(supported) < threshold``
       across all self-facts (no self-fact supports it AND it sits in
       persona-fact space -- i.e. the model thinks every self-fact
       contradicts it).
    4. Per-turn score = fraction of sentences NOT contradicted.

    Per-conversation score = mean across assistant turns.
    Aggregate value = mean across conversations.
    Higher is better (1.0 = no contradictions).

    The reading is conservative: a sentence is only flagged as a
    contradiction if NO self-fact supports it. Sentences that are about
    unrelated topics (the model says "unsupported" because there's
    nothing to ground them) ARE flagged -- this is the conservative side
    of the cost / sensitivity tradeoff documented in A1's notes. The
    sanity test in ``docs/notes/minicheck_sanity.md`` is the empirical
    check on this convention.
    """

    scorer: MiniCheckScorer
    threshold: float = 0.5
    name: str = "minicheck_self_fact_contradiction"

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        """Compute MiniCheck contradiction rate across ``conversations``."""
        self_facts = [sf.fact for sf in persona.self_facts]
        per_conv_scores: list[float] = []
        per_conv_ids: list[str] = []
        meta_short = 0
        meta_empty = 0
        meta_total_sentences = 0
        meta_contradicted = 0

        for conv in conversations:
            if not conv.turns:
                logger.debug("MiniCheck: empty conversation {}", conv.conversation_id)
                per_conv_scores.append(1.0)
                per_conv_ids.append(conv.conversation_id)
                continue

            turn_scores: list[float] = []
            for turn in conv.turns:
                turn_text = turn.assistant_text or ""
                if len(turn_text.strip()) < _SHORT_TURN_THRESHOLD:
                    meta_short += 1
                if not self_facts:
                    turn_scores.append(1.0)
                    continue
                sentences = split_sentences(turn_text)
                if not sentences:
                    meta_empty += 1
                    turn_scores.append(1.0)
                    continue
                pairs = [(sf, sentence) for sentence in sentences for sf in self_facts]
                probs = self.scorer.score_batch(pairs)
                # Reshape probs into (n_sentences, n_self_facts) and apply
                # the "no self-fact supports it" rule per sentence.
                n_facts = len(self_facts)
                ok_count = 0
                for i in range(len(sentences)):
                    fact_probs = probs[i * n_facts : (i + 1) * n_facts]
                    if max(fact_probs) >= self.threshold:
                        ok_count += 1
                meta_total_sentences += len(sentences)
                meta_contradicted += len(sentences) - ok_count
                turn_scores.append(ok_count / len(sentences) if sentences else 1.0)
            per_conv_scores.append(statistics.fmean(turn_scores) if turn_scores else 1.0)
            per_conv_ids.append(conv.conversation_id)

        aggregate = statistics.fmean(per_conv_scores) if per_conv_scores else 1.0
        return MetricResult(
            name=self.name,
            value=aggregate,
            per_conversation=per_conv_scores,
            per_conversation_ids=per_conv_ids,
            per_persona={persona.persona_id or "<unknown>": aggregate},
            metadata={
                "scorer": self.scorer.name,
                "threshold": self.threshold,
                "n_conversations": len(conversations),
                "n_self_facts": len(self_facts),
                "short_turns": meta_short,
                "empty_turns": meta_empty,
                "total_sentences": meta_total_sentences,
                "contradicted_sentences": meta_contradicted,
            },
        )


__all__ = [
    "DEFAULT_MINICHECK_MODEL_ID",
    "HFMiniCheckScorer",
    "MiniCheckMetric",
    "MiniCheckScorer",
    "split_sentences",
]
