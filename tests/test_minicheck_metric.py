"""Tests for the MiniCheck self-fact contradiction metric."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.minicheck_metric import (
    MiniCheckMetric,
    split_sentences,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


class _CannedScorer:
    """Returns a pre-set ``p(supported)`` per (document, claim) pair.

    Pairs are matched by ``(document_substring, claim_substring)`` -- if
    both substrings appear, the canned score wins; otherwise default
    ``0.0`` (treated as contradiction).
    """

    name = "canned-minicheck"

    def __init__(self, rules: list[tuple[str, str, float]], default: float = 0.0) -> None:
        self._rules = rules
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def score(self, document: str, claim: str) -> float:
        return self.score_batch([(document, claim)])[0]

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for doc, claim in pairs:
            self.calls.append((doc, claim))
            score = self._default
            for d_sub, c_sub, p in self._rules:
                if d_sub in doc and c_sub in claim:
                    score = p
                    break
            out.append(score)
        return out


def test_split_sentences_basic() -> None:
    text = "I tutor students. I have a PhD! Do I really? Yes."
    out = split_sentences(text)
    assert len(out) == 4
    assert out[0].endswith(".")
    assert out[2].endswith("?")


def test_split_sentences_no_terminal_punctuation() -> None:
    out = split_sentences("just one fragment with no period")
    assert out == ["just one fragment with no period"]


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def _conv(persona_id: str, turns: list[tuple[str, str]]) -> EvalConversation:
    return EvalConversation(
        conversation_id=f"{persona_id}_test",
        mechanism="test",
        persona_id=persona_id,
        turns=tuple(
            ScoredTurn(turn_index=i, user_text=u, assistant_text=a)
            for i, (u, a) in enumerate(turns)
        ),
    )


def test_clear_agreement_scores_high(cs_tutor: Persona) -> None:
    """Hand-constructed example #1: clear agreement -> score ≈ 1.0."""
    # Every sentence is supported by at least one self-fact.
    scorer = _CannedScorer(
        rules=[
            ("PhD", "PhD", 0.95),
            ("teaching", "teach", 0.92),
            ("backend engineer", "backend", 0.90),
        ],
        default=0.05,
    )
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "tell me about yourself",
                "I have a PhD from ETH. I teach distributed systems. I worked as a backend engineer.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    assert result.value == pytest.approx(1.0)
    assert result.per_conversation == [pytest.approx(1.0)]
    assert result.metadata["contradicted_sentences"] == 0


def test_clear_contradiction_scores_low(cs_tutor: Persona) -> None:
    """Hand-constructed example #2: clear contradiction -> score ≈ 0.0."""
    # All facts ALWAYS reject these claims (default 0.05 < 0.5 threshold).
    scorer = _CannedScorer(rules=[], default=0.05)
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "tell me about yourself",
                "I have no formal training. I have never taught anyone. I work in marketing.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    assert result.value == pytest.approx(0.0)
    assert result.metadata["contradicted_sentences"] == 3


def test_ambiguous_scores_mid(cs_tutor: Persona) -> None:
    """Hand-constructed example #3: ambiguous unrelated -> mid range.

    Some sentences supported by some fact, others unsupported. The
    metric reports the fraction not contradicted across the turn.
    """
    scorer = _CannedScorer(
        rules=[
            # Sentence about Zurich gets supported by the location fact.
            ("Zurich", "Zurich", 0.85),
        ],
        default=0.10,
    )
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "where do you live?",
                # 2 sentences: 1 supported (Zurich), 1 unrelated (cooking).
                "I am based in Zurich. I really love cooking pasta on weekends.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    # 1 of 2 sentences ok -> 0.5
    assert result.value == pytest.approx(0.5)


def test_empty_self_facts_returns_one(cs_tutor: Persona) -> None:
    """With no self_facts on the persona, every turn is uncontradictable."""
    bare = cs_tutor.model_copy(update={"self_facts": []})
    scorer = _CannedScorer(rules=[], default=0.0)
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [("anything?", "I am a unicorn.")],
    )
    result = metric.score([conv], bare)
    assert result.value == pytest.approx(1.0)
    assert scorer.calls == []  # Scorer never invoked when no facts.


def test_empty_conversation_lists() -> None:
    """No conversations -> aggregate 1.0 (vacuous), no per_conversation entries."""
    persona = Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")
    scorer = _CannedScorer(rules=[], default=0.0)
    metric = MiniCheckMetric(scorer=scorer)
    result = metric.score([], persona)
    assert result.value == pytest.approx(1.0)
    assert result.per_conversation == []


def test_short_turn_flagged(cs_tutor: Persona) -> None:
    """A < 10-char turn is still scored but flagged in metadata."""
    scorer = _CannedScorer(rules=[("PhD", "ok", 0.9)], default=0.05)
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [("?", "ok.")],
    )
    result = metric.score([conv], cs_tutor)
    assert result.metadata["short_turns"] == 1


def test_per_persona_breakdown_populated(cs_tutor: Persona) -> None:
    """Per-persona dict has one entry equal to the aggregate."""
    scorer = _CannedScorer(rules=[("PhD", "PhD", 0.9)], default=0.95)
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv("cs_tutor", [("hi", "I have a PhD.")])
    result = metric.score([conv], cs_tutor)
    assert "cs_tutor" in result.per_persona
    assert result.per_persona["cs_tutor"] == pytest.approx(result.value)
