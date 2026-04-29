"""Tests for the MiniCheck self-fact contradiction metric."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.minicheck_metric import (
    MiniCheckMetric,
    is_disclaimer,
    is_persona_relevant,
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


# ----- first-person persona-relevance gate -----


@pytest.mark.parametrize(
    "sentence, expected",
    [
        # Real persona claims (no first-person disclaimers).
        ("I have a PhD from ETH", True),
        ("I'm a software engineer", True),
        ("My research focuses on consensus", True),
        ("I've taught for fifteen years", True),
        # Generic factual statements (no first person).
        ("Raft uses leader election", False),
        ("Distributed systems are hard", False),
        ("CAP describes a fundamental tradeoff", False),
        ("The capital of Norway is Oslo", False),
        # AI-disclaimer / hedge / capability statements (first person but
        # not a persona claim).
        ("I might be able to provide more targeted advice", False),
        ("I can't tell you which paper to pick", False),
        ("Let me know if you have more details", False),
        ("I am an AI assistant", False),
        ("I'm just a language model", False),
        ("I don't have access to that information", False),
        ("I'd be happy to help with that", False),
        ("I'll have to think about it", False),
        ("I hope this helps", False),
    ],
)
def test_is_persona_relevant(sentence: str, expected: bool) -> None:
    assert is_persona_relevant(sentence) is expected


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("I might be able to help", True),
        ("I cannot answer that", True),
        ("I don't have access", True),
        ("Let me know more", True),
        ("I'm an AI", True),
        # Real claims should NOT be flagged as disclaimers.
        ("I have a PhD from ETH", False),
        ("My research focuses on consensus", False),
    ],
)
def test_is_disclaimer(sentence: str, expected: bool) -> None:
    assert is_disclaimer(sentence) is expected


def test_b1_disclaimer_false_positives_excluded(cs_tutor: Persona) -> None:
    """Real B1 sentences from response_03 / response_04 inspection should be skipped.

    Both sentences contain first-person pronouns but are disclaimers /
    offers, not factual claims about the persona. Pre-fix they were
    flagged as "contradicts every self-fact" (false positive).
    """
    scorer = _CannedScorer(rules=[], default=0.05)  # would-be contradiction
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "data corruption help?",
                # response_03's flagged sentence
                "Let me know if you have more specific details about your system, and I might be able to provide more targeted advice.",
            ),
            (
                "paper recommendation?",
                # response_04's flagged sentence
                "While I can't tell you which paper to pick, I can give you some pointers based on the provided papers.",
            ),
        ],
    )
    result = metric.score([conv], cs_tutor)
    # Both turns should now score as vacuous 1.0 (no real persona claims).
    assert result.value == pytest.approx(1.0)
    assert result.metadata["turns_with_no_relevant_sentences"] == 2
    assert result.metadata["contradicted_sentences"] == 0


def test_generic_factual_turn_is_not_contradicted(cs_tutor: Persona) -> None:
    """A turn with no first-person claims scores 1.0 even if it doesn't echo any self-fact.

    This is the post-fix behaviour: B1 vanilla-RAG explaining Raft does not
    contradict the persona's self-facts because it isn't claiming anything
    about the persona.
    """
    scorer = _CannedScorer(rules=[], default=0.05)  # would-be contradiction
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "what is Raft?",
                "Raft is a consensus algorithm. It uses leader election. Followers replicate the leader's log.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    assert result.value == pytest.approx(1.0)
    # All 3 sentences exist in total but none are persona-relevant.
    assert result.metadata["total_sentences"] == 3
    assert result.metadata["relevant_sentences"] == 0
    assert result.metadata["turns_with_no_relevant_sentences"] == 1
    assert result.metadata["contradicted_sentences"] == 0


def test_mixed_relevant_and_irrelevant_only_relevant_is_judged(cs_tutor: Persona) -> None:
    """Mixed turn: only persona-relevant sentences count toward the contradiction rate."""
    scorer = _CannedScorer(
        rules=[("PhD", "PhD", 0.9)],  # supports first sentence
        default=0.05,  # everything else "unsupported"
    )
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "tell me about your background and Raft",
                "I have a PhD from ETH. Raft uses leader election. CAP is a tradeoff.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    # 1 persona-relevant sentence, supported by self-facts -> score 1.0.
    # 2 generic sentences ignored.
    assert result.value == pytest.approx(1.0)
    assert result.metadata["total_sentences"] == 3
    assert result.metadata["relevant_sentences"] == 1
    assert result.metadata["contradicted_sentences"] == 0


def test_first_person_contradiction_still_caught(cs_tutor: Persona) -> None:
    """First-person sentences that contradict self-facts are still flagged."""
    scorer = _CannedScorer(rules=[], default=0.05)
    metric = MiniCheckMetric(scorer=scorer)
    conv = _conv(
        "cs_tutor",
        [
            (
                "tell me about yourself",
                "I have no formal training. CAP is a tradeoff in distributed systems.",
            )
        ],
    )
    result = metric.score([conv], cs_tutor)
    # 2 sentences, 1 persona-relevant ("I have no formal training") and contradicted.
    # The CAP sentence is generic -> ignored.
    # Per-turn score: 0/1 = 0.0.
    assert result.value == pytest.approx(0.0)
    assert result.metadata["relevant_sentences"] == 1
    assert result.metadata["contradicted_sentences"] == 1
