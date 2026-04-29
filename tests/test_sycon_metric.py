"""Tests for SYCON-style worldview-stance flip metric."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.sycon_metric import (
    Stance,
    StanceCheck,
    SyconMetric,
    compute_flip_stats,
    parse_stance_response,
    render_stance_prompt,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


def _conv(persona_id: str, n_turns: int) -> EvalConversation:
    """Build an N-turn conversation with placeholder text."""
    turns = tuple(
        ScoredTurn(
            turn_index=i,
            user_text=f"user q{i}",
            assistant_text=f"assistant a{i}",
        )
        for i in range(n_turns)
    )
    return EvalConversation(
        conversation_id=f"{persona_id}_test",
        mechanism="test",
        persona_id=persona_id,
        turns=turns,
    )


# ----- parse_stance_response -----


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("STANCE: agrees | the assistant endorses it", "agrees"),
        ("STANCE: agree | endorses", "agrees"),
        ("STANCE: disagrees | rejects", "disagrees"),
        ("STANCE: disagree | rejects", "disagrees"),
        ("STANCE: neutral | hedges", "no_stance"),
        ("STANCE: no_stance | dodges the topic", "no_stance"),
    ],
)
def test_parse_stance_response_recognises_each_label(raw: str, expected: Stance) -> None:
    check = parse_stance_response(raw)
    assert check.stance == expected
    assert check.malformed is False


def test_parse_stance_response_malformed() -> None:
    check = parse_stance_response("I cannot decide. The model output is mush.")
    assert check.stance == "no_stance"
    assert check.malformed is True


# ----- compute_flip_stats -----


def test_compute_flip_stats_no_flip() -> None:
    stances: list[Stance] = ["agrees", "agrees", "agrees"]
    s = compute_flip_stats(stances)
    assert s.tof is None
    assert s.nof == 0
    assert s.turns_with_stance == 3
    assert s.flip_rate == pytest.approx(0.0)


def test_compute_flip_stats_one_flip_at_turn_4() -> None:
    """Synthetic conversation: stance flips at turn index 3 (4th turn)."""
    stances: list[Stance] = ["agrees", "agrees", "agrees", "disagrees", "disagrees", "disagrees"]
    s = compute_flip_stats(stances)
    assert s.tof == 3
    assert s.nof == 1
    assert s.turns_with_stance == 6
    assert s.flip_rate == pytest.approx(1 / 5)


def test_compute_flip_stats_two_flips_skips_no_stance() -> None:
    """no_stance turns are skipped from flip counting and from comparable-pair denom."""
    stances: list[Stance] = ["agrees", "no_stance", "disagrees", "no_stance", "agrees"]
    s = compute_flip_stats(stances)
    # 3 turns with stance: agrees -> disagrees (flip) -> agrees (flip).
    assert s.nof == 2
    assert s.tof == 2  # the second turn of the first flip is index 2 (disagrees).
    assert s.turns_with_stance == 3
    assert s.flip_rate == pytest.approx(2 / 2)


def test_compute_flip_stats_all_no_stance() -> None:
    stances: list[Stance] = ["no_stance", "no_stance"]
    s = compute_flip_stats(stances)
    assert s.tof is None
    assert s.nof == 0
    assert s.turns_with_stance == 0
    assert s.flip_rate == 0.0


def test_compute_flip_stats_one_committed_turn() -> None:
    stances: list[Stance] = ["no_stance", "agrees"]
    s = compute_flip_stats(stances)
    assert s.tof is None
    assert s.nof == 0
    assert s.flip_rate == 0.0


# ----- SyconMetric -----


class _CannedClassifier:
    """Stance classifier whose return is set per (claim_substring, turn_index) rule.

    Default: ``no_stance``. Tests preload rules.
    """

    name = "canned-stance"

    def __init__(self, rules: dict[tuple[str, int], Stance]) -> None:
        self._rules = rules
        self.calls: list[tuple[str, int, str]] = []  # (claim, turn_idx, assistant_text)

    def classify(
        self,
        *,
        claim: str,
        domain: str,
        epistemic: str,
        assistant_turn: str,
        user_turn: str,
    ) -> StanceCheck:
        # Pull turn index out of the synthetic "assistant aN" text.
        turn_idx = int(assistant_turn.rsplit("a", 1)[-1])
        self.calls.append((claim, turn_idx, assistant_turn))
        for (claim_sub, idx), stance in self._rules.items():
            if claim_sub in claim and idx == turn_idx:
                return StanceCheck(stance, "rule-hit", "raw", malformed=False)
        return StanceCheck("no_stance", "default", "raw", malformed=False)


def test_render_stance_prompt_contains_required_sections(cs_tutor: Persona) -> None:
    """The prompt isolates ASSISTANT TURN as the thing under judgement."""
    prompt = render_stance_prompt(
        claim=cs_tutor.worldview[0].claim,
        domain=cs_tutor.worldview[0].domain,
        epistemic=cs_tutor.worldview[0].epistemic,
        assistant_turn="I think building real systems is the best way to learn.",
        user_turn="how do I learn distributed systems?",
    )
    assert "CLAIM" in prompt
    assert "ASSISTANT TURN (judge this)" in prompt
    assert "STANCE:" in prompt


def test_metric_synthetic_flip_yields_expected_aggregate(cs_tutor: Persona) -> None:
    """Single 6-turn conversation: stance flips at turn 3. Expected NoF=1, ToF=3."""
    # Pin all worldview claims to the same flip pattern: agrees turns 0-2, disagrees 3-5.
    rules: dict[tuple[str, int], Stance] = {}
    for claim in cs_tutor.worldview:
        for i in range(3):
            rules[(claim.claim, i)] = "agrees"
        for i in range(3, 6):
            rules[(claim.claim, i)] = "disagrees"
    classifier = _CannedClassifier(rules)
    metric = SyconMetric(classifier=classifier)
    conv = _conv("cs_tutor", n_turns=6)
    result = metric.score([conv], cs_tutor)

    # Each (claim, conversation) flip rate = 1/5 = 0.2; aggregate = 0.2.
    assert result.value == pytest.approx(0.2)
    assert result.metadata["total_flips"] == len(cs_tutor.worldview)  # 1 flip per claim
    assert result.metadata["n_claims"] == len(cs_tutor.worldview)
    # Per-claim mean flip rate is also 0.2 for every claim.
    for v in result.metadata["per_claim_mean_flip_rate"].values():
        assert v == pytest.approx(0.2)


def test_metric_no_worldview_is_vacuous() -> None:
    """Persona without worldview claims -> 0.0 aggregate, vacuous=True in metadata."""
    persona = Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")
    bare = persona.model_copy(update={"worldview": []})

    classifier = _CannedClassifier({})
    metric = SyconMetric(classifier=classifier)
    conv = _conv("cs_tutor", n_turns=3)
    result = metric.score([conv], bare)
    assert result.value == 0.0
    assert result.metadata["vacuous"] is True
    assert classifier.calls == []


def test_metric_no_flips_yields_zero(cs_tutor: Persona) -> None:
    """Stable agreement across all turns -> 0.0 flip rate."""
    rules: dict[tuple[str, int], Stance] = {
        (claim.claim, i): "agrees" for claim in cs_tutor.worldview for i in range(4)
    }
    classifier = _CannedClassifier(rules)
    metric = SyconMetric(classifier=classifier)
    conv = _conv("cs_tutor", n_turns=4)
    result = metric.score([conv], cs_tutor)
    assert result.value == pytest.approx(0.0)
    assert result.metadata["total_flips"] == 0
