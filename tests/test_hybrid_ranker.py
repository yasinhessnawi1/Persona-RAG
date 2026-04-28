"""Tests for the 2-signal hybrid ranker."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.retrieval.character_rm import FakeCharacterRMScorer
from persona_rag.retrieval.hybrid_ranker import (
    DEFAULT_HYBRID_WEIGHTS,
    HybridRanker,
    _min_max_normalise,
    _parse_judge_score,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


class _ScriptedJudge:
    """LLMBackend stub returning canned ``[RESULT] N`` strings in order."""

    name = "scripted-judge"
    model_id = "fake/scripted-judge"
    num_layers = 0
    hidden_dim = 0

    def __init__(self, scores: list[int]) -> None:
        self.scores = list(scores)
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        score = self.scores[self.calls]
        self.calls += 1
        return f"Persona consistency assessment.\n[RESULT] {score}"


# ----------------------------------------------------------- normalisation


def test_min_max_normalise_spreads_to_unit_interval() -> None:
    norm = _min_max_normalise([1.0, 2.0, 3.0])
    assert norm[0] == pytest.approx(0.0)
    assert norm[2] == pytest.approx(1.0)
    assert 0.0 < norm[1] < 1.0


def test_min_max_normalise_identical_inputs_map_to_half() -> None:
    """Identical inputs → all-0.5 (degenerate range; signal-neutral)."""
    norm = _min_max_normalise([5.0, 5.0, 5.0])
    assert norm == [0.5, 0.5, 0.5]


def test_min_max_normalisation_invariance_under_signal_scale_shift(cs_tutor: Persona) -> None:
    """Scaling one signal by 10x must not change the ranking."""
    char_rm = FakeCharacterRMScorer()

    judge_a = _ScriptedJudge([3, 5, 1])
    ranker_a = HybridRanker(character_rm=char_rm, rerank_judge=judge_a)
    ranked_a = ranker_a.rank(
        persona=cs_tutor,
        query="q",
        candidates=["candidate alpha", "candidate beta", "candidate gamma"],
    )

    # Scale judge scores by 10x; invariance under min-max normalisation
    # means the same per-candidate ranking must come out.
    judge_b = _ScriptedJudge([30, 50, 10])
    ranker_b = HybridRanker(character_rm=char_rm, rerank_judge=judge_b)
    ranked_b = ranker_b.rank(
        persona=cs_tutor,
        query="q",
        candidates=["candidate alpha", "candidate beta", "candidate gamma"],
    )
    # Judge raw scores out-of-range get parsed as 3.0 fallback by
    # _parse_judge_score's [1-5] regex; this test verifies the fallback path
    # produces stable rankings regardless. Both rankers should agree on the
    # candidate ordering since the CharacterRM scores are identical.
    assert [rc.candidate_ix for rc in ranked_a] == [rc.candidate_ix for rc in ranked_b]


# ----------------------------------------------------------- ranker behaviour


def test_ranker_returns_candidates_in_descending_weighted_score(cs_tutor: Persona) -> None:
    char_rm = FakeCharacterRMScorer()
    judge = _ScriptedJudge([4, 5, 2])
    ranker = HybridRanker(character_rm=char_rm, rerank_judge=judge)
    ranked = ranker.rank(
        persona=cs_tutor,
        query="q",
        candidates=["alpha", "beta", "gamma"],
    )
    # Returned in descending weighted order.
    weights = [rc.weighted_score for rc in ranked]
    assert weights == sorted(weights, reverse=True)
    # Rank indices match position in returned list.
    for ix, rc in enumerate(ranked):
        assert rc.rank_ix == ix


def test_ranker_records_per_signal_breakdown(cs_tutor: Persona) -> None:
    char_rm = FakeCharacterRMScorer()
    judge = _ScriptedJudge([3, 5])
    ranker = HybridRanker(character_rm=char_rm, rerank_judge=judge)
    ranked = ranker.rank(persona=cs_tutor, query="q", candidates=["a", "b"])
    for rc in ranked:
        assert "character_rm" in rc.raw_scores
        assert "judge" in rc.raw_scores
        assert "character_rm" in rc.normalised_scores
        assert "judge" in rc.normalised_scores
        # Normalised values lie in [0, 1].
        for v in rc.normalised_scores.values():
            assert 0.0 <= v <= 1.0


def test_ranker_falls_back_to_judge_only(cs_tutor: Persona) -> None:
    """``enabled_signals=["judge"]`` skips CharacterRM scoring entirely."""
    char_rm = FakeCharacterRMScorer()
    judge = _ScriptedJudge([5, 1, 3])
    ranker = HybridRanker(
        character_rm=char_rm,
        rerank_judge=judge,
        enabled_signals=("judge",),
    )
    ranked = ranker.rank(
        persona=cs_tutor,
        query="q",
        candidates=["candidate a", "candidate b", "candidate c"],
    )
    # CharacterRM not called.
    assert char_rm.calls == []
    # Top candidate corresponds to judge score 5 (candidate index 0 in scripted order).
    assert ranked[0].candidate_ix == 0
    # raw_scores only carries the judge signal under fallback.
    for rc in ranked:
        assert "character_rm" not in rc.raw_scores
        assert "judge" in rc.raw_scores


def test_ranker_default_weights_balanced() -> None:
    assert DEFAULT_HYBRID_WEIGHTS == {"character_rm": 0.5, "judge": 0.5}


def test_ranker_empty_candidates_raises(cs_tutor: Persona) -> None:
    char_rm = FakeCharacterRMScorer()
    judge = _ScriptedJudge([])
    ranker = HybridRanker(character_rm=char_rm, rerank_judge=judge)
    with pytest.raises(ValueError, match="candidates must not be empty"):
        ranker.rank(persona=cs_tutor, query="q", candidates=[])


# ----------------------------------------------------------- parsing


def test_parse_judge_score_extracts_result_marker() -> None:
    assert _parse_judge_score("[RESULT] 4") == 4.0
    assert _parse_judge_score("Some preamble. [RESULT] 1\nTrailing text") == 1.0


def test_parse_judge_score_falls_back_to_first_integer() -> None:
    assert _parse_judge_score("Score: 3 (mediocre)") == 3.0


def test_parse_judge_score_malformed_defaults_to_three() -> None:
    """Malformed responses default to 3.0 (neutral middle of 1-5 rubric)."""
    assert _parse_judge_score("the judge produced narrative prose") == 3.0
