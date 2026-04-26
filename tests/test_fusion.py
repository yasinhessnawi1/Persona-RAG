"""Tests for retrieval fusion: RRF + weighted-sum."""

from __future__ import annotations

import pytest

from persona_rag.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
    weighted_sum_fusion,
)

# ----------------------------------------------------------------- RRF


def test_default_rrf_k_is_60_per_cormack_2009() -> None:
    """The default RRF k is 60 (Cormack 2009 / Elastic / LlamaIndex)."""
    assert DEFAULT_RRF_K == 60


def test_rrf_single_ranking_returns_sorted_by_rank() -> None:
    """One ranking → fused order matches the ranking."""
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_rrf_two_rankings_promotes_overlap() -> None:
    """Documents in both rankings score higher than documents in only one."""
    dense = ["a", "b", "c"]
    bm25 = ["b", "d", "e"]
    fused = reciprocal_rank_fusion([dense, bm25])
    # b appears in both; should be top.
    assert fused[0][0] == "b"


def test_rrf_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_rrf_top_k_truncates() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], top_k=2)
    assert len(fused) == 2


def test_rrf_score_formula() -> None:
    """Scores follow 1/(k + rank); single ranking, document 'a' at rank 1."""
    fused = reciprocal_rank_fusion([["a"]], k=60)
    assert fused[0][0] == "a"
    assert fused[0][1] == pytest.approx(1.0 / (60 + 1))


def test_rrf_exact_term_beats_semantic_only() -> None:
    """Synthetic: hybrid promotes a doc only present in BM25.

    Construct the case where BM25 catches an exact-term match (only the
    rare-term doc, ranked first) and dense misses it. The fused output must
    include the rare-term doc above documents present only in dense.
    """
    dense = ["semantic_top", "noise_a", "noise_b"]
    bm25 = ["rare_term_doc", "semantic_top", "noise_a"]
    fused = reciprocal_rank_fusion([dense, bm25], top_k=3)
    fused_ids = [doc_id for doc_id, _ in fused]
    assert "rare_term_doc" in fused_ids
    # The doc present in both rankings (semantic_top) should out-score the
    # bm25-only doc — RRF rewards overlap over depth-1 BM25 presence.
    assert fused_ids.index("semantic_top") < fused_ids.index("rare_term_doc")


# ----------------------------------------------------------------- weighted-sum


def test_weighted_sum_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        weighted_sum_fusion([], [], alpha=1.5)


def test_weighted_sum_alpha_one_picks_dense() -> None:
    """alpha=1.0 → dense-only ordering."""
    dense = [("a", 0.9), ("b", 0.1)]
    bm25 = [("c", 1.0), ("a", 0.0)]
    fused = weighted_sum_fusion(dense, bm25, alpha=1.0)
    assert fused[0][0] == "a"


def test_weighted_sum_alpha_zero_picks_bm25() -> None:
    """alpha=0.0 → bm25-only ordering."""
    dense = [("a", 0.9), ("b", 0.1)]
    bm25 = [("c", 1.0), ("a", 0.0)]
    fused = weighted_sum_fusion(dense, bm25, alpha=0.0)
    assert fused[0][0] == "c"


def test_weighted_sum_top_k_truncates() -> None:
    dense = [("a", 0.9), ("b", 0.1), ("c", 0.05)]
    bm25 = [("d", 1.0)]
    fused = weighted_sum_fusion(dense, bm25, alpha=0.5, top_k=2)
    assert len(fused) == 2


def test_weighted_sum_handles_constant_scores_without_div_zero() -> None:
    """min == max edge case: every doc gets normalised score 1.0, ordering stable."""
    dense = [("a", 0.5), ("b", 0.5)]
    fused = weighted_sum_fusion(dense, [], alpha=1.0)
    # Both have equal normalised score; tie-break by doc_id ascending.
    assert [doc_id for doc_id, _ in fused] == ["a", "b"]
