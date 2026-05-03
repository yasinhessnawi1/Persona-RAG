"""Tests for the Reciprocal Rank Fusion helper."""

from __future__ import annotations

import pytest

from option8_rag.retrieve.bm25 import reciprocal_rank_fusion
from option8_rag.types import Chunk, RetrievedChunk


def _r(chunk_id: str, rank_score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text="", index=0),
        score=rank_score,
    )


def test_rrf_single_list_preserves_order() -> None:
    list_a = [[_r("a"), _r("b"), _r("c")]]
    fused = reciprocal_rank_fusion(list_a, top_k=3, rrf_k=60)
    assert [c.chunk.chunk_id for c in fused[0]] == ["a", "b", "c"]


def test_rrf_two_lists_boost_shared_doc() -> None:
    list_a = [[_r("x"), _r("y"), _r("z")]]
    list_b = [[_r("y"), _r("w"), _r("x")]]
    fused = reciprocal_rank_fusion(list_a, list_b, top_k=4, rrf_k=60)
    ranking = [c.chunk.chunk_id for c in fused[0]]
    # y (rank 2 + 1) and x (rank 1 + 3) should be top-2; the exact order
    # depends on the RRF tie-breaker but both must beat lone-list w and z.
    assert set(ranking[:2]) == {"x", "y"}
    assert ranking[2:] in (["w", "z"], ["z", "w"])


def test_rrf_mismatched_query_dims_raises() -> None:
    list_a = [[_r("a")]]
    list_b = [[_r("a")], [_r("b")]]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(list_a, list_b, top_k=3)


def test_rrf_top_k_truncation() -> None:
    list_a = [[_r("a"), _r("b"), _r("c"), _r("d")]]
    fused = reciprocal_rank_fusion(list_a, top_k=2, rrf_k=60)
    assert len(fused[0]) == 2
