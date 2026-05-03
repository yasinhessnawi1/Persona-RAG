"""Hybrid retrieval: dense + BM25 fused via Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence

from option8_rag.retrieve.bm25 import Bm25Retriever, reciprocal_rank_fusion
from option8_rag.retrieve.dense import DenseRetriever
from option8_rag.types import Query, RetrievedChunk


class HybridRetriever:
    """Hybrid retrieval combining a dense and a BM25 branch via RRF.

    Args:
        dense: Configured dense retriever.
        bm25: Configured BM25 retriever (over the same chunk corpus).
        top_k: Default number of fused results per query.
        dense_top_n: Per-query candidate-pool size for the dense branch.
        bm25_top_n: Per-query candidate-pool size for the BM25 branch.
        rrf_k: RRF k constant; 60 is the standard default.
    """

    def __init__(
        self,
        *,
        dense: DenseRetriever,
        bm25: Bm25Retriever,
        top_k: int = 10,
        dense_top_n: int = 100,
        bm25_top_n: int = 100,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.top_k = top_k
        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k

    def retrieve(
        self,
        queries: Sequence[Query],
        *,
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Return RRF-fused top-k chunks per query."""

        k = top_k if top_k is not None else self.top_k
        if not queries:
            return []

        dense_results = self.dense.retrieve(queries, top_k=self.dense_top_n)
        bm25_results = self.bm25.retrieve(queries, top_k=self.bm25_top_n)

        return reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=k,
            rrf_k=self.rrf_k,
        )
