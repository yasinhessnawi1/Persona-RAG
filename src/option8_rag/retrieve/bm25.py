"""BM25 retrieval over chunk texts using ``bm25s``.

Tokenisation is centralised so the index leg and the query leg cannot drift.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from loguru import logger

from option8_rag.types import Chunk, Query, RetrievedChunk


def _tokenize(texts: Iterable[str], *, lowercase: bool) -> list[list[str]]:
    import bm25s

    out = bm25s.tokenize(
        list(texts),
        lower=lowercase,
        stopwords="en",
        return_ids=False,
    )
    return list(out)


class Bm25Retriever:
    """In-memory BM25 retriever over a chunk corpus.

    Args:
        chunks: Chunks to index.
        top_k: Default number of results per query.
        lowercase: Whether to lowercase tokens.
    """

    def __init__(
        self,
        *,
        chunks: Sequence[Chunk],
        top_k: int = 10,
        lowercase: bool = True,
    ) -> None:
        import bm25s

        self.chunks = list(chunks)
        self.top_k = top_k
        self.lowercase = lowercase

        if not self.chunks:
            raise ValueError("Bm25Retriever requires at least one chunk")

        tokens = _tokenize((c.text for c in self.chunks), lowercase=lowercase)
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens)
        logger.info("BM25 indexed {n} chunks (lowercase={lc})", n=len(self.chunks), lc=lowercase)

    def retrieve(
        self,
        queries: Sequence[Query],
        *,
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Return top-k retrieved chunks per query, ordered by descending BM25 score."""

        k = top_k if top_k is not None else self.top_k
        if not queries:
            return []

        q_tokens = _tokenize((q.text for q in queries), lowercase=self.lowercase)
        # bm25s.retrieve returns (docs, scores) or per-query (indices, scores).
        # The scalar k case below uses the indices form.
        results, scores = self._retriever.retrieve(q_tokens, k=k)

        out: list[list[RetrievedChunk]] = []
        for row_idx in range(len(queries)):
            row: list[RetrievedChunk] = []
            for col in range(min(k, len(results[row_idx]))):
                idx = int(results[row_idx][col])
                score = float(scores[row_idx][col])
                row.append(RetrievedChunk(chunk=self.chunks[idx], score=score))
            out.append(row)
        return out

    @property
    def num_indexed(self) -> int:
        return len(self.chunks)


def reciprocal_rank_fusion(
    *ranked_lists: list[list[RetrievedChunk]],
    top_k: int,
    rrf_k: int = 60,
) -> list[list[RetrievedChunk]]:
    """Fuse multiple ranked-lists per query via Reciprocal Rank Fusion.

    Score for a chunk c at rank r in list i is ``1 / (rrf_k + r)`` (1-indexed).
    Final score is the sum across lists. The top-k by fused score is returned
    per query, ties broken by the chunk_id ordering.

    Args:
        ranked_lists: One list-of-list-of-RetrievedChunk per retrieval branch.
            All inputs must have the same outer length (one row per query).
        top_k: Number of results per query.
        rrf_k: RRF k constant; defaults to 60 per Cormack et al. (2009).

    Returns:
        Fused ranked list per query.
    """

    if not ranked_lists:
        return []
    n_queries = len(ranked_lists[0])
    for rl in ranked_lists:
        if len(rl) != n_queries:
            raise ValueError(
                "all ranked lists must share the same number of queries; "
                f"got {[len(r) for r in ranked_lists]}",
            )

    fused: list[list[RetrievedChunk]] = []
    for qi in range(n_queries):
        accum: dict[str, tuple[float, RetrievedChunk]] = {}
        for rl in ranked_lists:
            for rank, item in enumerate(rl[qi], start=1):
                contribution = 1.0 / (rrf_k + rank)
                prev = accum.get(item.chunk.chunk_id)
                if prev is None:
                    accum[item.chunk.chunk_id] = (contribution, item)
                else:
                    accum[item.chunk.chunk_id] = (prev[0] + contribution, prev[1])
        # Sort by fused score desc, then chunk_id asc for stability.
        ordered = sorted(
            accum.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )[:top_k]
        fused.append(
            [RetrievedChunk(chunk=item.chunk, score=score) for _, (score, item) in ordered],
        )
    return fused


# Light helpers exposed for unit tests.
__all__ = [
    "Bm25Retriever",
    "reciprocal_rank_fusion",
]


def _ensure_numpy_array(arr) -> np.ndarray:  # pragma: no cover — defensive helper
    return np.asarray(arr)
