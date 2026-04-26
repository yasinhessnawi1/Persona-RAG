"""Hybrid-retrieval fusion: Reciprocal Rank Fusion (RRF) + optional weighted-sum.

RRF with ``k = 60`` (Cormack et al., SIGIR 2009) is the default. The
weighted-sum fallback (``alpha`` in [0, 1]) is kept as an ablation toggle.
Implementation is intentionally short (~30 lines) so it is auditable and
unit-testable with a synthetic "exact-term query beats semantic-only" probe.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple rankings via Reciprocal Rank Fusion.

    Cormack, Clarke & Büttcher 2009: ``score(d) = Σ_r 1 / (k + rank_r(d))``,
    where ``rank_r(d)`` is the 1-indexed rank of doc id ``d`` in ranking
    ``r``. Documents missing from a ranking contribute 0 from that ranking.

    Parameters
    ----------
    rankings:
        List of rankings; each is a list of doc ids in descending-relevance
        order. The two rankings here are typically dense top-`N` and BM25
        top-`N`.
    k:
        RRF constant. Default 60 per Cormack 2009 / Elastic / LlamaIndex.
    top_k:
        Cap on the fused output length. If ``None``, return all unique ids
        seen in any input ranking.

    Returns
    -------
    list of ``(doc_id, fused_score)`` tuples, descending by score. Stable
    secondary order: doc id ascending, so tied scores produce reproducible
    output.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        fused = fused[:top_k]
    return fused


def weighted_sum_fusion(
    dense: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    *,
    alpha: float,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists via min-max-normalised weighted sum.

    ``score(d) = alpha * dense_score_norm(d) + (1 - alpha) * bm25_score_norm(d)``.
    Min-max normalisation collapses absolute-score differences between BM25
    (raw similarity sum, ~unbounded) and dense cosine (in [0, 1] for
    normalised embeddings) so the weighting is meaningful. Documents missing
    from one list contribute 0 from that side.

    Used only as an ablation toggle to compare against RRF; reproduces the
    literature's standard hybrid-fusion comparator.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def _normalise(pairs: list[tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        lo, hi = min(scores), max(scores)
        spread = hi - lo
        if spread == 0:
            return {doc_id: 1.0 for doc_id, _ in pairs}
        return {doc_id: (s - lo) / spread for doc_id, s in pairs}

    dn = _normalise(dense)
    bn = _normalise(bm25)
    all_ids = set(dn) | set(bn)
    fused = sorted(
        (
            (doc_id, alpha * dn.get(doc_id, 0.0) + (1 - alpha) * bn.get(doc_id, 0.0))
            for doc_id in all_ids
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k is not None:
        fused = fused[:top_k]
    return fused


__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion", "weighted_sum_fusion"]
