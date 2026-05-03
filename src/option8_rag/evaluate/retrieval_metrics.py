"""Retrieval metric helpers.

Wraps ``pytrec_eval`` for Recall@k, nDCG@k, and MAP@k. MRR@k is computed by
a small custom helper because ``pytrec_eval`` does not expose it across
versions consistently. The signature mirrors the BEIR reference evaluator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from option8_rag.types import Qrels, Run


def evaluate_retrieval(
    *,
    qrels: Qrels,
    run: Run,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@k, nDCG@k, MAP@k, and MRR@k macro-averaged over queries.

    Args:
        qrels: ``{qid: {docid: int_relevance}}``.
        run: ``{qid: {docid: float_score}}``. Higher is better.
        k_values: Cutoffs to report.

    Returns:
        Flat dict, e.g. ``{"recall@10": 0.62, "ndcg@10": 0.51, ...}``.
    """

    if not qrels:
        raise ValueError("qrels is empty")
    if not run:
        raise ValueError("run is empty")

    metrics: dict[str, float] = {}
    metrics.update(_pytrec_metrics(qrels=qrels, run=run, k_values=k_values))
    metrics.update(_mrr_at_k(qrels=qrels, run=run, k_values=k_values))
    return metrics


def _pytrec_metrics(
    *,
    qrels: Qrels,
    run: Run,
    k_values: Sequence[int],
) -> dict[str, float]:
    import pytrec_eval

    k_str = ",".join(str(k) for k in k_values)
    measures = {
        f"ndcg_cut.{k_str}",
        f"map_cut.{k_str}",
        f"recall.{k_str}",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)

    if not per_query:
        return {}

    out: dict[str, float] = {}
    for k in k_values:
        out[f"ndcg@{k}"] = _macro_avg(per_query, f"ndcg_cut_{k}")
        out[f"map@{k}"] = _macro_avg(per_query, f"map_cut_{k}")
        out[f"recall@{k}"] = _macro_avg(per_query, f"recall_{k}")
    return out


def _macro_avg(per_query: dict[str, dict[str, float]], key: str) -> float:
    if not per_query:
        return 0.0
    total = 0.0
    n = 0
    for m in per_query.values():
        if key in m:
            total += float(m[key])
            n += 1
    return total / n if n else 0.0


def _mrr_at_k(
    *,
    qrels: Qrels,
    run: Run,
    k_values: Iterable[int],
) -> dict[str, float]:
    """Mean Reciprocal Rank @ k.

    A document is considered relevant if its qrels score is > 0.
    """

    out: dict[str, float] = {}
    for k in k_values:
        rrs: list[float] = []
        for qid, judged in qrels.items():
            relevant = {d for d, s in judged.items() if s > 0}
            if not relevant:
                continue
            ranked = sorted(
                run.get(qid, {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:k]
            rr = 0.0
            for rank, (did, _) in enumerate(ranked, start=1):
                if did in relevant:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
        out[f"mrr@{k}"] = sum(rrs) / len(rrs) if rrs else 0.0
    return out


__all__ = ["evaluate_retrieval"]
