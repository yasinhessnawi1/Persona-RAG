"""Tests for the retrieval-metric helpers."""

from __future__ import annotations

import pytest

from option8_rag.evaluate.retrieval_metrics import evaluate_retrieval


def _try_metrics():
    try:
        return evaluate_retrieval(
            qrels={"q1": {"d1": 1, "d2": 0, "d3": 1}},
            run={"q1": {"d1": 0.9, "d3": 0.8, "d2": 0.1}},
            k_values=(1, 5, 10),
        )
    except ImportError:  # pytrec_eval missing
        pytest.skip("pytrec_eval not installed")


def test_perfect_run_yields_recall_1_at_2() -> None:
    metrics = _try_metrics()
    # Both relevant docs (d1, d3) appear in the top-2 of the run.
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    # nDCG@10 should be high (top-2 are both relevant; max possible is 1.0).
    assert metrics["ndcg@10"] == pytest.approx(1.0, abs=1e-6)
    # MRR@10: d1 at rank 1 -> 1.0
    assert metrics["mrr@10"] == pytest.approx(1.0)


def test_mrr_when_relevant_at_rank_2() -> None:
    metrics = (
        evaluate_retrieval(
            qrels={"q1": {"d1": 1, "d2": 0}},
            run={"q1": {"d2": 0.9, "d1": 0.5}},
            k_values=(1, 5, 10),
        )
        if _has_pytrec()
        else None
    )
    if metrics is None:
        pytest.skip("pytrec_eval not installed")
    assert metrics["mrr@10"] == pytest.approx(0.5)
    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["recall@5"] == pytest.approx(1.0)


def test_empty_qrels_raises() -> None:
    if not _has_pytrec():
        pytest.skip("pytrec_eval not installed")
    with pytest.raises(ValueError):
        evaluate_retrieval(qrels={}, run={"q1": {"d1": 0.1}}, k_values=(1,))
    with pytest.raises(ValueError):
        evaluate_retrieval(qrels={"q1": {"d1": 1}}, run={}, k_values=(1,))


def _has_pytrec() -> bool:
    try:
        import pytrec_eval  # noqa: F401

        return True
    except ImportError:
        return False
