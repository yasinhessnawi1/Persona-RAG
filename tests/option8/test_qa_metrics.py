"""Tests for short-answer QA metrics."""

from __future__ import annotations

import pytest

from option8_rag.evaluate.qa_metrics import (
    aggregate,
    contains_match,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_strips_articles_and_punctuation() -> None:
    assert normalize_answer("The Answer.") == "answer"
    assert normalize_answer("a Quick Brown FOX!") == "quick brown fox"
    assert normalize_answer("  multiple   spaces  ") == "multiple spaces"


def test_exact_match_basic() -> None:
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("The Paris", ["Paris."]) == 1.0
    assert exact_match("London", ["Paris"]) == 0.0


def test_exact_match_with_multiple_refs() -> None:
    assert exact_match("Eiffel Tower", ["The Eiffel Tower", "Tour Eiffel"]) == 1.0
    assert exact_match("Big Ben", ["Eiffel Tower", "Tour Eiffel"]) == 0.0


def test_token_f1_partial_overlap() -> None:
    f1 = token_f1("the quick brown fox", ["a quick fox"])
    assert 0.0 < f1 < 1.0
    assert f1 == pytest.approx((2 * (2 / 3) * (2 / 2)) / ((2 / 3) + 1.0))


def test_aggregate_macro_average() -> None:
    scores = aggregate(
        predictions=["paris", "london"],
        references=[["Paris"], ["Berlin"]],
    )
    assert scores["em"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.5)
    assert scores["contains"] == pytest.approx(0.5)


def test_empty_prediction_yields_zero_f1() -> None:
    assert token_f1("", ["something"]) == 0.0


def test_contains_match_credits_wrapped_answers() -> None:
    # Verbose RAG-style answer that wraps the bare fact in a sentence.
    assert (
        contains_match(
            "According to passage [c1], the course leader is Morten Goodwin.",
            ["Morten Goodwin"],
        )
        == 1.0
    )


def test_contains_match_negatives() -> None:
    assert contains_match("London is the capital of the UK.", ["Paris"]) == 0.0
    assert contains_match("", ["Morten Goodwin"]) == 0.0
    assert contains_match("Morten Goodwin teaches it.", []) == 0.0


def test_contains_match_normalises_articles_and_case() -> None:
    # "the" and "An" are stripped on both sides; case is folded.
    assert (
        contains_match(
            "Answer: An Eiffel TOWER stands in Paris.",
            ["the Eiffel Tower"],
        )
        == 1.0
    )


def test_aggregate_includes_contains_field_when_empty() -> None:
    scores = aggregate(predictions=[], references=[])
    assert scores == {"em": 0.0, "f1": 0.0, "contains": 0.0}
