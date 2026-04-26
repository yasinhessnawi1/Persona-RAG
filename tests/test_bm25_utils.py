"""Tests for the BM25 tokenization helper."""

from __future__ import annotations

from persona_rag.retrieval.bm25_utils import tokenize


def test_single_string_returns_list_of_one() -> None:
    """Calling with one string still returns a list-of-one — callers don't branch on type."""
    out = tokenize("hello world")
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], list)


def test_empty_input_returns_empty_list() -> None:
    assert tokenize([]) == []


def test_lowercase_and_stopwords_applied() -> None:
    """Default config lowercases + drops English stopwords."""
    out = tokenize(["The Quick Brown Fox"])
    tokens = out[0]
    # All lowercase.
    assert all(t == t.lower() for t in tokens)
    # 'the' is an English stopword.
    assert "the" not in tokens
    assert "quick" in tokens


def test_index_query_consistency() -> None:
    """The pitfall #1 mitigation: same tokenizer for index and query produces identical tokens."""
    corpus_tokens = tokenize(["The quick brown fox"])
    query_tokens = tokenize("the quick brown fox")
    assert corpus_tokens[0] == query_tokens[0]
