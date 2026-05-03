"""Tests for the SentenceChunker wrapper."""

from __future__ import annotations

import pytest

from option8_rag.chunking.splitter import SentenceChunker
from option8_rag.types import Document


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=10, chunk_overlap=10)


def test_empty_document_returns_no_chunks() -> None:
    chunker = SentenceChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker.chunk_document(Document(doc_id="d0", text=""))
    assert chunks == []


def test_short_document_produces_one_chunk() -> None:
    chunker = SentenceChunker(chunk_size=64, chunk_overlap=8)
    text = "This is a short sentence. It should fit in a single chunk."
    chunks = chunker.chunk_document(Document(doc_id="d1", text=text, title="t", source="s"))
    assert len(chunks) >= 1
    first = chunks[0]
    assert first.chunk_id == "d1::0000"
    assert first.doc_id == "d1"
    assert first.metadata["title"] == "t"
    assert first.metadata["source"] == "s"


def test_long_document_produces_multiple_chunks() -> None:
    chunker = SentenceChunker(chunk_size=32, chunk_overlap=4)
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 30
    chunks = chunker.chunk_document(Document(doc_id="d2", text=text))
    assert len(chunks) >= 2
    indices = [c.index for c in chunks]
    assert indices == sorted(indices)
    assert indices[0] == 0
    # IDs must be zero-padded and ordered.
    assert chunks[0].chunk_id == "d2::0000"
    assert chunks[1].chunk_id == "d2::0001"


def test_chunk_documents_concatenates_in_order() -> None:
    chunker = SentenceChunker(chunk_size=64, chunk_overlap=8)
    docs = [
        Document(doc_id="a", text="alpha sentence."),
        Document(doc_id="b", text="beta sentence."),
    ]
    chunks = chunker.chunk_documents(docs)
    assert {c.doc_id for c in chunks} == {"a", "b"}
    a_chunks = [c for c in chunks if c.doc_id == "a"]
    b_chunks = [c for c in chunks if c.doc_id == "b"]
    assert a_chunks[0].chunk_id.startswith("a::")
    assert b_chunks[0].chunk_id.startswith("b::")
