"""Tests for shared types and helpers."""

from __future__ import annotations

from option8_rag.types import Chunk, Document, chunk_id_for


def test_chunk_id_zero_padding() -> None:
    assert chunk_id_for("doc1", 0) == "doc1::0000"
    assert chunk_id_for("doc1", 42) == "doc1::0042"
    assert chunk_id_for("abc", 9999) == "abc::9999"


def test_document_immutable() -> None:
    doc = Document(doc_id="d", text="t")
    try:
        doc.text = "x"  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("Document should be frozen")


def test_chunk_metadata_default_factory_independent() -> None:
    a = Chunk(chunk_id="c1", doc_id="d", text="t", index=0)
    b = Chunk(chunk_id="c2", doc_id="d", text="t", index=1)
    a.metadata["x"] = 1
    assert "x" not in b.metadata
