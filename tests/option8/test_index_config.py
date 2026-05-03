"""Tests for the IndexConfig collection-name derivation."""

from __future__ import annotations

from option8_rag.index.chroma_index import IndexConfig


def test_collection_name_is_deterministic_and_keyed_on_all_fields() -> None:
    base = IndexConfig(
        corpus_name="beir_nq",
        embedder_name="bge_large",
        chunk_size=512,
        chunk_overlap=64,
    )
    assert base.collection_name == "beir_nq__bge_large__chunk512_ov64"

    other = IndexConfig(
        corpus_name="beir_nq",
        embedder_name="bge_base",
        chunk_size=512,
        chunk_overlap=64,
    )
    assert other.collection_name != base.collection_name
