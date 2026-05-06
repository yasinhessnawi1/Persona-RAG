"""Tests for the chunk-store JSONL roundtrip used by BM25 / hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

from option8_rag.index.chunk_store import (
    chunks_dump_path,
    read_chunks_jsonl,
    write_chunks_jsonl,
)
from option8_rag.types import Chunk


def test_chunks_dump_path_is_deterministic(tmp_path: Path) -> None:
    p = chunks_dump_path(
        chroma_root=tmp_path,
        corpus_name="uia_ikt",
        chunk_size=512,
        chunk_overlap=64,
    )
    assert p.parent == tmp_path
    assert p.name == "chunks_uia_ikt_chunk512_ov64.jsonl"


def test_roundtrip_preserves_chunk_fields(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="d1::0000",
            doc_id="d1",
            text="hello world",
            index=0,
            metadata={"source": "u", "leader": "X"},
        ),
        Chunk(
            chunk_id="d1::0001",
            doc_id="d1",
            text="second piece",
            index=1,
            metadata={},
        ),
    ]
    path = tmp_path / "dump.jsonl"
    write_chunks_jsonl(path, chunks)

    loaded = read_chunks_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].chunk_id == "d1::0000"
    assert loaded[0].text == "hello world"
    assert loaded[0].index == 0
    assert loaded[0].metadata == {"source": "u", "leader": "X"}
    assert loaded[1].chunk_id == "d1::0001"
    assert loaded[1].metadata == {}


def test_read_missing_file_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        read_chunks_jsonl(tmp_path / "does-not-exist.jsonl")
