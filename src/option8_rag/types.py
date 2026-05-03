"""Shared lightweight type definitions for the option8 RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A pre-chunking source document.

    Attributes:
        doc_id: Unique identifier within a corpus.
        text: Cleaned plain text body.
        title: Optional title; empty string when unavailable.
        source: URL, file path, or other origin reference.
        metadata: Arbitrary string-keyed metadata.
    """

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable text chunk derived from a Document.

    Attributes:
        chunk_id: Unique identifier; convention `{doc_id}::{index:04d}`.
        doc_id: Identifier of the source document.
        text: Chunk text.
        index: Position of the chunk within the source document.
        metadata: Arbitrary string-keyed metadata propagated from the source.
    """

    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval query.

    Attributes:
        query_id: Unique identifier within an evaluation set.
        text: Query text.
    """

    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its score.

    Attributes:
        chunk: The chunk itself.
        score: A higher-is-better score (cosine similarity, BM25 score, or
            fused score depending on the retriever).
    """

    chunk: Chunk
    score: float


# Type alias for qrels in the form pytrec_eval consumes:
#   {qid: {docid: relevance_grade_int}}
Qrels = dict[str, dict[str, int]]

# Type alias for run files in the form pytrec_eval consumes:
#   {qid: {docid: score_float}}
Run = dict[str, dict[str, float]]


def chunk_id_for(doc_id: str, index: int) -> str:
    """Canonical chunk-id constructor; kept centralised so the convention is one place."""

    return f"{doc_id}::{index:04d}"


def ensure_dir(path: Path) -> Path:
    """Create the directory if missing; return the path for chaining."""

    path.mkdir(parents=True, exist_ok=True)
    return path
