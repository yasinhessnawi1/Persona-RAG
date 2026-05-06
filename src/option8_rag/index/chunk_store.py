"""On-disk JSONL store for chunked corpora.

The dense-only path persists chunks inside Chroma. BM25 and hybrid
retrieval want the same chunked corpus available outside Chroma so they
can rebuild a sparse index without redoing chunking. We persist a single
JSONL per ``(corpus, chunk_size, chunk_overlap)`` triple — embedder-
independent because chunking happens before encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

from option8_rag.types import Chunk


def chunks_dump_path(
    *,
    chroma_root: Path,
    corpus_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    """Deterministic path for the JSONL dump of a chunked corpus."""

    name = f"chunks_{corpus_name}_chunk{chunk_size}_ov{chunk_overlap}.jsonl"
    return chroma_root / name


def write_chunks_jsonl(path: Path, chunks: list[Chunk]) -> None:
    """Write chunks to JSONL (one per line)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "index": c.index,
                        "metadata": c.metadata,
                    },
                )
            )
            f.write("\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    """Read chunks back from a JSONL dump."""

    if not path.exists():
        raise FileNotFoundError(
            f"chunk dump not found at {path}; rerun the index step for this corpus.",
        )
    out: list[Chunk] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                Chunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    text=str(row["text"]),
                    index=int(row.get("index", 0)),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return out
