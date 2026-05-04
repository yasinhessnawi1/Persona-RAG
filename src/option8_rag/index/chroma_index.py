"""Chroma persistent index wrapper.

The wrapper is intentionally thin: it owns a single ``PersistentClient``
and exposes ``upsert`` / ``query`` over an HNSW collection configured for
cosine distance with externally-computed embeddings.

The collection name encodes the embedding model and chunking parameters so
a mismatch between the index and the encoder is impossible by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from option8_rag.types import Chunk, RetrievedChunk


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Identity of a Chroma collection.

    Attributes:
        corpus_name: Logical corpus name (e.g. ``beir_nq``, ``uia_ikt``).
        embedder_name: Identifier for the embedder; baked into the
            collection name so two encoders cannot share an index.
        chunk_size: Chunk size used to populate the collection.
        chunk_overlap: Chunk overlap used to populate the collection.
        hnsw_space: Distance metric. Cosine for normalised embeddings.
        hnsw_ef_search: HNSW search-time candidate list size.
    """

    corpus_name: str
    embedder_name: str
    chunk_size: int
    chunk_overlap: int
    hnsw_space: str = "cosine"
    hnsw_ef_search: int = 100

    @property
    def collection_name(self) -> str:
        """Deterministic collection name from the identity tuple."""

        return (
            f"{self.corpus_name}__{self.embedder_name}__"
            f"chunk{self.chunk_size}_ov{self.chunk_overlap}"
        )


class ChromaIndex:
    """Wrapper over a single persistent Chroma collection."""

    def __init__(self, *, persist_path: Path, config: IndexConfig) -> None:
        import chromadb

        self.persist_path = persist_path
        self.config = config

        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            configuration={"hnsw": {"space": config.hnsw_space}},
            embedding_function=None,
        )
        logger.info(
            "Chroma collection ready: name={name} path={path} space={space}",
            name=config.collection_name,
            path=str(persist_path),
            space=config.hnsw_space,
        )

    @property
    def count(self) -> int:
        """Number of items currently in the collection."""

        return int(self._collection.count())

    # -- write ---------------------------------------------------------

    def upsert(self, *, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Upsert chunks + embeddings into the collection.

        Args:
            chunks: Chunks to write.
            embeddings: ``(len(chunks), dim)`` array of float embeddings.
        """

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) lengths differ",
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            md: dict[str, Any] = {
                "doc_id": c.doc_id,
                "chunk_index": c.index,
            }
            for k, v in c.metadata.items():
                if isinstance(v, str | int | float | bool):
                    md[k] = v
            metadatas.append(md)

        # Chroma >=1.x prefers `upsert` for idempotent writes.
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    # -- read ----------------------------------------------------------

    # Chroma's SQLite-backed query path binds host parameters per query
    # row, so a 3,452-query batch on BEIR/NQ overflows the engine's
    # variable cap ("too many SQL variables"). Slice into windows that
    # stay below SQLite's default 999 / 32,766 limits with margin.
    _QUERY_BATCH_CAP = 64

    def query(
        self,
        *,
        query_embeddings: np.ndarray,
        top_k: int,
        batch_size: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Top-k retrieval for a batch of query embeddings.

        Args:
            query_embeddings: ``(n_queries, dim)`` float array.
            top_k: Number of results per query.
            batch_size: Optional override for the per-call query batch
                size. Defaults to ``_QUERY_BATCH_CAP``; lower if the
                SQLite "too many SQL variables" error reappears on a
                build with a tighter compile-time cap.

        Returns:
            A list of length ``n_queries``; each entry is a list of
            :class:`RetrievedChunk` ordered by descending similarity.
        """

        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D, got shape {query_embeddings.shape}",
            )

        bs = max(1, min(int(batch_size or self._QUERY_BATCH_CAP), self._QUERY_BATCH_CAP))
        n = int(query_embeddings.shape[0])
        out: list[list[RetrievedChunk]] = []

        for start in range(0, n, bs):
            end = min(start + bs, n)
            sub = query_embeddings[start:end]
            result = self._collection.query(
                query_embeddings=sub.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out.extend(self._materialise_query_result(result))
        return out

    def _materialise_query_result(self, result: dict) -> list[list[RetrievedChunk]]:
        """Convert one Chroma query response into our RetrievedChunk rows."""

        ids_batch = result.get("ids") or []
        docs_batch = result.get("documents") or []
        meta_batch = result.get("metadatas") or []
        dist_batch = result.get("distances") or []

        rows: list[list[RetrievedChunk]] = []
        for ids, docs, metas, dists in zip(
            ids_batch,
            docs_batch,
            meta_batch,
            dist_batch,
            strict=False,
        ):
            row: list[RetrievedChunk] = []
            for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
                meta = meta or {}
                row.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            chunk_id=str(chunk_id),
                            doc_id=str(meta.get("doc_id", "")),
                            text=str(text or ""),
                            index=int(meta.get("chunk_index", 0)),
                            metadata=dict(meta),
                        ),
                        # Cosine distance -> similarity. With normalised
                        # embeddings, similarity = 1 - distance.
                        score=float(1.0 - float(dist)),
                    ),
                )
            rows.append(row)
        return rows
