"""Knowledge (domain) vector store: ChromaDB dense + bm25s sparse, hybrid via RRF.

Mirrors :class:`TypedMemoryStore`'s shape but stands alone because the chunk
shape (:class:`KnowledgeChunk`) and metadata semantics
(`doc_id` / `source` / `chunk_ix`) differ from persona chunks.

Architecture
------------

- One ChromaDB collection (``knowledge_chunks``) under a `PersistentClient` at
  a configurable path (default ``.chroma/knowledge/``). Cosine space.
- Default embedder: ``BAAI/bge-small-en-v1.5``.
- Chunker: LlamaIndex ``SentenceSplitter`` (chunk_size=512, chunk_overlap=50),
  imported only as a one-function call — no llama-index types leak.
- BM25: ``bm25s`` index built parallel to dense at ``index_corpus`` time, held
  in-process. For corpora up to ~10 k chunks this is sub-second to rebuild;
  BM25 is not persisted to disk.
- Hybrid retrieval: dense top-`N` and BM25 top-`N` fused via RRF (default
  k=60) or weighted-sum (ablation toggle when `alpha` is set).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from persona_rag.stores.knowledge_chunk import KnowledgeChunk

# `persona_rag.retrieval.{bm25_utils,fusion}` are imported lazily inside the
# methods that use them — `persona_rag.retrieval.__init__` imports the baseline
# classes which in turn import this module, so a top-level import would close
# a cycle. The lazy import is purely a cycle-break; the modules are pure-logic
# (no heavy initialisation).

# Re-exported from the lazy import for the default RRF k constant — pin here
# so callers can pass an explicit value without importing the fusion module.
DEFAULT_RRF_K = 60

DEFAULT_KNOWLEDGE_COLLECTION_NAME = "knowledge_chunks"
DEFAULT_KNOWLEDGE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


class _EmbeddingFunctionLike(Protocol):
    """Duck-typed embedding function. Subset of ChromaDB's contract we use."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...


def _default_embedding_function(model_name: str) -> _EmbeddingFunctionLike:
    """Lazy import of sentence-transformers; tests inject deterministic fakes instead."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)  # type: ignore[return-value]


@dataclass(frozen=True)
class KnowledgeDocument:
    """One source document to index. `source` is a human-readable label (e.g. filename)."""

    doc_id: str
    text: str
    source: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def _chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text via LlamaIndex's `SentenceSplitter`.

    Wrapped here so callers depend on a `list[str]` not on llama-index types.
    Imports lazily so the module is importable without llama-index installed
    (helpful for type checking + tooling).
    """
    if not text or not text.strip():
        return []
    from llama_index.core import Document as LIDocument
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([LIDocument(text=text)])
    return [n.get_content() for n in nodes]


class KnowledgeStore:
    """ChromaDB-backed dense store + parallel bm25s sparse index.

    One instance per project run. `index_corpus` rebuilds both indexes; for
    incremental updates use `add_documents` (rebuilds BM25, upserts dense).
    """

    def __init__(
        self,
        persist_path: Path | str,
        *,
        collection_name: str = DEFAULT_KNOWLEDGE_COLLECTION_NAME,
        embedding_model: str = DEFAULT_KNOWLEDGE_EMBEDDING_MODEL,
        embedding_function: _EmbeddingFunctionLike | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        client: Any | None = None,
    ) -> None:
        """Open (or create) the knowledge store.

        `client` lets callers share a `PersistentClient` with the persona
        stores if they're running in the same process. `embedding_function`
        overrides `embedding_model` — tests inject a deterministic fake to
        avoid the ~133 MB bge weight download.
        """
        import chromadb

        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._ef: _EmbeddingFunctionLike = (
            embedding_function
            if embedding_function is not None
            else _default_embedding_function(embedding_model)
        )

        self._client = client or chromadb.PersistentClient(path=str(self._persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

        # bm25s index — built lazily on first `index_corpus` / `add_documents`,
        # held in-process. `_bm25_corpus` keeps the parallel doc-id list so we
        # can map BM25's integer indices back to chunk ids.
        self._bm25 = None  # type: ignore[assignment]
        self._bm25_corpus: list[str] = []  # parallel chunk-id list

        logger.info(
            "KnowledgeStore ready: collection={!r} path={!s} embedding_model={!r}",
            collection_name,
            self._persist_path,
            embedding_model,
        )

    # ----------------------------------------------------------------- writes

    def index_corpus(self, documents: Iterable[KnowledgeDocument]) -> int:
        """Chunk + embed + index every document. Replaces any existing chunks for those doc_ids."""
        chunks = self._chunk_documents(list(documents))
        if not chunks:
            logger.warning("KnowledgeStore.index_corpus: no chunks produced")
            return 0
        # Idempotent dense upsert by chunk id.
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[dict(c.metadata) for c in chunks],
        )
        self._rebuild_bm25_from_collection()
        logger.info(
            "KnowledgeStore indexed {} chunks across {} document(s)",
            len(chunks),
            len({c.metadata.get("doc_id", c.id) for c in chunks}),
        )
        return len(chunks)

    def add_documents(self, documents: Iterable[KnowledgeDocument]) -> int:
        """Add documents incrementally. Same semantics as `index_corpus` for a single batch."""
        return self.index_corpus(documents)

    def remove_documents(self, doc_ids: Iterable[str]) -> int:
        """Remove every chunk belonging to the given source ``doc_id``s.

        Used by the counterfactual-probe runner to eject planted counter-
        evidence after the probe turn so the next turn sees the
        pre-injection store. Operates on the source ``doc_id`` (set at
        ``add_documents`` time), not on chunk ids — one input doc may have
        produced several chunks.

        Returns the number of chunk rows removed. Idempotent: calling on
        a doc that's already absent is a no-op.
        """
        wanted = list(doc_ids)
        if not wanted:
            return 0
        # ChromaDB's `where` clause matches per chunk; `doc_id` is set in metadata
        # by `_chunk_documents`. Pull matching chunk ids first so we can return
        # the count and rebuild bm25 only when something was removed.
        try:
            raw = self._collection.get(
                where={"doc_id": {"$in": wanted}},
                include=["metadatas"],
            )
        except Exception:  # pragma: no cover — fallback for older chroma `where` shapes
            # Older chromadb releases don't accept `$in`; fall back to a scan.
            raw = self._collection.get(include=["metadatas"])
            keep_meta = raw.get("metadatas") or []
            keep_ids = raw.get("ids") or []
            matching_ids = [
                cid
                for cid, meta in zip(keep_ids, keep_meta, strict=False)
                if (meta or {}).get("doc_id") in set(wanted)
            ]
        else:
            matching_ids = list(raw.get("ids") or [])
        if not matching_ids:
            return 0
        self._collection.delete(ids=matching_ids)
        self._rebuild_bm25_from_collection()
        logger.info(
            "KnowledgeStore removed {} chunks across doc_ids {}",
            len(matching_ids),
            wanted,
        )
        return len(matching_ids)

    def _chunk_documents(self, documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        out: list[KnowledgeChunk] = []
        for doc in documents:
            pieces = _chunk_text(
                doc.text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
            )
            for ix, piece in enumerate(pieces):
                meta = {"doc_id": doc.doc_id, "source": doc.source, "chunk_ix": str(ix)}
                meta.update(doc.metadata)
                out.append(
                    KnowledgeChunk(
                        id=f"{doc.doc_id}:{ix}",
                        text=piece,
                        metadata=meta,
                    )
                )
        return out

    def _rebuild_bm25_from_collection(self) -> None:
        """Rebuild the in-memory bm25s index from every chunk in the collection."""
        import bm25s

        from persona_rag.retrieval.bm25_utils import tokenize as bm25_tokenize

        raw = self._collection.get(include=["documents", "metadatas"])
        ids = list(raw.get("ids") or [])
        docs = list(raw.get("documents") or [])
        if not ids:
            self._bm25 = None
            self._bm25_corpus = []
            return
        tokens = bm25_tokenize(docs)
        retriever = bm25s.BM25()
        retriever.index(tokens)
        self._bm25 = retriever
        self._bm25_corpus = ids
        logger.debug("BM25 index rebuilt: {} chunks", len(ids))

    # ----------------------------------------------------------------- reads

    def query_dense(self, text: str, *, top_k: int = 5) -> list[KnowledgeChunk]:
        """Dense (embedding) top-k via ChromaDB."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not text or not text.strip():
            return []
        raw = self._collection.query(query_texts=[text], n_results=top_k)
        return self._unpack_query_result(raw)

    def query_bm25(self, text: str, *, top_k: int = 5) -> list[KnowledgeChunk]:
        """Sparse (BM25) top-k via the in-memory bm25s index."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not text or not text.strip() or self._bm25 is None:
            return []
        from persona_rag.retrieval.bm25_utils import tokenize as bm25_tokenize

        query_tokens = bm25_tokenize(text)
        results, scores = self._bm25.retrieve(query_tokens, k=min(top_k, len(self._bm25_corpus)))
        # bm25s returns (n_queries, k) arrays. We only ever pass one query, so [0].
        out: list[KnowledgeChunk] = []
        for row_ix, score in zip(results[0], scores[0], strict=False):
            chunk_id = self._bm25_corpus[int(row_ix)]
            chunk = self._fetch_chunk(chunk_id, distance=float(-score))
            if chunk is not None:
                out.append(chunk)
        return out

    def query_hybrid(
        self,
        text: str,
        *,
        top_k: int = 5,
        candidate_pool: int = 20,
        alpha: float | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[KnowledgeChunk]:
        """Hybrid retrieval: dense + BM25 fused via RRF (default) or weighted sum.

        `candidate_pool` is the per-leg top-N before fusion. `alpha=None` → RRF
        (the default). `alpha` ∈ [0, 1] → weighted-sum ablation.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if candidate_pool < top_k:
            candidate_pool = top_k
        dense_results = self.query_dense(text, top_k=candidate_pool)
        bm25_results = self.query_bm25(text, top_k=candidate_pool)

        from persona_rag.retrieval.fusion import (
            reciprocal_rank_fusion,
            weighted_sum_fusion,
        )

        if alpha is None:
            fused = reciprocal_rank_fusion(
                [[c.id for c in dense_results], [c.id for c in bm25_results]],
                k=rrf_k,
                top_k=top_k,
            )
        else:
            # weighted_sum_fusion needs (id, score) pairs. For dense use
            # 1 - distance (cosine similarity); for BM25 use raw score.
            dense_pairs = [(c.id, 1.0 - (c.distance or 0.0)) for c in dense_results]
            bm25_pairs = [(c.id, -(c.distance or 0.0)) for c in bm25_results]  # we stored -score
            fused = weighted_sum_fusion(dense_pairs, bm25_pairs, alpha=alpha, top_k=top_k)

        # Re-fetch in fused order so callers get full KnowledgeChunk objects.
        by_id = {c.id: c for c in dense_results}
        for c in bm25_results:
            by_id.setdefault(c.id, c)
        out: list[KnowledgeChunk] = []
        for chunk_id, score in fused:
            chunk = by_id.get(chunk_id) or self._fetch_chunk(chunk_id, distance=None)
            if chunk is None:
                continue
            # Replace distance with the fused score (negated so smaller = better,
            # matching the rest of the project's "distance" convention).
            out.append(
                KnowledgeChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    distance=-score,
                )
            )
        return out

    def _fetch_chunk(self, chunk_id: str, *, distance: float | None) -> KnowledgeChunk | None:
        raw = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = list(raw.get("ids") or [])
        if not ids:
            return None
        docs = list(raw.get("documents") or [])
        metas = list(raw.get("metadatas") or [])
        meta = dict(metas[0]) if metas else {}
        return KnowledgeChunk(
            id=ids[0],
            text=docs[0] if docs else "",
            metadata={k: str(v) for k, v in meta.items()},
            distance=distance,
        )

    @staticmethod
    def _unpack_query_result(raw: dict[str, Any]) -> list[KnowledgeChunk]:
        ids_batches = raw.get("ids") or [[]]
        if not ids_batches or not ids_batches[0]:
            return []
        ids = ids_batches[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        out: list[KnowledgeChunk] = []
        for i, _id in enumerate(ids):
            meta = dict(metas[i]) if metas and metas[i] else {}
            out.append(
                KnowledgeChunk(
                    id=_id,
                    text=docs[i] if docs else "",
                    metadata={k: str(v) for k, v in meta.items()},
                    distance=dists[i] if dists else None,
                )
            )
        return out

    # ------------------------------------------------------------ properties

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name."""
        return self._collection_name

    @property
    def persist_path(self) -> Path:
        """Filesystem path of the underlying ChromaDB database."""
        return self._persist_path

    def count(self) -> int:
        """Total chunk count across all documents in the collection."""
        return int(self._collection.count())
