"""Persona vector store: ChromaDB-backed collection for persona chunks.

Separated from the knowledge vector store so retrieval mechanisms can query
each independently. One collection, many personas — the ``persona_id``
metadata disambiguates.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from persona_rag.schema.chunker import PersonaChunk, chunk_persona
from persona_rag.schema.persona import Persona

if TYPE_CHECKING:  # pragma: no cover
    import chromadb


DEFAULT_COLLECTION_NAME = "persona_chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingFunctionLike(Protocol):
    """Duck-typed embedding function. Matches ChromaDB's EmbeddingFunction contract."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...


def _default_embedding_function(model_name: str) -> EmbeddingFunctionLike:
    """Return ChromaDB's sentence-transformers embedding function for `model_name`.

    Imported lazily so the module is importable without sentence-transformers
    installed (tests that use a fake embedder don't need the ~90 MB download).
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)  # type: ignore[return-value]


class PersonaStore:
    """Persistent ChromaDB-backed store for persona chunks.

    - One collection (`DEFAULT_COLLECTION_NAME`) holds chunks for all personas.
    - Idempotent by chunk id (via `upsert`): re-indexing the same persona
      overwrites, does not duplicate.
    - Cosine distance (matches sentence-transformers training objective).
    """

    def __init__(
        self,
        persist_path: Path | str,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_function: EmbeddingFunctionLike | None = None,
    ) -> None:
        """Open (or create) a PersonaStore at `persist_path`.

        `embedding_function` overrides `embedding_model` if supplied — used by
        tests to inject a fake, deterministic embedder without downloading
        MiniLM weights.
        """
        import chromadb

        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        self._ef: EmbeddingFunctionLike = (
            embedding_function
            if embedding_function is not None
            else _default_embedding_function(embedding_model)
        )

        self._client: chromadb.ClientAPI = chromadb.PersistentClient(path=str(self._persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "PersonaStore ready: collection={!r} path={!s} embedding_model={!r}",
            collection_name,
            self._persist_path,
            embedding_model,
        )

    # ------------------------------------------------------------------ writes

    def index(self, persona: Persona) -> list[PersonaChunk]:
        """Chunk a persona and upsert it into the collection. Idempotent by chunk id."""
        chunks = chunk_persona(persona)
        if not chunks:
            logger.warning("persona {!r} produced 0 chunks", persona.persona_id)
            return []

        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[dict(c.metadata) for c in chunks],
        )
        logger.info(
            "indexed persona {!r}: {} chunks ({} self_fact, {} worldview, {} constraint, 1 identity)",
            persona.persona_id,
            len(chunks),
            sum(1 for c in chunks if c.metadata["type"] == "self_fact"),
            sum(1 for c in chunks if c.metadata["type"] == "worldview"),
            sum(1 for c in chunks if c.metadata["type"] == "constraint"),
        )
        return chunks

    def delete_persona(self, persona_id: str) -> int:
        """Delete all chunks for one persona. Returns the number of chunks deleted."""
        before = self._collection.get(where={"persona_id": persona_id}, include=[])
        ids = list(before.get("ids") or [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info("deleted {} chunks for persona {!r}", len(ids), persona_id)
        return len(ids)

    # ------------------------------------------------------------------ reads

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        filter_type: str | None = None,
        persona_id: str | None = None,
    ) -> list[PersonaChunk]:
        """Embed `text`, return the `top_k` nearest chunks.

        `filter_type` restricts to one of `identity | self_fact | worldview | constraint`.
        `persona_id` restricts to a single persona.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not text or not text.strip():
            # Empty query: return no chunks instead of crashing.
            logger.debug("empty query text — returning []")
            return []

        where = self._build_where(filter_type=filter_type, persona_id=persona_id)

        kwargs: dict[str, Any] = {"query_texts": [text], "n_results": top_k}
        if where is not None:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        return self._unpack_query_result(raw)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _build_where(
        *,
        filter_type: str | None,
        persona_id: str | None,
    ) -> dict[str, Any] | None:
        """Assemble a ChromaDB `where` filter, collapsing a single-clause case to no $and."""
        clauses: list[dict[str, Any]] = []
        if filter_type is not None:
            clauses.append({"type": filter_type})
        if persona_id is not None:
            clauses.append({"persona_id": persona_id})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _unpack_query_result(raw: dict[str, Any]) -> list[PersonaChunk]:
        """Flatten ChromaDB's list-of-lists query result into a list[PersonaChunk]."""
        ids_batches = raw.get("ids") or [[]]
        docs_batches = raw.get("documents") or [[]]
        meta_batches = raw.get("metadatas") or [[]]
        dist_batches = raw.get("distances") or [[]]

        if not ids_batches or not ids_batches[0]:
            return []

        ids = ids_batches[0]
        docs = docs_batches[0]
        metas = meta_batches[0]
        dists = dist_batches[0]

        out: list[PersonaChunk] = []
        for i, _id in enumerate(ids):
            meta = dict(metas[i]) if metas and metas[i] else {}
            chunk = PersonaChunk(
                id=_id,
                text=docs[i] if docs else "",
                metadata={k: str(v) for k, v in meta.items()},
            )
            distance = dists[i] if dists else None
            out.append(replace(chunk, distance=distance))
        return out

    # -------------------------------------------------------------- properties

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name."""
        return self._collection_name

    @property
    def persist_path(self) -> Path:
        """Directory on disk where the ChromaDB database lives."""
        return self._persist_path

    def count(self) -> int:
        """Total chunk count across all personas in the collection."""
        return int(self._collection.count())
