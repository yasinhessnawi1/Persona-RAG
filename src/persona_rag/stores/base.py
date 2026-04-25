"""Base class for typed ChromaDB-backed persona memory stores."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from loguru import logger

from persona_rag.schema.chunker import PersonaChunk

if TYPE_CHECKING:  # pragma: no cover
    pass


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingFunctionLike(Protocol):
    """Duck-typed embedding function. Matches the subset of ChromaDB's contract we rely on."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...


class RuntimeWriteForbiddenError(RuntimeError):
    """Raised when a caller tries to runtime-write to a store whose flag is False."""


def _default_embedding_function(model_name: str) -> EmbeddingFunctionLike:
    """Lazy import so modules are importable without sentence-transformers installed."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)  # type: ignore[return-value]


class TypedMemoryStore:
    """Shared ChromaDB plumbing for the four typed persona stores.

    Concrete subclasses set `COLLECTION_NAME` and `ALLOW_RUNTIME_WRITE` class
    attributes. Subclasses override `query` to add typed filters (epistemic,
    valid_time, decay) where relevant.
    """

    COLLECTION_NAME: ClassVar[str] = ""
    ALLOW_RUNTIME_WRITE: ClassVar[bool] = False

    def __init__(
        self,
        persist_path: Path | str,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_function: EmbeddingFunctionLike | None = None,
        client: Any | None = None,
    ) -> None:
        """Open (or create) the store's ChromaDB collection.

        `client` lets a caller share one `PersistentClient` across the four
        stores (the registry does this). `embedding_function` overrides
        `embedding_model` — tests use it to inject a deterministic fake
        embedder and avoid the MiniLM download.
        """
        import chromadb

        if not self.COLLECTION_NAME:
            raise TypeError(
                f"{type(self).__name__}.COLLECTION_NAME is unset — subclasses must define it."
            )

        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model

        self._ef: EmbeddingFunctionLike = (
            embedding_function
            if embedding_function is not None
            else _default_embedding_function(embedding_model)
        )

        self._client = client or chromadb.PersistentClient(path=str(self._persist_path))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "{} ready: collection={!r} path={!s} embedding_model={!r} runtime_write={}",
            type(self).__name__,
            self.COLLECTION_NAME,
            self._persist_path,
            embedding_model,
            self.ALLOW_RUNTIME_WRITE,
        )

    # ---------------------------------------------------------- writes

    def index(self, chunks: list[PersonaChunk]) -> int:
        """Upsert a list of chunks. Idempotent by chunk id.

        Chunks whose kind does not belong to this store are filtered out silently —
        so the registry can pass the full flat list and every store picks its own.
        """
        accepted = [c for c in chunks if self._accepts_kind(c.kind)]
        if not accepted:
            return 0
        self._collection.upsert(
            ids=[c.id for c in accepted],
            documents=[c.text for c in accepted],
            metadatas=[dict(c.metadata) for c in accepted],
        )
        logger.debug(
            "{} indexed {} chunks",
            type(self).__name__,
            len(accepted),
        )
        return len(accepted)

    def write(self, chunk: PersonaChunk) -> None:
        """Runtime-write a single chunk. Raises `RuntimeWriteForbiddenError` if not allowed."""
        if not self.ALLOW_RUNTIME_WRITE:
            raise RuntimeWriteForbiddenError(
                f"{type(self).__name__} does not allow runtime writes; chunk {chunk.id!r} rejected."
            )
        if not self._accepts_kind(chunk.kind):
            raise ValueError(
                f"{type(self).__name__} does not accept kind={chunk.kind!r}; chunk {chunk.id!r}"
            )
        self._collection.upsert(
            ids=[chunk.id],
            documents=[chunk.text],
            metadatas=[dict(chunk.metadata)],
        )

    def delete_persona(self, persona_id: str) -> int:
        """Delete every chunk belonging to one persona. Returns count removed."""
        result = self._collection.get(where={"persona_id": persona_id}, include=[])
        ids = list(result.get("ids") or [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Total chunk count in this collection."""
        return int(self._collection.count())

    # ---------------------------------------------------------- reads

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        persona_id: str | None = None,
        extra_where: dict[str, Any] | None = None,
    ) -> list[PersonaChunk]:
        """Semantic top-k query. Subclasses add typed filters in `extra_where`."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not text or not text.strip():
            return []

        where = self._compose_where(persona_id=persona_id, extra_where=extra_where)
        kwargs: dict[str, Any] = {"query_texts": [text], "n_results": top_k}
        if where is not None:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        return self._unpack_query_result(raw)

    def get_all(self, persona_id: str | None = None) -> list[PersonaChunk]:
        """Return every chunk (optionally filtered to one persona). Used by IdentityStore."""
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if persona_id is not None:
            kwargs["where"] = {"persona_id": persona_id}
        raw = self._collection.get(**kwargs)
        return self._unpack_get_result(raw)

    # ---------------------------------------------------------- helpers

    def _accepts_kind(self, kind: str) -> bool:
        """True if this store should hold chunks of that kind. Subclass to override."""
        raise NotImplementedError

    @staticmethod
    def _compose_where(
        *,
        persona_id: str | None,
        extra_where: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if persona_id is not None:
            clauses.append({"persona_id": persona_id})
        if extra_where:
            clauses.append(extra_where)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _unpack_query_result(raw: dict[str, Any]) -> list[PersonaChunk]:
        ids_batches = raw.get("ids") or [[]]
        if not ids_batches or not ids_batches[0]:
            return []

        ids = ids_batches[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        out: list[PersonaChunk] = []
        for i, _id in enumerate(ids):
            meta = dict(metas[i]) if metas and metas[i] else {}
            kind = meta.get("kind", "identity")  # fallback — should always be set
            chunk = PersonaChunk(
                id=_id,
                text=docs[i] if docs else "",
                kind=kind,  # type: ignore[arg-type]
                metadata={k: str(v) for k, v in meta.items()},
                distance=dists[i] if dists else None,
            )
            out.append(replace(chunk, distance=chunk.distance))
        return out

    @staticmethod
    def _unpack_get_result(raw: dict[str, Any]) -> list[PersonaChunk]:
        ids = list(raw.get("ids") or [])
        if not ids:
            return []
        docs = list(raw.get("documents") or [])
        metas = list(raw.get("metadatas") or [])

        out: list[PersonaChunk] = []
        for i, _id in enumerate(ids):
            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            kind = meta.get("kind", "identity")
            out.append(
                PersonaChunk(
                    id=_id,
                    text=docs[i] if i < len(docs) else "",
                    kind=kind,  # type: ignore[arg-type]
                    metadata={k: str(v) for k, v in meta.items()},
                )
            )
        return out

    # ---------------------------------------------------------- properties

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name for this store."""
        return self.COLLECTION_NAME

    @property
    def persist_path(self) -> Path:
        """Filesystem path of the underlying ChromaDB database."""
        return self._persist_path

    @property
    def allow_runtime_write(self) -> bool:
        """True if `write()` may be called at runtime."""
        return self.ALLOW_RUNTIME_WRITE
