"""Dense retrieval: encode queries, query the Chroma index, return top-k chunks."""

from __future__ import annotations

from collections.abc import Sequence

from option8_rag.embeddings.encoder import TextEncoder
from option8_rag.index.chroma_index import ChromaIndex
from option8_rag.types import Query, RetrievedChunk


class DenseRetriever:
    """Encode queries and look them up in a Chroma collection.

    Args:
        encoder: A configured :class:`TextEncoder`.
        index: A :class:`ChromaIndex` populated for the same encoder.
        top_k: Default number of results per query.
    """

    def __init__(
        self,
        *,
        encoder: TextEncoder,
        index: ChromaIndex,
        top_k: int = 10,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.top_k = top_k

    def retrieve(
        self,
        queries: Sequence[Query],
        *,
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Return top-k retrieved chunks per query."""

        k = top_k if top_k is not None else self.top_k
        if not queries:
            return []

        embeddings = self.encoder.encode_queries([q.text for q in queries])
        return self.index.query(query_embeddings=embeddings, top_k=k)
