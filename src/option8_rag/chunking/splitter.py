"""Token-aware recursive chunking.

Wraps LlamaIndex's ``SentenceSplitter`` to keep boundary-aware chunking with
configurable size and overlap. The wrapper adapts the splitter's text-only
output back to our :class:`Chunk` type and assigns canonical chunk ids.
"""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from option8_rag.types import Chunk, Document, chunk_id_for


class SentenceChunker:
    """Token-aware chunker with configurable size and overlap.

    Args:
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.

    The chunker is stateless beyond its parameters; it is safe to reuse.
    """

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})",
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        from llama_index.core.node_parser import SentenceSplitter

        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks.

        Args:
            document: Source document.

        Returns:
            Ordered list of chunks. Empty inputs produce an empty list.
        """

        if not document.text.strip():
            return []

        pieces = self._splitter.split_text(document.text)
        chunks: list[Chunk] = []
        for index, text in enumerate(pieces):
            cleaned = text.strip()
            if not cleaned:
                continue
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document.doc_id, index),
                    doc_id=document.doc_id,
                    text=cleaned,
                    index=index,
                    metadata={
                        "title": document.title,
                        "source": document.source,
                        **document.metadata,
                    },
                ),
            )
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """Chunk many documents and concatenate the results.

        The order of chunks within a document is preserved; documents are
        processed in iteration order.
        """

        result: list[Chunk] = []
        n_docs = 0
        for doc in documents:
            n_docs += 1
            result.extend(self.chunk_document(doc))
        logger.info(
            "chunked {n_docs} documents into {n_chunks} chunks (size={size}, overlap={overlap})",
            n_docs=n_docs,
            n_chunks=len(result),
            size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        return result
