"""Cross-encoder reranker that wraps any base retriever.

The base retriever (dense / BM25 / hybrid) returns the top-N candidates;
the cross-encoder rescues semantic precision by jointly encoding each
``(query, candidate)`` pair and re-ranking by the predicted relevance
score. Cross-encoders are slower per item but only run on the small
candidate pool, not the whole corpus, so the wall-clock cost is modest
and the precision gain is well documented in the IR literature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from loguru import logger

from option8_rag.types import Query, RetrievedChunk


class CrossEncoderReranker:
    """Wraps a sentence-transformers ``CrossEncoder``.

    Args:
        model_id: HuggingFace model id, e.g.
            ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        batch_size: Per-batch ``(query, candidate)`` pair count.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        device = self._resolve_device()
        logger.info(
            "loading cross-encoder model_id={model_id} device={device}",
            model_id=self.model_id,
            device=device,
        )
        self._model = CrossEncoder(self.model_id, device=device)
        return self._model

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def rerank(
        self,
        *,
        query_text: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank a single query's candidate list and return the top-k."""

        if not candidates:
            return []
        model = self._load()
        pairs = [(query_text, c.chunk.text or "") for c in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scored = [replace(c, score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, top_k)]


class RerankRetriever:
    """Run a base retriever, then rerank its top-N with a cross-encoder.

    Args:
        base: Any retriever exposing ``retrieve(queries, top_k=...)``.
        reranker: Configured :class:`CrossEncoderReranker`.
        base_top_n: How many candidates to ask the base retriever for.
            The cross-encoder reranks across this pool.
        top_k: Final result count after reranking.
    """

    def __init__(
        self,
        *,
        base,
        reranker: CrossEncoderReranker,
        base_top_n: int = 50,
        top_k: int = 10,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.base_top_n = base_top_n
        self.top_k = top_k

    def retrieve(
        self,
        queries: Sequence[Query],
        *,
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Retrieve and rerank, returning the final top-k per query."""

        k = top_k if top_k is not None else self.top_k
        if not queries:
            return []
        base_results = self.base.retrieve(queries, top_k=self.base_top_n)
        out: list[list[RetrievedChunk]] = []
        for q, candidates in zip(queries, base_results, strict=True):
            out.append(
                self.reranker.rerank(query_text=q.text, candidates=candidates, top_k=k),
            )
        return out
