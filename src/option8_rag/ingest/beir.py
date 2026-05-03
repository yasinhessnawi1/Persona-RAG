"""BEIR/NQ loader.

Pulls the corpus, the test queries, and the qrels from the official
HuggingFace mirrors and converts them into the project's :class:`Document`
and :class:`Query` types plus a ``Qrels`` dict in the shape that
``pytrec_eval`` consumes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from option8_rag.types import Document, Qrels, Query


@dataclass(frozen=True, slots=True)
class BeirSplit:
    """Bundle of corpus, queries, and qrels for a BEIR-style benchmark."""

    corpus: list[Document]
    queries: list[Query]
    qrels: Qrels


class BeirLoader:
    """Load a BEIR-style benchmark from HuggingFace.

    Args:
        corpus_dataset: HF dataset id for the corpus subset.
        corpus_config: HF config name for the corpus subset.
        corpus_split: HF split for the corpus.
        queries_dataset: HF dataset id for the queries subset.
        queries_config: HF config name for the queries subset.
        queries_split: HF split for the queries.
        qrels_dataset: HF dataset id for the qrels.
        qrels_split: HF split for the qrels.
        cache_dir: Optional cache directory for HF downloads.
    """

    def __init__(
        self,
        *,
        corpus_dataset: str = "BeIR/nq",
        corpus_config: str = "corpus",
        corpus_split: str = "corpus",
        queries_dataset: str = "BeIR/nq",
        queries_config: str = "queries",
        queries_split: str = "queries",
        qrels_dataset: str = "BeIR/nq-qrels",
        qrels_split: str = "test",
        cache_dir: Path | None = None,
    ) -> None:
        self.corpus_dataset = corpus_dataset
        self.corpus_config = corpus_config
        self.corpus_split = corpus_split
        self.queries_dataset = queries_dataset
        self.queries_config = queries_config
        self.queries_split = queries_split
        self.qrels_dataset = qrels_dataset
        self.qrels_split = qrels_split
        self.cache_dir = cache_dir

    def load(self, *, max_corpus: int | None = None) -> BeirSplit:
        """Load corpus, queries, and qrels.

        Args:
            max_corpus: Optional cap on corpus size; primarily for smoke
                tests. The cap does not enforce qrels coverage — for a
                full evaluation, pass ``None``.

        Returns:
            A :class:`BeirSplit`.
        """

        corpus = list(self._load_corpus(max_corpus=max_corpus))
        queries = list(self._load_queries())
        qrels = self._load_qrels()

        logger.info(
            "loaded BEIR-style benchmark: "
            "{n_corpus} docs, {n_queries} queries, {n_judged} judged queries",
            n_corpus=len(corpus),
            n_queries=len(queries),
            n_judged=len(qrels),
        )
        return BeirSplit(corpus=corpus, queries=queries, qrels=qrels)

    def _load_corpus(self, *, max_corpus: int | None) -> Iterable[Document]:
        from datasets import load_dataset

        ds = load_dataset(
            self.corpus_dataset,
            self.corpus_config,
            split=self.corpus_split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        if max_corpus is not None and max_corpus > 0:
            ds = ds.select(range(min(max_corpus, len(ds))))

        for row in ds:
            yield self._row_to_document(row)

    def _load_queries(self) -> Iterable[Query]:
        from datasets import load_dataset

        ds = load_dataset(
            self.queries_dataset,
            self.queries_config,
            split=self.queries_split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        for row in ds:
            yield self._row_to_query(row)

    def _load_qrels(self) -> Qrels:
        from datasets import load_dataset

        ds = load_dataset(
            self.qrels_dataset,
            split=self.qrels_split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        # The qrels HF datasets use lower-case column names ``query-id``,
        # ``corpus-id``, and ``score`` (with the dash, not underscore).
        # Be liberal in what we accept to survive minor schema drift.
        qrels: Qrels = {}
        qid_keys = ("query-id", "query_id", "qid")
        did_keys = ("corpus-id", "corpus_id", "doc-id", "doc_id", "did")
        score_keys = ("score", "relevance", "rel")

        for row in ds:
            qid = _first_present(row, qid_keys)
            did = _first_present(row, did_keys)
            score = _first_present(row, score_keys)
            if qid is None or did is None or score is None:
                raise KeyError(
                    f"qrels row missing required keys (qid/did/score) in {sorted(row.keys())}",
                )
            qrels.setdefault(str(qid), {})[str(did)] = int(score)
        return qrels

    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> Document:
        doc_id = str(row.get("_id") or row.get("id") or "")
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        if not doc_id:
            raise KeyError(f"corpus row missing _id/id, keys={sorted(row.keys())}")
        # BEIR convention: prepend title to text for retrieval if both present.
        body = f"{title}\n\n{text}".strip() if title else text
        return Document(
            doc_id=doc_id,
            text=body,
            title=title,
            source=f"hf://{doc_id}",
            metadata={"raw_text": text},
        )

    @staticmethod
    def _row_to_query(row: dict[str, Any]) -> Query:
        qid = str(row.get("_id") or row.get("id") or "")
        # On BeIR/nq the human-readable query lives in ``text``; on some
        # BEIR mirrors it lives in ``title``. Prefer non-empty.
        text = str(row.get("text") or "").strip()
        if not text:
            text = str(row.get("title") or "").strip()
        if not qid or not text:
            raise KeyError(
                f"query row missing _id/text, keys={sorted(row.keys())}",
            )
        return Query(query_id=qid, text=text)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None
