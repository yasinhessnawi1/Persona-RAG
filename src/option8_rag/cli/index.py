"""Build (or update) the Chroma index for a configured (corpus, embedder) pair.

Loads the corpus, chunks it, encodes the chunks via the embedder's
``encode_passages``, and upserts into a Chroma collection whose name is
deterministically derived from ``(corpus, embedder, chunk_size, overlap)``.

The embed + upsert step streams in mini-batches: for each slice of
``upsert_batch_size`` chunks we encode and immediately upsert into Chroma.
This keeps peak RAM bounded (we never materialise the full ``(N, dim)``
embedding tensor) and stays under Chroma's hard ~5,461 row cap on a
single ``upsert`` call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from option8_rag.chunking.splitter import SentenceChunker
from option8_rag.cli._common import resolve_paths, set_seed, setup_logging
from option8_rag.embeddings.encoder import TextEncoder
from option8_rag.index.chroma_index import ChromaIndex, IndexConfig
from option8_rag.index.chunk_store import chunks_dump_path, write_chunks_jsonl
from option8_rag.ingest.beir import BeirLoader
from option8_rag.types import Chunk, Document

# Chroma's single-call upsert cap is ~5,461 rows in current builds; keep a
# margin so future bumps don't surprise us.
_CHROMA_UPSERT_CAP = 5000


@hydra.main(
    config_path="../config",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    set_seed(int(cfg.seed))
    paths = resolve_paths(cfg)

    documents = list(_load_documents(cfg, paths))
    chunker = SentenceChunker(
        chunk_size=int(cfg.corpus.chunking.size),
        chunk_overlap=int(cfg.corpus.chunking.overlap),
    )
    chunks: list[Chunk] = chunker.chunk_documents(documents)

    # For UiA, repeat the synthesised course header at the head of every
    # chunk so retrieval pulls back the title + leader + ECTS regardless
    # of which paragraph the SentenceSplitter put them in.
    if str(cfg.corpus.name) == "uia_ikt":
        chunks = _prefix_uia_chunks_with_header(chunks, documents)

    encoder = TextEncoder(
        model_id=str(cfg.embedder.model_id),
        query_prefix=str(cfg.embedder.query_prefix),
        passage_prefix=str(cfg.embedder.passage_prefix),
        normalize=bool(cfg.embedder.normalize),
        batch_size=int(cfg.embedder.batch_size),
        device=str(cfg.embedder.device),
        dtype=str(cfg.embedder.dtype),
    )
    index = ChromaIndex(
        persist_path=paths["chroma_root"],
        config=IndexConfig(
            corpus_name=str(cfg.corpus.name),
            embedder_name=str(cfg.embedder.name),
            chunk_size=int(cfg.corpus.chunking.size),
            chunk_overlap=int(cfg.corpus.chunking.overlap),
            hnsw_space=str(cfg.retriever.chroma.hnsw_space)
            if "chroma" in cfg.retriever and "hnsw_space" in cfg.retriever.chroma
            else "cosine",
        ),
    )

    if not chunks:
        logger.warning("no chunks produced; index unchanged (count={c})", c=index.count)
        return

    upsert_batch_size = _resolve_upsert_batch_size(cfg)
    embedding_dim = _stream_embed_and_upsert(
        encoder=encoder,
        index=index,
        chunks=chunks,
        upsert_batch_size=upsert_batch_size,
    )

    # Dump the chunked corpus to JSONL so BM25 / hybrid retrieval can
    # rebuild a sparse index later without redoing chunking. One file
    # per (corpus, chunk_size, chunk_overlap) — embedder-independent.
    chunks_path = chunks_dump_path(
        chroma_root=paths["chroma_root"],
        corpus_name=str(cfg.corpus.name),
        chunk_size=int(cfg.corpus.chunking.size),
        chunk_overlap=int(cfg.corpus.chunking.overlap),
    )
    write_chunks_jsonl(chunks_path, chunks)
    logger.info("chunks dumped to {p}", p=str(chunks_path))

    summary = {
        "corpus": cfg.corpus.name,
        "embedder": cfg.embedder.name,
        "chunk_size": cfg.corpus.chunking.size,
        "chunk_overlap": cfg.corpus.chunking.overlap,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_dim": embedding_dim,
        "upsert_batch_size": upsert_batch_size,
        "collection": index.config.collection_name,
        "count": index.count,
    }
    out = paths["results_root"] / "index_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("index summary written to {p}", p=str(out))


def _load_documents(cfg: DictConfig, paths: dict[str, Path]) -> Iterable[Document]:
    name = str(cfg.corpus.name)
    if name == "beir_nq":
        loader = BeirLoader(
            corpus_dataset=cfg.corpus.hf.corpus_dataset,
            corpus_config=cfg.corpus.hf.corpus_config,
            corpus_split=cfg.corpus.hf.corpus_split,
            queries_dataset=cfg.corpus.hf.queries_dataset,
            queries_config=cfg.corpus.hf.queries_config,
            queries_split=cfg.corpus.hf.queries_split,
            qrels_dataset=cfg.corpus.hf.qrels_dataset,
            qrels_split=cfg.corpus.hf.qrels_split,
            cache_dir=paths["cache_root"],
        )
        full_corpus = bool(cfg.corpus.full_corpus)
        smoke = bool(cfg.run.smoke)
        if smoke:
            max_corpus = int(cfg.run.smoke_max_chunks)
        elif full_corpus:
            max_corpus = None
        else:
            max_corpus = int(cfg.corpus.subsample_size)
        return loader.load(max_corpus=max_corpus).corpus
    if name == "uia_ikt":
        return _load_uia_documents(paths["raw_root"] / "uia_ikt")
    raise ValueError(f"unknown corpus {name!r}")


def _load_uia_documents(out_dir: Path) -> list[Document]:
    text_dir = out_dir / "text"
    raw_dir = out_dir / "raw"
    fields_dir = out_dir / "fields"
    if not text_dir.exists():
        raise FileNotFoundError(
            f"UiA text directory not found at {text_dir}; run the ingest step first.",
        )
    documents: list[Document] = []
    for text_path in sorted(text_dir.glob("*.txt")):
        doc_id = text_path.stem
        text = text_path.read_text()
        if not text.strip():
            continue

        import contextlib

        fields: dict[str, str] = {}
        fields_path = fields_dir / f"{doc_id}.json"
        if fields_path.exists():
            with contextlib.suppress(json.JSONDecodeError):
                fields = json.loads(fields_path.read_text())

        # Prepend a synthesised header to the page body. The chunker
        # then sees this header as the first sentence of the document,
        # which the SentenceSplitter keeps inside the first chunk; the
        # per-chunk header re-prefixing happens further down via the
        # post-chunk hook (see _prefix_uia_chunks below). Embedding the
        # header as part of the document text is also valuable on its
        # own because every chunk's overlap window can include it.
        header = ""
        try:
            from option8_rag.ingest.uia import synthesize_header

            header = synthesize_header(fields)
        except Exception:  # pragma: no cover — extraction is best-effort
            pass

        body = f"{header}\n\n{text}" if header else text

        metadata: dict[str, object] = {"text_path": str(text_path)}
        for k, v in fields.items():
            if isinstance(v, str | int | float | bool):
                metadata[k] = v

        documents.append(
            Document(
                doc_id=doc_id,
                text=body,
                title=str(fields.get("title", "")),
                source=str(raw_dir / f"{doc_id}.html"),
                metadata=metadata,
            ),
        )
    return documents


_UIA_HEADER_MARKER = "[course-header] "


def _prefix_uia_chunks_with_header(
    chunks: list[Chunk],
    documents: list[Document],
) -> list[Chunk]:
    """Prepend each UiA chunk with its source document's [course-header] line.

    The first chunk of each document already contains the header (it
    was prepended to the document text before chunking). For subsequent
    chunks the header has been lost, so we re-attach it. Idempotent:
    chunks that already start with the marker are left alone.
    """

    from dataclasses import replace

    header_by_doc: dict[str, str] = {}
    for doc in documents:
        text = doc.text or ""
        if text.startswith(_UIA_HEADER_MARKER):
            # Header is the first line of the document text.
            first_line = text.split("\n", 1)[0]
            header_by_doc[doc.doc_id] = first_line

    out: list[Chunk] = []
    rewritten = 0
    for chunk in chunks:
        header = header_by_doc.get(chunk.doc_id)
        if not header or chunk.text.startswith(_UIA_HEADER_MARKER):
            out.append(chunk)
            continue
        new_text = f"{header}\n\n{chunk.text}"
        out.append(replace(chunk, text=new_text))
        rewritten += 1
    if rewritten:
        logger.info(
            "prefixed {n} UiA chunks with course header (out of {total})",
            n=rewritten,
            total=len(chunks),
        )
    return out


def _resolve_upsert_batch_size(cfg: DictConfig) -> int:
    """Pull `index.upsert_batch_size` from config, clamping to Chroma's cap."""

    raw = None
    if "index" in cfg and "upsert_batch_size" in cfg.index:
        raw = int(cfg.index.upsert_batch_size)
    elif "upsert_batch_size" in cfg:
        raw = int(cfg.upsert_batch_size)
    if raw is None or raw <= 0:
        raw = _CHROMA_UPSERT_CAP
    return min(raw, _CHROMA_UPSERT_CAP)


def _stream_embed_and_upsert(
    *,
    encoder: TextEncoder,
    index: ChromaIndex,
    chunks: list[Chunk],
    upsert_batch_size: int,
) -> int:
    """Encode and upsert chunks in mini-batches; return embedding dim.

    Embedding the full corpus into a single ``(N, dim)`` tensor before
    writing exceeds RAM at BEIR/NQ scale and exceeds Chroma's per-call
    upsert cap. This streams the work: for each window we encode just
    that slice and upsert it before moving on.
    """

    n = len(chunks)
    bs = max(1, min(upsert_batch_size, _CHROMA_UPSERT_CAP))
    embedding_dim = 0
    log_every = max(1, n // 50)  # ~50 progress lines for the full run

    for start in range(0, n, bs):
        batch = chunks[start : start + bs]
        embeddings = encoder.encode_passages([c.text for c in batch])
        index.upsert(chunks=batch, embeddings=embeddings)
        embedding_dim = int(embeddings.shape[1])

        end = start + len(batch)
        if start == 0 or end == n or (end // log_every) > (start // log_every):
            logger.info(
                "indexed {done}/{total} chunks ({pct:.1f}%)",
                done=end,
                total=n,
                pct=100.0 * end / n,
            )
    return embedding_dim


if __name__ == "__main__":
    main()
