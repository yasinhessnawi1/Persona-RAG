"""Build (or update) the Chroma index for a configured (corpus, embedder) pair.

Loads the corpus, chunks it, encodes the chunks via the embedder's
``encode_passages``, and upserts into a Chroma collection whose name is
deterministically derived from ``(corpus, embedder, chunk_size, overlap)``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from option8_rag.chunking.splitter import SentenceChunker
from option8_rag.cli._common import resolve_paths, set_seed, setup_logging
from option8_rag.embeddings.encoder import TextEncoder
from option8_rag.index.chroma_index import ChromaIndex, IndexConfig
from option8_rag.ingest.beir import BeirLoader
from option8_rag.types import Chunk, Document


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

    embeddings = _encode_in_batches(encoder=encoder, chunks=chunks)
    index.upsert(chunks=chunks, embeddings=embeddings)

    summary = {
        "corpus": cfg.corpus.name,
        "embedder": cfg.embedder.name,
        "chunk_size": cfg.corpus.chunking.size,
        "chunk_overlap": cfg.corpus.chunking.overlap,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
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
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title="",
                source=str(raw_dir / f"{doc_id}.html"),
                metadata={"text_path": str(text_path)},
            ),
        )
    return documents


def _encode_in_batches(*, encoder: TextEncoder, chunks: list[Chunk]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    bs = max(1, int(encoder.batch_size))
    for start in range(0, len(chunks), bs):
        batch = chunks[start : start + bs]
        pieces.append(encoder.encode_passages([c.text for c in batch]))
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, encoder.dimension))


if __name__ == "__main__":
    main()
