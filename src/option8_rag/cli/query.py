"""One-shot interactive query: retrieve and (optionally) synthesise an answer."""

from __future__ import annotations

import sys

import hydra
from loguru import logger
from omegaconf import DictConfig

from option8_rag.cli._common import resolve_paths, set_seed, setup_logging
from option8_rag.embeddings.encoder import TextEncoder
from option8_rag.index.chroma_index import ChromaIndex, IndexConfig
from option8_rag.retrieve.dense import DenseRetriever
from option8_rag.synthesize.generator import (
    GenerationConfig,
    GroundedGenerator,
)
from option8_rag.types import Query


@hydra.main(
    config_path="../config",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    set_seed(int(cfg.seed))
    paths = resolve_paths(cfg)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Hydra strips its own overrides; positional argv survives only when
        # invoked through a thin shim. Use ${query} on the CLI in practice.
        question = " ".join(sys.argv[1:])
    elif "query" in cfg and cfg.query:
        question = str(cfg.query)
    else:
        question = input("query> ").strip()

    if not question:
        logger.error("empty query")
        sys.exit(2)

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
            hnsw_space=str(cfg.retriever.chroma.hnsw_space),
        ),
    )
    retriever = DenseRetriever(
        encoder=encoder,
        index=index,
        top_k=int(cfg.retriever.top_k),
    )
    retrieved = retriever.retrieve([Query(query_id="adhoc", text=question)])[0]

    print(f"\n=== retrieved (top-{len(retrieved)}) ===")
    for i, item in enumerate(retrieved, start=1):
        head = item.chunk.text[:160].replace("\n", " ")
        print(f"[{i}] score={item.score:.4f} chunk={item.chunk.chunk_id}\n    {head}...")

    generator = GroundedGenerator(
        model_id=str(cfg.generator.model_id),
        device=str(cfg.generator.device),
        dtype=str(cfg.generator.dtype),
        attn_implementation=str(cfg.generator.attn_implementation),
        generation=GenerationConfig(
            max_new_tokens=int(cfg.generator.max_new_tokens),
            temperature=float(cfg.generator.temperature),
            top_p=float(cfg.generator.top_p),
            do_sample=bool(cfg.generator.do_sample),
        ),
        context_top_k=int(cfg.generator.context_top_k),
    )
    answer = generator.generate(query=question, retrieved=retrieved)

    print("\n=== answer ===")
    print(answer.answer)


if __name__ == "__main__":
    main()
