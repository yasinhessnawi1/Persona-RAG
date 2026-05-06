"""Run retrieval + (optional) end-to-end evaluation for the configured stack.

For ``corpus=beir_nq`` the CLI computes retrieval metrics
(Recall@k, nDCG@k, MAP@k, MRR@k) over the test queries with qrels.

For ``corpus=uia_ikt`` the CLI expects a held-out QA file at
``corpus.qa_path`` (JSONL with ``{question, answers, [reference_chunks]}``)
and reports EM, F1, and a groundedness score.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from option8_rag.cli._common import (
    cfg_to_dict,
    resolve_paths,
    set_seed,
    setup_logging,
)
from option8_rag.embeddings.encoder import TextEncoder
from option8_rag.evaluate.qa_metrics import aggregate as aggregate_qa
from option8_rag.evaluate.retrieval_metrics import evaluate_retrieval
from option8_rag.index.chroma_index import ChromaIndex, IndexConfig
from option8_rag.ingest.beir import BeirLoader
from option8_rag.retrieve.bm25 import Bm25Retriever
from option8_rag.retrieve.dense import DenseRetriever
from option8_rag.retrieve.hybrid import HybridRetriever
from option8_rag.synthesize.generator import (
    GenerationConfig,
    GroundedGenerator,
)
from option8_rag.types import Chunk, Query, Run


@hydra.main(
    config_path="../config",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    set_seed(int(cfg.seed))
    paths = resolve_paths(cfg)

    name = str(cfg.corpus.name)
    if name == "beir_nq":
        results = _eval_beir(cfg=cfg, paths=paths)
    elif name == "uia_ikt":
        results = _eval_uia(cfg=cfg, paths=paths)
    else:
        raise ValueError(f"unknown corpus {name!r}")

    out_path = paths["results_root"] / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("metrics written to {p}", p=str(out_path))
    print(json.dumps(results, indent=2))


# ---------- BEIR/NQ ------------------------------------------------------


def _eval_beir(*, cfg: DictConfig, paths: dict[str, Path]) -> dict[str, object]:
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
    smoke = bool(cfg.run.smoke)
    full_corpus = bool(cfg.corpus.full_corpus)
    if smoke:
        max_corpus = int(cfg.run.smoke_max_chunks)
    elif full_corpus:
        max_corpus = None
    else:
        max_corpus = int(cfg.corpus.subsample_size)

    split = loader.load(max_corpus=max_corpus)
    queries = split.queries
    qrels = split.qrels
    if smoke:
        cap = int(cfg.run.smoke_max_queries)
        queries = queries[:cap]
        qrels = {
            qid: rels for qid, rels in qrels.items() if any(q.query_id == qid for q in queries)
        }

    encoder, index = _build_dense_stack(cfg=cfg, paths=paths)
    retriever = _build_retriever(cfg=cfg, encoder=encoder, index=index, paths=paths)

    top_k = int(cfg.retriever.top_k)
    logger.info("retrieving for {n} queries (top_k={k})", n=len(queries), k=top_k)
    retrieved = retriever.retrieve(queries, top_k=top_k)
    run: Run = {
        q.query_id: {item.chunk.doc_id: float(item.score) for item in row}
        for q, row in zip(queries, retrieved, strict=True)
    }

    metrics = evaluate_retrieval(qrels=qrels, run=run, k_values=(1, 5, 10))
    return {
        "corpus": "beir_nq",
        "embedder": str(cfg.embedder.name),
        "retriever": str(cfg.retriever.name),
        "n_queries": len(queries),
        "metrics": metrics,
        "config": cfg_to_dict(cfg),
    }


# ---------- UiA-IKT -----------------------------------------------------


def _eval_uia(*, cfg: DictConfig, paths: dict[str, Path]) -> dict[str, object]:
    qa_path = Path(str(cfg.corpus.qa_path)).expanduser().resolve()
    if not qa_path.exists():
        raise FileNotFoundError(
            f"held-out QA file not found at {qa_path}; create one with the "
            "schema: {question, answers: [...], [reference_chunks]}",
        )
    items = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]
    queries = [Query(query_id=str(i), text=str(item["question"])) for i, item in enumerate(items)]

    encoder, index = _build_dense_stack(cfg=cfg, paths=paths)
    retriever = _build_retriever(cfg=cfg, encoder=encoder, index=index, paths=paths)
    retrieved = retriever.retrieve(queries, top_k=int(cfg.retriever.top_k))

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
        prompt_style=str(cfg.generator.get("prompt_style", "verbose")),
    )
    predictions: list[str] = []
    answers_per_q: list[list[str]] = []
    for q, row, item in zip(queries, retrieved, items, strict=True):
        out = generator.generate(query=q.text, retrieved=row)
        predictions.append(out.answer)
        answers_per_q.append([str(a) for a in item.get("answers", [])])

    qa_scores = aggregate_qa(predictions=predictions, references=answers_per_q)

    # Free the generator's GPU footprint before loading the judge.
    # Without this, both 8B+ models try to live on the V100 and accelerate
    # silently CPU-offloads the second one (visible as a "Some parameters
    # are on the meta device because they were offloaded to the cpu"
    # warning), which breaks generation under torch.compile / dynamo.
    generator.unload()

    grounded_scores: list[float] = []
    if not bool(cfg.run.smoke):
        from option8_rag.evaluate.groundedness import GroundednessJudge, JudgeConfig

        judge = GroundednessJudge(
            model_id=str(cfg.judge.model_id),
            device=str(cfg.judge.device),
            dtype=str(cfg.judge.dtype),
            attn_implementation=str(cfg.judge.attn_implementation),
            config=JudgeConfig(
                max_new_tokens=int(cfg.judge.max_new_tokens),
                temperature=float(cfg.judge.temperature),
                top_p=float(cfg.judge.top_p),
                do_sample=bool(cfg.judge.do_sample),
            ),
        )
        for q, pred, row in zip(queries, predictions, retrieved, strict=True):
            score = judge.score(question=q.text, answer=pred, retrieved=row)
            if score.fused is not None:
                grounded_scores.append(score.fused)

    grounded_mean = sum(grounded_scores) / len(grounded_scores) if grounded_scores else None

    return {
        "corpus": "uia_ikt",
        "embedder": str(cfg.embedder.name),
        "retriever": str(cfg.retriever.name),
        "generator": str(cfg.generator.name),
        "judge": str(cfg.judge.name),
        "n_queries": len(queries),
        "metrics": {
            **qa_scores,
            "groundedness": grounded_mean,
        },
        "config": cfg_to_dict(cfg),
    }


# ---------- shared builders --------------------------------------------


def _build_dense_stack(
    *,
    cfg: DictConfig,
    paths: dict[str, Path],
) -> tuple[TextEncoder, ChromaIndex]:
    encoder = TextEncoder(
        model_id=str(cfg.embedder.model_id),
        query_prefix=str(cfg.embedder.query_prefix),
        passage_prefix=str(cfg.embedder.passage_prefix),
        normalize=bool(cfg.embedder.normalize),
        batch_size=int(cfg.embedder.batch_size),
        device=str(cfg.embedder.device),
        dtype=str(cfg.embedder.dtype),
    )
    space = "cosine"
    if "chroma" in cfg.retriever and "hnsw_space" in cfg.retriever.chroma:
        space = str(cfg.retriever.chroma.hnsw_space)
    index = ChromaIndex(
        persist_path=paths["chroma_root"],
        config=IndexConfig(
            corpus_name=str(cfg.corpus.name),
            embedder_name=str(cfg.embedder.name),
            chunk_size=int(cfg.corpus.chunking.size),
            chunk_overlap=int(cfg.corpus.chunking.overlap),
            hnsw_space=space,
        ),
    )
    return encoder, index


def _build_retriever(
    *,
    cfg: DictConfig,
    encoder: TextEncoder,
    index: ChromaIndex,
    paths: dict,
):
    name = str(cfg.retriever.name)
    if name == "dense":
        retriever = DenseRetriever(encoder=encoder, index=index, top_k=int(cfg.retriever.top_k))
    elif name == "bm25":
        chunks = _materialise_chunks_for_bm25(cfg=cfg, paths=paths)
        retriever = Bm25Retriever(
            chunks=chunks,
            top_k=int(cfg.retriever.top_k),
            lowercase=bool(cfg.retriever.lowercase),
        )
    elif name == "hybrid":
        chunks = _materialise_chunks_for_bm25(cfg=cfg, paths=paths)
        bm25 = Bm25Retriever(
            chunks=chunks,
            top_k=int(cfg.retriever.bm25_top_n),
            lowercase=bool(cfg.retriever.lowercase),
        )
        dense = DenseRetriever(encoder=encoder, index=index, top_k=int(cfg.retriever.dense_top_n))
        retriever = HybridRetriever(
            dense=dense,
            bm25=bm25,
            top_k=int(cfg.retriever.top_k),
            dense_top_n=int(cfg.retriever.dense_top_n),
            bm25_top_n=int(cfg.retriever.bm25_top_n),
            rrf_k=int(cfg.retriever.rrf_k),
        )
    else:
        raise ValueError(f"unknown retriever {name!r}")

    # Optional cross-encoder reranker — wraps any base retriever.
    if "reranker" in cfg and bool(cfg.reranker.get("enabled", False)):
        from option8_rag.retrieve.rerank import CrossEncoderReranker, RerankRetriever

        reranker = CrossEncoderReranker(
            model_id=str(cfg.reranker.model_id),
            device=str(cfg.reranker.get("device", "auto")),
            batch_size=int(cfg.reranker.get("batch_size", 32)),
        )
        retriever = RerankRetriever(
            base=retriever,
            reranker=reranker,
            base_top_n=int(cfg.reranker.base_top_n),
            top_k=int(cfg.retriever.top_k),
        )
    return retriever


def _materialise_chunks_for_bm25(*, cfg: DictConfig, paths: dict) -> list[Chunk]:
    """Load the chunk JSONL dump produced by the index step.

    The BM25 / hybrid path needs the same chunked corpus the dense
    index sees. We persist it as JSONL during the index step (see
    `option8_rag.cli.index`); here we just read it back.
    """

    from option8_rag.index.chunk_store import chunks_dump_path, read_chunks_jsonl

    path = chunks_dump_path(
        chroma_root=paths["chroma_root"],
        corpus_name=str(cfg.corpus.name),
        chunk_size=int(cfg.corpus.chunking.size),
        chunk_overlap=int(cfg.corpus.chunking.overlap),
    )
    chunks = read_chunks_jsonl(path)
    if not chunks:
        raise RuntimeError(
            f"chunk dump at {path} is empty; rerun the index step for this corpus.",
        )
    return chunks


if __name__ == "__main__":
    main()
