"""Run a baseline retrieval pipeline (vanilla RAG or prompt-persona) end-to-end.

Hydra entry point composes ``model + knowledge_store + retrieval`` from
``src/persona_rag/config/``. Loads the model, indexes the knowledge corpus
(idempotent — re-runs are cheap), runs each test query through the configured
pipeline, and writes per-query response artefacts + a summary JSON.

Examples:
    uv run python scripts/run_baseline.py
    uv run python scripts/run_baseline.py retrieval=prompt_persona
    uv run python scripts/run_baseline.py model=llama retrieval=prompt_persona
    uv run python scripts/run_baseline.py \\
        retrieval=prompt_persona retrieval.b2_variant=v02_one_liner
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.retrieval import (
    FewShotBundle,
    PromptPersonaRAG,
    Response,
    VanillaRAG,
)
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_store import KnowledgeDocument, KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path) -> Path:
    """Make config-relative paths absolute against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _load_corpus(corpus_path: Path) -> list[KnowledgeDocument]:
    """Load every .md file under `corpus_path` into a `KnowledgeDocument`."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Knowledge corpus not found at {corpus_path}. "
            "Default lives at benchmarks_data/knowledge_corpora/cs_tutor/."
        )
    docs: list[KnowledgeDocument] = []
    for md_path in sorted(corpus_path.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(
            KnowledgeDocument(
                doc_id=md_path.stem,
                text=text,
                source=md_path.name,
                metadata={"corpus": corpus_path.name},
            )
        )
    if not docs:
        raise FileNotFoundError(f"No .md files under {corpus_path}")
    logger.info("Loaded {} document(s) from {}", len(docs), corpus_path)
    return docs


def _build_backend(model_cfg: DictConfig) -> Any:
    """Construct a Gemma or Llama LLMBackend from the Hydra model group.

    Mirrors ``scripts/smoke_test_models.py::_build_backend`` so both scripts
    share one pattern for backend construction.
    """
    from persona_rag.models import GemmaBackend, HFBackendConfig, LlamaBackend

    cfg = HFBackendConfig(
        model_id=model_cfg.model_id,
        name=model_cfg.name,
        revision=model_cfg.revision,
        compute_dtype=model_cfg.compute_dtype,
        attn_implementation=model_cfg.attn_implementation,
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        max_input_tokens=model_cfg.max_input_tokens,
        trust_remote_code=model_cfg.trust_remote_code,
        warmup_nan_guard=bool(model_cfg.get("warmup_nan_guard", True)),
    )
    name = str(model_cfg.name)
    if name.startswith("gemma"):
        return GemmaBackend(cfg)
    if name.startswith("llama"):
        return LlamaBackend(cfg)
    raise ValueError(f"Unknown model.name {name!r} — expected gemma* or llama*")


def _build_baseline(
    cfg: DictConfig,
    backend: Any,
    knowledge_store: KnowledgeStore,
    persona: Persona,
) -> Any:
    """Construct the configured baseline pipeline from `cfg.retrieval.type`."""
    rcfg = cfg.retrieval
    btype = str(rcfg.type)
    if btype == "b1":
        return VanillaRAG(
            backend=backend,
            knowledge_store=knowledge_store,
            top_k=int(rcfg.top_k),
            candidate_pool=int(rcfg.candidate_pool),
            alpha=rcfg.alpha,
            max_new_tokens=int(rcfg.max_new_tokens),
            max_input_tokens=int(rcfg.max_input_tokens),
        )
    if btype == "b2":
        few_shots_path = _resolve(rcfg.few_shots_dir) / f"{persona.persona_id}.yaml"
        few_shots = FewShotBundle.from_yaml(few_shots_path)
        return PromptPersonaRAG(
            backend=backend,
            knowledge_store=knowledge_store,
            few_shots=few_shots,
            top_k=int(rcfg.top_k),
            candidate_pool=int(rcfg.candidate_pool),
            alpha=rcfg.alpha,
            max_new_tokens=int(rcfg.max_new_tokens),
            max_input_tokens=int(rcfg.max_input_tokens),
            b2_variant=str(rcfg.b2_variant),
        )
    raise ValueError(f"Unknown retrieval.type {btype!r} — expected b1 or b2")


def _init_wandb(cfg: DictConfig, persona_id: str) -> Any | None:
    """Initialise wandb if available + permitted by config; return run handle or None."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping experiment logging")
        return None
    mode = os.environ.get("WANDB_MODE") or str(cfg.wandb.mode)
    run_name = (
        cfg.wandb.run_name_template
        if cfg.wandb.run_name_template is not None
        else f"{cfg.retrieval.name}_{cfg.model.name}_{persona_id}"
    )
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=mode,
        name=str(run_name),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(_resolve(cfg.report_dir)),
    )


def _write_response(report_dir: Path, ix: int, query: str, response: Response) -> None:
    """Dump one query/response pair as JSON for later inspection."""
    out = report_dir / f"response_{ix:02d}.json"
    payload = {
        "query": query,
        "text": response.text,
        "prompt_used": response.prompt_used,
        "metadata": response.metadata,
        "retrieved_knowledge": [
            {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance}
            for c in response.retrieved_knowledge
        ],
        "retrieved_persona": {
            kind: [
                {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance}
                for c in chunks
            ]
            for kind, chunks in response.retrieved_persona.items()
        },
        "steering_applied": response.steering_applied,
        "drift_signal": response.drift_signal,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path="../src/persona_rag/config", config_name="baseline")
def main(cfg: DictConfig) -> int:
    """Hydra entry point. Returns 0 on success, nonzero on failure."""
    report_dir = Path(cfg.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Spec-04 baseline run — report_dir={}", report_dir)
    logger.info("Resolved config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    # Persona.
    persona_path = _resolve(cfg.personas_dir) / f"{cfg.persona_id}.yaml"
    persona = Persona.from_yaml(persona_path)
    logger.info("Persona loaded: {!r} ({})", persona.persona_id, persona.identity.name)

    # Knowledge store + corpus.
    ks_cfg = cfg.knowledge_store
    knowledge_store = KnowledgeStore(
        persist_path=_resolve(ks_cfg.persist_path),
        collection_name=str(ks_cfg.collection_name),
        embedding_model=str(ks_cfg.embedding_model),
        chunk_size=int(ks_cfg.chunk_size),
        chunk_overlap=int(ks_cfg.chunk_overlap),
    )
    corpus = _load_corpus(_resolve(ks_cfg.corpus_path))
    indexed = knowledge_store.index_corpus(corpus)
    logger.info(
        "Knowledge store: {} chunks indexed (total in collection: {})",
        indexed,
        knowledge_store.count(),
    )

    # Backend + baseline.
    backend = _build_backend(cfg.model)
    baseline = _build_baseline(cfg, backend, knowledge_store, persona)
    logger.info("Baseline: {} ({})", baseline.name, type(baseline).__name__)

    # wandb (best-effort).
    wandb_run = _init_wandb(cfg, persona.persona_id or "<unknown>")

    # Run queries.
    summary: dict[str, Any] = {
        "persona_id": persona.persona_id,
        "backend": backend.name,
        "baseline": baseline.name,
        "queries_n": len(cfg.test_queries),
        "results": [],
    }
    for ix, query in enumerate(cfg.test_queries):
        logger.info("Q{}: {}", ix, query)
        response = baseline.respond(query, persona)
        _write_response(report_dir, ix, query, response)
        summary["results"].append(
            {
                "ix": ix,
                "query": query,
                "text_preview": response.text[:200],
                "trimmed_chunks": response.metadata.get("trimmed_chunks"),
                "retrieved_knowledge_n": len(response.retrieved_knowledge),
            }
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "query_ix": ix,
                    "trimmed_chunks": response.metadata.get("trimmed_chunks"),
                    "retrieved_knowledge_n": len(response.retrieved_knowledge),
                }
            )

    summary_path = report_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Summary written to {}", summary_path)

    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
