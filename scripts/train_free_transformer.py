"""Train one Free Transformer on the persona-labelled corpus (off-spec experiment).

Usage on V100 (per CLAUDE.md: bitsandbytes is Linux-only, but this script does
NOT use bitsandbytes — only torch + transformers tokenizer):

    # First time: build the corpus.
    uv run python scripts/train_free_transformer.py build_corpus=true

    # κ = 0.5 bits/token (paper's headline value for the 1T-token 8B run).
    uv run python scripts/train_free_transformer.py freet.free_bits_kappa_bits=0.5

    # Sweep the paper band:
    uv run python scripts/train_free_transformer.py freet.free_bits_kappa_bits=0.25
    uv run python scripts/train_free_transformer.py freet.free_bits_kappa_bits=1.0

The script writes a checkpoint at ``results/freet/<run_id>/final.pt``. Pass
that path to ``scripts/validate_freet_z_separation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.freet.corpus import CorpusBuildConfig, build_corpus
from persona_rag.freet.train import TrainConfig, train


@hydra.main(config_path="../src/persona_rag/config", config_name="train_freet", version_base=None)
def main(cfg: DictConfig) -> int:
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "driver.log", level="DEBUG")
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    corpus_dir = Path(cfg.freet.corpus_dir)
    if cfg.build_corpus or not (corpus_dir / "train.jsonl").exists():
        logger.info("building corpus at {}", corpus_dir)
        build_corpus(corpus_dir, CorpusBuildConfig(seed=int(cfg.freet.seed)))
    else:
        logger.info("re-using corpus at {}", corpus_dir)

    train_cfg = TrainConfig(
        out_dir=report_dir,
        train_jsonl=corpus_dir / "train.jsonl",
        test_jsonl=corpus_dir / "test.jsonl",
        tokenizer_id=str(cfg.freet.tokenizer_id),
        seq_len=int(cfg.freet.seq_len),
        dim=int(cfg.freet.dim),
        n_layers=int(cfg.freet.n_layers),
        n_q_heads=int(cfg.freet.n_q_heads),
        n_kv_heads=int(cfg.freet.n_kv_heads),
        mlp_ratio=int(cfg.freet.mlp_ratio),
        latent_bits=int(cfg.freet.latent_bits),
        inject_after_layer=cfg.freet.inject_after_layer,
        batch_size=int(cfg.freet.batch_size),
        grad_accum=int(cfg.freet.grad_accum),
        lr=float(cfg.freet.lr),
        weight_decay=float(cfg.freet.weight_decay),
        warmup_steps=int(cfg.freet.warmup_steps),
        max_steps=int(cfg.freet.max_steps),
        grad_clip=float(cfg.freet.grad_clip),
        free_bits_kappa_bits=float(cfg.freet.free_bits_kappa_bits),
        log_every=int(cfg.freet.log_every),
        eval_every=int(cfg.freet.eval_every),
        save_every=int(cfg.freet.save_every),
        seed=int(cfg.freet.seed),
        fp16=bool(cfg.freet.fp16),
    )
    final_ckpt = train(train_cfg)
    logger.info("training complete. final checkpoint: {}", final_ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
