"""Train the Free-Transformer adapter on a frozen pretrained backbone.

Usage on V100 (huggingface-cli login required for gated models):

    # Reuse existing corpus.
    uv run python scripts/train_freet_adapter.py

    # Override backbone.
    uv run python scripts/train_freet_adapter.py \\
        freet.backbone_model_id=meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.freet.adapter_train import AdapterTrainConfig, train
from persona_rag.freet.corpus import CorpusBuildConfig, build_corpus


@hydra.main(
    config_path="../src/persona_rag/config",
    config_name="train_freet_adapter",
    version_base=None,
)
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

    train_cfg = AdapterTrainConfig(
        out_dir=report_dir,
        train_jsonl=corpus_dir / "train.jsonl",
        test_jsonl=corpus_dir / "test.jsonl",
        backbone_model_id=str(cfg.freet.backbone_model_id),
        seq_len=int(cfg.freet.seq_len),
        n_personas=int(cfg.freet.n_personas),
        encoder_mlp_ratio=int(cfg.freet.encoder_mlp_ratio),
        inject_after_layer=cfg.freet.inject_after_layer,
        persona_loss_weight=float(cfg.freet.persona_loss_weight),
        lora_rank=int(cfg.freet.get("lora_rank", 0)),
        lora_alpha=float(cfg.freet.get("lora_alpha", 16.0)),
        batch_size=int(cfg.freet.batch_size),
        grad_accum=int(cfg.freet.grad_accum),
        lr=float(cfg.freet.lr),
        weight_decay=float(cfg.freet.weight_decay),
        warmup_steps=int(cfg.freet.warmup_steps),
        max_steps=int(cfg.freet.max_steps),
        grad_clip=float(cfg.freet.grad_clip),
        log_every=int(cfg.freet.log_every),
        eval_every=int(cfg.freet.eval_every),
        save_every=int(cfg.freet.save_every),
        seed=int(cfg.freet.seed),
        fp16=bool(cfg.freet.fp16),
    )
    final_ckpt = train(train_cfg)
    logger.info("adapter training complete. final checkpoint: {}", final_ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
