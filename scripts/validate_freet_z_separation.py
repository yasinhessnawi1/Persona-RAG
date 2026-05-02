"""Validate that Z separates personas — analogue of `validate_persona_vectors.py`.

Usage:

    uv run python scripts/validate_freet_z_separation.py \\
        ckpt=results/freet/20260502_xxxxxx_kappa0.5/final.pt

Loads the checkpoint, runs the encoder on the test split of the persona-
labelled corpus, computes per-persona one-vs-rest AUROC of a logistic
regression on the encoder's H-dim mean-pooled feature, runs Dubanowska
shuffled-label and random-feature controls, draws a UMAP figure of the
encoder features coloured by persona, and runs the encoder turn-by-turn on
the drift trajectories.

Writes:

  - z_separability.json    structured numbers (replayable to wandb later)
  - verdict.md             human-readable per-persona table + verdict
  - umap_freet.png         encoder feature UMAP, coloured by persona
  - drift_trajectory.json  per-turn encoder logits for in_persona / drifting
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.freet.analysis import (
    encode_dataset,
    encoder_classification_accuracy,
    load_checkpoint,
    run_drift_trajectory,
    run_separability,
    umap_figure,
    write_verdict,
)
from persona_rag.freet.corpus import iter_jsonl
from persona_rag.freet.tokenizer import load_tokenizer


@hydra.main(config_path="../src/persona_rag/config", config_name="validate_freet", version_base=None)
def main(cfg: DictConfig) -> int:
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "validation.log", level="DEBUG")
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    if cfg.ckpt in (None, "???"):
        logger.error("ckpt path is required (e.g. ckpt=results/freet/.../final.pt)")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda={}", device, torch.cuda.is_available())

    model, ckpt_meta = load_checkpoint(Path(cfg.ckpt), device)
    train_cfg_blob = ckpt_meta.get("train_cfg", {}) or {}
    seq_len = int(cfg.freet.seq_len)
    tokenizer_id = str(cfg.freet.tokenizer_id)
    if "tokenizer_id" in train_cfg_blob:
        tokenizer_id = str(train_cfg_blob["tokenizer_id"])
    if "seq_len" in train_cfg_blob:
        seq_len = int(train_cfg_blob["seq_len"])
    logger.info("loaded checkpoint at step {} (kappa_bits={})",
                ckpt_meta.get("step"), train_cfg_blob.get("free_bits_kappa_bits"))

    tokenizer = load_tokenizer(tokenizer_id)

    corpus_dir = Path(cfg.freet.corpus_dir)
    test_path = corpus_dir / "test.jsonl"
    if not test_path.exists():
        logger.error("test corpus not found at {}; run train_free_transformer.py "
                     "with build_corpus=true first", test_path)
        return 2

    examples = list(iter_jsonl(test_path))
    if cfg.max_eval and len(examples) > int(cfg.max_eval):
        examples = examples[: int(cfg.max_eval)]
    logger.info("encoding {} held-out examples", len(examples))
    features = encode_dataset(model, examples, tokenizer, seq_len, device)

    sep = run_separability(
        features,
        train_frac=float(cfg.probe_train_fraction),
        seed=int(cfg.freet.seed),
        n_shuffles=int(cfg.n_shuffles),
        n_random_repeats=int(cfg.n_random_repeats),
    )

    # UMAP figure.
    try:
        umap_figure(features, report_dir / "umap_freet.png")
    except Exception as exc:
        logger.warning("UMAP figure failed: {}", exc)

    # Drift trajectory probe (turn-by-turn encoder logits).
    drift_rows = run_drift_trajectory(model, tokenizer, seq_len, device)
    (report_dir / "drift_trajectory.json").write_text(
        json.dumps([asdict(r) for r in drift_rows], indent=2) + "\n", encoding="utf-8"
    )

    # Supervised mode: read the persona-id map saved alongside the checkpoint
    # and compute direct encoder→persona classification accuracy.
    classifier_diag = None
    pid_map = ckpt_meta.get("persona_id_map")
    latent_mode = (ckpt_meta.get("model_cfg") or {}).get("latent_mode", "unsupervised")
    if latent_mode == "supervised" and pid_map:
        classifier_diag = encoder_classification_accuracy(features, pid_map)
        logger.info(
            "supervised classifier diag: overall_accuracy={:.3f}",
            classifier_diag["overall_accuracy"],
        )

    write_verdict(report_dir, sep, ckpt_meta, classifier_diag=classifier_diag)
    logger.info("Z-separability verdict = {} (macro AUROC = {:.3f})",
                sep.overall_verdict, sep.macro_auroc)
    return 0 if sep.overall_verdict != "refuted" else 2


if __name__ == "__main__":
    sys.exit(main())
