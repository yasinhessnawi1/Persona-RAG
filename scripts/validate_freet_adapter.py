"""Validate Z separability for an adapter checkpoint.

Loads the adapter (re-instantiating the frozen backbone), runs the encoder
on a held-out subset of the persona corpus, computes per-persona AUROC and
direct classification accuracy. Mirrors `validate_freet_z_separation.py`
for the from-scratch model.
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

from persona_rag.freet.adapter_analysis import (
    encode_dataset,
    encoder_classification_accuracy,
    load_adapter_checkpoint,
)
from persona_rag.freet.analysis import (
    run_separability,
    umap_figure,
    write_verdict,
)
from persona_rag.freet.corpus import iter_jsonl


@hydra.main(
    config_path="../src/persona_rag/config",
    config_name="validate_freet_adapter",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "validation.log", level="DEBUG")
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    if cfg.ckpt in (None, "???"):
        logger.error("ckpt= is required (results/freet_adapter/.../final.pt)")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda={}", device, torch.cuda.is_available())

    adapter, ckpt_meta, tokenizer = load_adapter_checkpoint(Path(cfg.ckpt), device)
    seq_len = int(cfg.freet.seq_len)
    train_blob = ckpt_meta.get("train_cfg", {}) or {}
    if "seq_len" in train_blob:
        seq_len = int(train_blob["seq_len"])
    logger.info("loaded adapter checkpoint at step {}", ckpt_meta.get("step"))

    corpus_dir = Path(cfg.freet.corpus_dir)
    test_path = corpus_dir / "test.jsonl"
    if not test_path.exists():
        logger.error("test corpus not found at {}", test_path)
        return 2

    examples = list(iter_jsonl(test_path))
    if cfg.max_eval and len(examples) > int(cfg.max_eval):
        examples = examples[: int(cfg.max_eval)]
    logger.info("encoding {} held-out examples", len(examples))
    features = encode_dataset(adapter, examples, tokenizer, seq_len, device)

    sep = run_separability(
        features,
        train_frac=float(cfg.probe_train_fraction),
        seed=int(cfg.freet.seed),
        n_shuffles=int(cfg.n_shuffles),
        n_random_repeats=int(cfg.n_random_repeats),
    )

    try:
        umap_figure(features, report_dir / "umap_freet_adapter.png")
    except Exception as exc:
        logger.warning("UMAP figure failed: {}", exc)

    pid_map = ckpt_meta.get("persona_id_map") or {}
    classifier_diag = encoder_classification_accuracy(features, pid_map)
    logger.info(
        "adapter classifier diag: overall_accuracy={:.3f}",
        classifier_diag["overall_accuracy"],
    )

    # write_verdict expects a "model_cfg" with a latent_mode field — synthesise
    # one so the verdict.md prints the right header.
    ckpt_meta = {
        **ckpt_meta,
        "model_cfg": {**(ckpt_meta.get("model_cfg") or {}), "latent_mode": "supervised"},
    }
    write_verdict(report_dir, sep, ckpt_meta, classifier_diag=classifier_diag)
    logger.info(
        "Z-separability verdict = {} (macro AUROC = {:.3f}); classifier acc = {:.3f}",
        sep.overall_verdict, sep.macro_auroc, classifier_diag["overall_accuracy"],
    )

    # Also dump a thin summary.json for replay.
    (report_dir / "adapter_summary.json").write_text(
        json.dumps(
            {
                "ckpt": str(cfg.ckpt),
                "step": ckpt_meta.get("step"),
                "persona_id_map": pid_map,
                "classifier_diag": classifier_diag,
                "separation": asdict(sep),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if sep.overall_verdict != "refuted" else 2


if __name__ == "__main__":
    sys.exit(main())
