"""Training loop for the off-spec Free-Transformer persona experiment.

Single-GPU, single-node, fp16 (V100). One config = one κ. The script writes a
checkpoint plus a metrics JSON; the analysis script consumes both.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from persona_rag.freet.corpus import iter_jsonl
from persona_rag.freet.model import (
    FreeTransformer,
    FreeTransformerConfig,
    free_transformer_loss,
)
from persona_rag.freet.tokenizer import TokenizerBundle, load_tokenizer

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class PersonaCorpusDataset(Dataset):
    """JSONL-backed dataset of pre-tokenised persona-labelled sequences."""

    def __init__(self, jsonl_path: Path, tokenizer: TokenizerBundle, seq_len: int) -> None:
        self.examples = list(iter_jsonl(jsonl_path))
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        ids, attn = self.tokenizer.encode(ex.text, self.seq_len)
        return {"input_ids": ids, "attn": attn, "persona_id": ex.persona_id, "source": ex.source}


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attn": torch.stack([b["attn"] for b in batch]),
        "persona_id": [b["persona_id"] for b in batch],
        "source": [b["source"] for b in batch],
    }


# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    out_dir: Path
    train_jsonl: Path
    test_jsonl: Path
    tokenizer_id: str = "google/gemma-2-9b-it"
    seq_len: int = 1024
    # Model.
    dim: int = 384
    n_layers: int = 6
    n_q_heads: int = 6
    n_kv_heads: int = 2
    mlp_ratio: int = 4
    latent_bits: int = 8
    inject_after_layer: int | None = None
    # Optim.
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    max_steps: int = 8_000
    grad_clip: float = 1.0
    free_bits_kappa_bits: float = 0.5
    """κ in bits per token (paper sweeps {0.25, 0.5, 1.0, 8.0}). Multiplied by ln(2) internally."""
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 2_000
    seed: int = 42
    fp16: bool = True

    @property
    def kappa_nats(self) -> float:
        return self.free_bits_kappa_bits * math.log(2.0)


def _cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _next_token_targets(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Right-shift inputs by one to produce next-token targets; pad the last position."""
    targets = input_ids.clone()
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = pad_id
    return targets


def evaluate(
    model: FreeTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    kappa_nats: float,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "kl_mean": 0.0, "kl_excess": 0.0, "n": 0.0}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attn"].to(device)
            targets = _next_token_targets(input_ids, pad_id)
            out = model(input_ids, attn_mask=attn)
            assert out.kl_per_token is not None
            _, metrics = free_transformer_loss(
                out.logits, targets, out.kl_per_token,
                pad_token_id=pad_id, free_bits_kappa=kappa_nats,
            )
            for k in ("loss", "ce", "kl_mean", "kl_excess"):
                totals[k] += metrics[k]
            totals["n"] += 1.0
    n = max(totals["n"], 1.0)
    return {k: totals[k] / n for k in ("loss", "ce", "kl_mean", "kl_excess")}


def train(cfg: TrainConfig) -> Path:
    """Train one Free Transformer; return the checkpoint path."""
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "train.log", level="DEBUG")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda_available={}", device, torch.cuda.is_available())

    tokenizer = load_tokenizer(cfg.tokenizer_id)
    logger.info("tokenizer loaded: vocab_size={} pad={} eos={}",
                tokenizer.vocab_size, tokenizer.pad_token_id, tokenizer.eos_token_id)

    model_cfg = FreeTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_q_heads=cfg.n_q_heads,
        n_kv_heads=cfg.n_kv_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_seq_len=cfg.seq_len,
        latent_bits=cfg.latent_bits,
        inject_after_layer=cfg.inject_after_layer,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = FreeTransformer(model_cfg).to(device)
    if cfg.fp16 and device.type == "cuda":
        model = model.to(torch.float16)
    logger.info("model built: {} params, inject_after_layer={}",
                model.num_parameters(), model_cfg.inject_after_layer)

    train_ds = PersonaCorpusDataset(Path(cfg.train_jsonl), tokenizer, cfg.seq_len)
    test_ds = PersonaCorpusDataset(Path(cfg.test_jsonl), tokenizer, cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    logger.info("dataset sizes: train={} test={}", len(train_ds), len(test_ds))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    metrics_log: list[dict] = []

    def _save_ckpt(tag: str, step: int) -> Path:
        path = out_dir / f"{tag}.pt"
        torch.save(
            {
                "step": step,
                "model_cfg": asdict(model_cfg),
                "train_cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "state_dict": model.state_dict(),
            },
            path,
        )
        return path

    step = 0
    micro_loss = 0.0
    micro_ce = 0.0
    micro_kl = 0.0
    micro_excess = 0.0
    t0 = time.time()

    train_iter = _infinite(train_loader)
    while step < cfg.max_steps:
        model.train()
        # Adjust LR.
        lr = _cosine_lr(step, cfg.lr, cfg.warmup_steps, cfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attn"].to(device)
            targets = _next_token_targets(input_ids, tokenizer.pad_token_id)
            out = model(input_ids, attn_mask=attn)
            assert out.kl_per_token is not None
            loss, metrics = free_transformer_loss(
                out.logits, targets, out.kl_per_token,
                pad_token_id=tokenizer.pad_token_id,
                free_bits_kappa=cfg.kappa_nats,
            )
            (loss / cfg.grad_accum).backward()
            micro_loss += metrics["loss"] / cfg.grad_accum
            micro_ce += metrics["ce"] / cfg.grad_accum
            micro_kl += metrics["kl_mean"] / cfg.grad_accum
            micro_excess += metrics["kl_excess"] / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            row = {
                "step": step,
                "lr": lr,
                "loss": micro_loss,
                "ce": micro_ce,
                "kl_mean": micro_kl,
                "kl_excess": micro_excess,
                "elapsed_s": elapsed,
            }
            metrics_log.append(row)
            logger.info(
                "step {:>5d}  lr={:.2e}  loss={:.4f}  ce={:.4f}  kl_mean={:.4f}  kl_excess={:.4f}  ({:.1f}s elapsed)",
                step, lr, micro_loss, micro_ce, micro_kl, micro_excess, elapsed,
            )
            micro_loss = micro_ce = micro_kl = micro_excess = 0.0

        if step % cfg.eval_every == 0:
            eval_metrics = evaluate(
                model, test_loader, device, tokenizer.pad_token_id, cfg.kappa_nats,
            )
            logger.info("[eval @ step {}] {}", step, eval_metrics)
            metrics_log.append({"step": step, "phase": "eval", **eval_metrics})

        if step % cfg.save_every == 0:
            ck = _save_ckpt(f"ckpt_step{step:05d}", step)
            logger.info("checkpoint saved: {}", ck)

    # Final.
    final_path = _save_ckpt("final", step)
    logger.info("final checkpoint saved: {}", final_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n", encoding="utf-8")
    return final_path


def _infinite(loader: DataLoader) -> Iterable[dict]:
    while True:
        yield from loader
