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
    supervised_freet_loss,
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
    """κ in bits per token (paper sweeps {0.25, 0.5, 1.0, 8.0}). Multiplied by ln(2) internally.
    Used only in unsupervised mode.
    """
    latent_mode: str = "unsupervised"
    """One of: 'unsupervised' | 'supervised'."""
    n_personas: int = 3
    """Used only in supervised mode."""
    persona_loss_weight: float = 1.0
    """Scalar on the persona-classifier CE term in supervised mode."""
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


def build_persona_id_map(*jsonl_paths: Path) -> dict[str, int]:
    """Deterministic persona_id → integer mapping (sorted by string id)."""
    seen: set[str] = set()
    for p in jsonl_paths:
        for ex in iter_jsonl(p):
            seen.add(ex.persona_id)
    return {pid: i for i, pid in enumerate(sorted(seen))}


def _persona_label_tensor(
    persona_ids: list[str], pid_map: dict[str, int], device: torch.device
) -> torch.Tensor:
    return torch.tensor([pid_map[p] for p in persona_ids], dtype=torch.long, device=device)


def evaluate(
    model: FreeTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    kappa_nats: float,
    max_batches: int = 32,
    use_amp: bool = False,
    *,
    latent_mode: str = "unsupervised",
    pid_map: dict[str, int] | None = None,
    persona_loss_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    if latent_mode == "unsupervised":
        keys = ("loss", "ce", "kl_mean", "kl_excess")
    else:
        keys = ("loss", "ce", "persona_ce", "persona_acc")
    totals: dict[str, float] = {k: 0.0 for k in keys}
    totals["n"] = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attn"].to(device)
            targets = _next_token_targets(input_ids, pad_id)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                out = model(input_ids, attn_mask=attn)
                if latent_mode == "unsupervised":
                    assert out.kl_per_token is not None
                    _, metrics = free_transformer_loss(
                        out.logits, targets, out.kl_per_token,
                        pad_token_id=pad_id, free_bits_kappa=kappa_nats,
                    )
                else:
                    assert out.persona_logits is not None and pid_map is not None
                    labels = _persona_label_tensor(batch["persona_id"], pid_map, device)
                    _, metrics = supervised_freet_loss(
                        out.logits, targets, out.persona_logits, labels,
                        pad_token_id=pad_id, persona_loss_weight=persona_loss_weight,
                    )
            for k in keys:
                totals[k] += metrics[k]
            totals["n"] += 1.0
    n = max(totals["n"], 1.0)
    return {k: totals[k] / n for k in keys}


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

    pid_map = build_persona_id_map(Path(cfg.train_jsonl), Path(cfg.test_jsonl))
    n_personas = max(len(pid_map), cfg.n_personas)
    if cfg.latent_mode == "supervised":
        logger.info("supervised-Z mode: persona id map = {}", pid_map)

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
        latent_mode=cfg.latent_mode,
        n_personas=n_personas,
    )
    # Master weights stay fp32; AMP autocasts matmuls/conv to fp16 inside the
    # forward and a GradScaler keeps grads in fp16 range. Pure-fp16 training of
    # a 256k-vocab softmax from scratch on V100 NaNs out within a few steps —
    # AMP is the standard recipe and what V100 fp16 actually wants.
    model = FreeTransformer(model_cfg).to(device)
    use_amp = bool(cfg.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(
        "model built: {} params, inject_after_layer={}, amp={}",
        model.num_parameters(), model_cfg.inject_after_layer, use_amp,
    )

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
                "persona_id_map": pid_map,
            },
            path,
        )
        return path

    step = 0
    if cfg.latent_mode == "unsupervised":
        log_keys = ("loss", "ce", "kl_mean", "kl_excess")
    else:
        log_keys = ("loss", "ce", "persona_ce", "persona_acc")
    micro: dict[str, float] = {k: 0.0 for k in log_keys}
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
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                out = model(input_ids, attn_mask=attn)
                if cfg.latent_mode == "unsupervised":
                    assert out.kl_per_token is not None
                    loss, metrics = free_transformer_loss(
                        out.logits, targets, out.kl_per_token,
                        pad_token_id=tokenizer.pad_token_id,
                        free_bits_kappa=cfg.kappa_nats,
                    )
                else:
                    assert out.persona_logits is not None
                    labels = _persona_label_tensor(batch["persona_id"], pid_map, device)
                    loss, metrics = supervised_freet_loss(
                        out.logits, targets, out.persona_logits, labels,
                        pad_token_id=tokenizer.pad_token_id,
                        persona_loss_weight=cfg.persona_loss_weight,
                    )
            scaler.scale(loss / cfg.grad_accum).backward()
            for k in log_keys:
                micro[k] += metrics[k] / (cfg.grad_accum * cfg.log_every)

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            row = {"step": step, "lr": lr, "elapsed_s": elapsed, **micro}
            metrics_log.append(row)
            logger.info(
                "step {:>5d}  lr={:.2e}  " + "  ".join(f"{k}={{:.4f}}" for k in log_keys) + "  ({:.1f}s)",
                step, lr, *(micro[k] for k in log_keys), elapsed,
            )
            for k in log_keys:
                micro[k] = 0.0

        if step % cfg.eval_every == 0:
            eval_metrics = evaluate(
                model, test_loader, device, tokenizer.pad_token_id, cfg.kappa_nats,
                use_amp=use_amp,
                latent_mode=cfg.latent_mode,
                pid_map=pid_map,
                persona_loss_weight=cfg.persona_loss_weight,
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
