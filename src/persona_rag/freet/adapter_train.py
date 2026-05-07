"""Training loop for the Free-Transformer adapter on a frozen pretrained backbone.

Loads Gemma-2-9B (or any HF causal LM) in 4-bit, freezes it, attaches a
trainable encoder + persona head + ``z_to_residual`` via forward hooks,
trains on the persona-labelled corpus with the same AMP recipe as the
from-scratch run.

Only the new modules receive gradients; the backbone is in 4-bit and stays
frozen throughout training. Optimizer steps are correspondingly cheap.
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
from torch.utils.data import DataLoader

from persona_rag.freet.adapter import FreetAdapter, FreetAdapterConfig
from persona_rag.freet.model import supervised_freet_loss
from persona_rag.freet.tokenizer import TokenizerBundle, load_tokenizer
from persona_rag.freet.train import (
    PersonaCorpusDataset,
    _next_token_targets,
    _persona_label_tensor,
    build_persona_id_map,
    collate,
)


@dataclass
class AdapterTrainConfig:
    out_dir: Path
    train_jsonl: Path
    test_jsonl: Path
    backbone_model_id: str = "google/gemma-2-9b-it"
    seq_len: int = 1024
    n_personas: int = 3
    encoder_mlp_ratio: int = 4
    inject_after_layer: int | None = None
    persona_loss_weight: float = 1.0
    # Optim.
    batch_size: int = 2
    grad_accum: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 4_000
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 500
    save_every: int = 1_000
    seed: int = 42
    fp16: bool = True


def _load_backbone(model_id: str) -> tuple[torch.nn.Module, TokenizerBundle]:
    """Load a frozen 4-bit causal LM. Imports are scoped so Darwin can import this module."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    logger.info("loading frozen backbone: {}", model_id)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        attn_implementation="eager",  # Gemma-2 requires eager
        torch_dtype=torch.float16,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_tokenizer(model_id)
    return model, tok


def _cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def evaluate(
    adapter: FreetAdapter,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    pid_map: dict[str, int],
    persona_loss_weight: float,
    max_batches: int = 16,
) -> dict[str, float]:
    adapter.eval()
    keys = ("loss", "ce", "persona_ce", "persona_acc")
    totals = {k: 0.0 for k in keys}
    totals["n"] = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attn"].to(device)
            targets = _next_token_targets(input_ids, pad_id)
            out = adapter(input_ids, attn)
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


def train(cfg: AdapterTrainConfig) -> Path:
    """Train the adapter; return final checkpoint path."""
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "train.log", level="DEBUG")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda={}", device, torch.cuda.is_available())

    pid_map = build_persona_id_map(Path(cfg.train_jsonl), Path(cfg.test_jsonl))
    n_personas = max(len(pid_map), cfg.n_personas)
    logger.info("persona id map: {}", pid_map)

    backbone, tokenizer = _load_backbone(cfg.backbone_model_id)
    logger.info(
        "backbone loaded: vocab_size={} pad={} eos={}",
        tokenizer.vocab_size, tokenizer.pad_token_id, tokenizer.eos_token_id,
    )

    adapter_cfg = FreetAdapterConfig(
        n_personas=n_personas,
        encoder_mlp_ratio=cfg.encoder_mlp_ratio,
        inject_after_layer=cfg.inject_after_layer,
    )
    adapter = FreetAdapter(backbone, adapter_cfg).to(device)
    if cfg.fp16:
        # Cast only the new modules to fp16; backbone is already 4-bit.
        adapter.encoder = adapter.encoder.to(torch.float16)
        adapter.persona_head = adapter.persona_head.to(torch.float16)
        adapter.z_to_residual = adapter.z_to_residual.to(torch.float16)

    n_train = adapter.num_trainable_parameters()
    logger.info(
        "adapter built: {} trainable params; inject_after_layer={}; n_personas={}",
        n_train, adapter.inject_after_layer, n_personas,
    )

    train_ds = PersonaCorpusDataset(Path(cfg.train_jsonl), tokenizer, cfg.seq_len)
    test_ds = PersonaCorpusDataset(Path(cfg.test_jsonl), tokenizer, cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    logger.info("dataset sizes: train={} test={}", len(train_ds), len(test_ds))

    opt = torch.optim.AdamW(
        list(adapter.trainable_parameters()),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    use_amp = bool(cfg.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_log: list[dict] = []

    def _save_ckpt(tag: str, step: int) -> Path:
        path = out_dir / f"{tag}.pt"
        # Save ONLY the trainable modules + the persona id map. Backbone is
        # not serialised — it can be re-loaded from HF by id at validate time.
        state = {
            "encoder": adapter.encoder.state_dict(),
            "persona_head": adapter.persona_head.state_dict(),
            "z_to_residual": adapter.z_to_residual.state_dict(),
        }
        torch.save(
            {
                "step": step,
                "adapter_cfg": asdict(adapter_cfg),
                "train_cfg": {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
                },
                "trainable_state_dict": state,
                "persona_id_map": pid_map,
                "backbone_model_id": cfg.backbone_model_id,
                "inject_after_layer": adapter.inject_after_layer,
            },
            path,
        )
        return path

    step = 0
    log_keys = ("loss", "ce", "persona_ce", "persona_acc")
    micro: dict[str, float] = {k: 0.0 for k in log_keys}
    t0 = time.time()

    train_iter = _infinite(train_loader)
    while step < cfg.max_steps:
        adapter.train()
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
                out = adapter(input_ids, attn)
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
        torch.nn.utils.clip_grad_norm_(list(adapter.trainable_parameters()), cfg.grad_clip)
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
                adapter, test_loader, device, tokenizer.pad_token_id,
                pid_map, cfg.persona_loss_weight,
            )
            logger.info("[eval @ step {}] {}", step, eval_metrics)
            metrics_log.append({"step": step, "phase": "eval", **eval_metrics})

        if step % cfg.save_every == 0:
            ck = _save_ckpt(f"ckpt_step{step:05d}", step)
            logger.info("checkpoint saved: {}", ck)

    final_path = _save_ckpt("final", step)
    logger.info("final checkpoint saved: {}", final_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n", encoding="utf-8")
    return final_path


def _infinite(loader: DataLoader) -> Iterable[dict]:
    while True:
        yield from loader
