"""Generation-time Z-steering test for the adapter.

Mirrors `scripts/sample_freet_with_fixed_z.py` for the adapter (frozen
pretrained backbone). Same neutral prompts, same encoder-roundtrip
attribution metric, same report shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.freet.adapter_analysis import load_adapter_checkpoint

NEUTRAL_PROMPTS: tuple[str, ...] = (
    "User: I want to learn about something new. What's worth my time?\nAssistant:",
    "User: How do you usually approach a hard problem?\nAssistant:",
    "User: What do you think about how people learn?\nAssistant:",
    "User: Tell me a bit about your background.\nAssistant:",
    "User: What's something you'd push back on if I asked?\nAssistant:",
)


@hydra.main(
    config_path="../src/persona_rag/config",
    config_name="sample_freet_adapter",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "sampling.log", level="DEBUG")
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    if cfg.ckpt in (None, "???"):
        logger.error("ckpt= is required")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda={}", device, torch.cuda.is_available())

    adapter, ckpt_meta, tokenizer = load_adapter_checkpoint(Path(cfg.ckpt), device)
    seq_len = int(cfg.freet.seq_len)
    train_blob = ckpt_meta.get("train_cfg", {}) or {}
    if "seq_len" in train_blob:
        seq_len = int(train_blob["seq_len"])

    pid_map = ckpt_meta.get("persona_id_map") or {}
    if not pid_map:
        logger.error("checkpoint missing persona_id_map")
        return 2
    inv_pid = {v: k for k, v in pid_map.items()}

    n_samples = int(cfg.n_samples_per_persona)
    max_new_tokens = int(cfg.max_new_tokens)
    temperature = float(cfg.temperature)
    top_k = int(cfg.top_k)

    prompt_payload = []
    for p in NEUTRAL_PROMPTS:
        ids, attn = tokenizer.encode(p, seq_len)
        keep = int(attn.sum().item())
        prompt_payload.append((
            ids[:keep].unsqueeze(0).to(device),
            attn[:keep].unsqueeze(0).to(device),
            p,
        ))

    rows: list[dict] = []
    correct = 0
    total = 0
    per_correct: dict[str, int] = {pid: 0 for pid in pid_map}
    per_total: dict[str, int] = {pid: 0 for pid in pid_map}

    for target_pid, target_idx in sorted(pid_map.items(), key=lambda kv: kv[1]):
        logger.info("sampling for target persona {!r} (idx={})", target_pid, target_idx)
        for prompt_ids, prompt_mask, prompt_text in prompt_payload:
            for sample_i in range(n_samples):
                gen = adapter.generate(
                    prompt_ids, prompt_mask,
                    fixed_z_index=target_idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_full = gen[0].tolist()
                suffix_ids = gen_full[prompt_ids.shape[1]:]
                while suffix_ids and suffix_ids[-1] == tokenizer.pad_token_id:
                    suffix_ids.pop()
                suffix_text = tokenizer.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                full_text = prompt_text + suffix_text
                enc_ids, enc_attn = tokenizer.encode(full_text, seq_len)
                with torch.no_grad():
                    enc_logits = adapter.encode(
                        enc_ids.unsqueeze(0).to(device),
                        enc_attn.unsqueeze(0).to(device),
                    )
                keep = enc_attn.bool().to(device)
                feat = enc_logits.float().squeeze(0)[keep].mean(dim=0)
                pred_idx = int(feat.argmax().item())
                pred_pid = inv_pid.get(pred_idx, "<oov>")
                rows.append({
                    "target_persona": target_pid,
                    "target_idx": target_idx,
                    "prompt": prompt_text,
                    "sample_idx": sample_i,
                    "completion": suffix_text,
                    "predicted_persona": pred_pid,
                    "predicted_idx": pred_idx,
                    "correct": pred_pid == target_pid,
                })
                total += 1
                per_total[target_pid] += 1
                if pred_pid == target_pid:
                    correct += 1
                    per_correct[target_pid] += 1
        logger.info(
            "  target={} : recall = {} / {}",
            target_pid, per_correct[target_pid], per_total[target_pid],
        )

    overall = correct / max(total, 1)
    per_recall = {pid: per_correct[pid] / max(per_total[pid], 1) for pid in pid_map}
    n_targets = len(pid_map)
    chance = 1.0 / n_targets

    summary = {
        "script": "scripts/sample_freet_adapter_with_fixed_z.py",
        "checkpoint_step": ckpt_meta.get("step"),
        "ckpt": str(cfg.ckpt),
        "backbone_model_id": ckpt_meta.get("backbone_model_id"),
        "n_samples_per_persona": n_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "n_prompts": len(NEUTRAL_PROMPTS),
        "attribution_accuracy": overall,
        "per_target_recall": per_recall,
        "samples": rows,
    }
    (report_dir / "samples.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Free-Transformer Adapter — generation-time Z-steering",
        "",
        f"- Backbone: `{ckpt_meta.get('backbone_model_id')}`",
        f"- Checkpoint step: {ckpt_meta.get('step')}",
        f"- {n_samples} samples x {len(NEUTRAL_PROMPTS)} prompts x {n_targets} targets = {total} generations",
        f"- Sampling: temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}",
        "",
        f"## Attribution accuracy: **{overall:.3f}** (chance = {chance:.3f})",
        "",
        "| Target persona | Recall |",
        "|---|---|",
    ]
    for pid in sorted(pid_map):
        lines.append(f"| `{pid}` | {per_recall[pid]:.3f} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    if overall >= 0.80:
        lines.append(
            "**Z steers generation through the frozen backbone.** The "
            "residual-stream injection is sufficient on a pretrained model; "
            "the encoder + z_to_residual adapter is a viable persona-conditioner "
            "architecture."
        )
    elif overall >= chance + 0.10:
        lines.append(
            "**Partial steering.** Z affects generation but the backbone is "
            "consuming it weakly. Natural escalation: option (b) — add a "
            "LoRA on the L/2+1 K/V projections so the consumer block can "
            "learn to read R."
        )
    else:
        lines.append(
            "**Z is ignored at generation time.** The frozen backbone treats "
            "R as a perturbation it has no parameters to consume. Option (b) "
            "with a small LoRA on the modulated layer is the next experiment."
        )
    lines.append("")
    lines.append("## A few sample completions (one per target x prompt)")
    lines.append("")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["target_persona"], row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        snippet = row["completion"].strip().replace("\n", " ")[:200]
        tick = "OK" if row["correct"] else "MISS"
        lines.append(
            f"- target=`{row['target_persona']}` -> predicted=`{row['predicted_persona']}` [{tick}]\n"
            f"  prompt: _{row['prompt'].splitlines()[0]}_\n"
            f"  completion: {snippet}"
        )
    (report_dir / "sampling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "attribution accuracy = {:.3f} ({}/{}); chance = {:.3f}",
        overall, correct, total, chance,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
