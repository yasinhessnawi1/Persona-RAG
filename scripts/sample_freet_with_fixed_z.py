"""Generation-time test: does the decoder use Z?

Loads a supervised Free-Transformer checkpoint, samples continuations from a
shared neutral prompt with Z **hard-clamped** to each persona index in turn,
and feeds the generations back through the trained encoder to compute
attribution accuracy:

  * For each (target_persona, sample) pair, run the encoder on the generation,
    take argmax → predicted persona.
  * Attribution accuracy = fraction of samples whose predicted persona equals
    the target_persona we conditioned on.

Reading:

  * accuracy ≈ 1.0  → the decoder consumes Z; Z steers generation.
  * accuracy ≈ 1/N  → the decoder ignores Z; generations are constant w.r.t. Z.
  * accuracy in between → partial steering; informative but ambiguous.

Usage:

    uv run python scripts/sample_freet_with_fixed_z.py \\
        ckpt=results/freet/<run_id>_supervised_kappa0.5/final.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.freet.analysis import load_checkpoint
from persona_rag.freet.tokenizer import load_tokenizer

# Neutral prompts — phrased so any of the three personas could plausibly answer
# without lexical leak of which persona is speaking. Each prompt is fed to the
# decoder with Z clamped to each of {climate_scientist, cs_tutor, historian}.
NEUTRAL_PROMPTS: tuple[str, ...] = (
    "User: I want to learn about something new. What's worth my time?\nAssistant:",
    "User: How do you usually approach a hard problem?\nAssistant:",
    "User: What do you think about how people learn?\nAssistant:",
    "User: Tell me a bit about your background.\nAssistant:",
    "User: What's something you'd push back on if I asked?\nAssistant:",
)


@hydra.main(config_path="../src/persona_rag/config", config_name="sample_freet", version_base=None)
def main(cfg: DictConfig) -> int:
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "sampling.log", level="DEBUG")
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    if cfg.ckpt in (None, "???"):
        logger.error("ckpt is required (e.g. ckpt=results/freet/.../final.pt)")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device={} cuda={}", device, torch.cuda.is_available())

    model, ckpt_meta = load_checkpoint(Path(cfg.ckpt), device)
    train_cfg_blob = ckpt_meta.get("train_cfg", {}) or {}
    seq_len = int(cfg.freet.seq_len)
    if "seq_len" in train_cfg_blob:
        seq_len = int(train_cfg_blob["seq_len"])

    pid_map = ckpt_meta.get("persona_id_map")
    latent_mode = (ckpt_meta.get("model_cfg") or {}).get("latent_mode", "unsupervised")
    if latent_mode != "supervised" or not pid_map:
        logger.error(
            "this script only makes sense for supervised checkpoints with a "
            "persona_id_map; got latent_mode={} pid_map={}", latent_mode, bool(pid_map),
        )
        return 2
    inv_pid = {v: k for k, v in pid_map.items()}
    logger.info("persona id map: {}", pid_map)

    tokenizer = load_tokenizer(str(train_cfg_blob.get("tokenizer_id", cfg.freet.tokenizer_id)))

    n_samples = int(cfg.get("n_samples_per_persona", 5))
    max_new_tokens = int(cfg.get("max_new_tokens", 80))
    temperature = float(cfg.get("temperature", 0.8))
    top_k = int(cfg.get("top_k", 40))

    # Prepare prompts as a (B=1, T) tensor each.
    prompt_token_lists = []
    for p in NEUTRAL_PROMPTS:
        ids, attn = tokenizer.encode(p, seq_len)
        # Trim trailing padding so generate() appends after the actual prompt.
        keep = int(attn.sum().item())
        prompt_token_lists.append((ids[:keep].unsqueeze(0).to(device),
                                   attn[:keep].unsqueeze(0).to(device),
                                   p))

    rows: list[dict] = []
    correct = 0
    total = 0
    per_target_correct: dict[str, int] = {pid: 0 for pid in pid_map}
    per_target_total: dict[str, int] = {pid: 0 for pid in pid_map}

    for target_pid, target_idx in sorted(pid_map.items(), key=lambda kv: kv[1]):
        logger.info("sampling for target persona {!r} (idx={})", target_pid, target_idx)
        for prompt_ids, prompt_mask, prompt_text in prompt_token_lists:
            for sample_i in range(n_samples):
                gen = model.generate(
                    prompt_ids, prompt_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_token_id,
                    fixed_z_index=target_idx,
                )
                # Decode generated suffix (after the original prompt).
                gen_full = gen[0].tolist()
                suffix_ids = gen_full[prompt_ids.shape[1]:]
                # Strip pad tokens from the tail.
                while suffix_ids and suffix_ids[-1] == tokenizer.pad_token_id:
                    suffix_ids.pop()
                suffix_text = tokenizer.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                # Re-encode the full generated sequence and run the encoder for
                # attribution. Use only what fits in seq_len; pad-aware encode
                # already in TokenizerBundle.
                full_text = prompt_text + suffix_text
                enc_ids, enc_attn = tokenizer.encode(full_text, seq_len)
                with torch.no_grad():
                    enc_logits = model.encode(
                        enc_ids.unsqueeze(0).to(device),
                        attn_mask=enc_attn.unsqueeze(0).to(device),
                    )
                # Mean-pool over valid tokens, argmax → predicted persona.
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
                per_target_total[target_pid] += 1
                if pred_pid == target_pid:
                    correct += 1
                    per_target_correct[target_pid] += 1
        logger.info(
            "  target={} : recall = {} / {}",
            target_pid, per_target_correct[target_pid], per_target_total[target_pid],
        )

    overall = correct / max(total, 1)
    per_recall = {
        pid: per_target_correct[pid] / max(per_target_total[pid], 1) for pid in pid_map
    }

    summary = {
        "script": "scripts/sample_freet_with_fixed_z.py",
        "checkpoint_step": ckpt_meta.get("step"),
        "ckpt": str(cfg.ckpt),
        "latent_mode": latent_mode,
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

    n_targets = len(pid_map)
    chance = 1.0 / n_targets
    lines = [
        "# Free-Transformer Z-Steering — generation-time attribution",
        "",
        f"- Checkpoint step: {ckpt_meta.get('step')}",
        f"- {n_samples} samples x {len(NEUTRAL_PROMPTS)} prompts x {n_targets} target personas "
        f"= {total} generations",
        f"- Sampling: temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}",
        "",
        f"## Attribution accuracy: **{overall:.3f}** (chance = {chance:.3f})",
        "",
        "Per-target recall — fraction of generations conditioned on Z=`target` "
        "that the encoder argmaxes back to `target`:",
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
            "**Z steers generation.** The decoder consumes Z; conditioning on a "
            "different persona index produces text the encoder reliably "
            "classifies as that persona. The supervised-Z plumbing is doing real "
            "work, not just being consumed by the encoder and discarded by the "
            "decoder."
        )
    elif overall >= chance + 0.10:
        lines.append(
            "**Partial steering.** Z affects generation but the effect is weak "
            "or unreliable. Possible reasons: (a) the decoder uses Z mostly for "
            "low-frequency stylistic cues that the encoder doesn't pick up "
            "back, (b) the decoder ignores Z and tracks the prompt, with the "
            "encoder reading the prompt back to the target persona by lexical "
            "leak."
        )
    else:
        lines.append(
            "**Z is ignored at generation time.** Conditioning on different "
            "persona indices produces text the encoder cannot reliably "
            "attribute back. Z is a write-only channel: the encoder uses it "
            "to satisfy the supervised classification loss, but the decoder "
            "does not consume it to shape generation."
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
        lines.append(
            f"- target=`{row['target_persona']}` → predicted=`{row['predicted_persona']}`"
            f" {'✓' if row['correct'] else '✗'}\n"
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
