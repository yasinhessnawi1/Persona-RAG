"""Adapter-side analysis: load checkpoint, run encoder + classifier diag.

Mirrors `freet/analysis.py` but for the adapter checkpoints, which
serialize only the trainable modules (encoder + persona head + z_to_residual)
and reload the backbone fresh from HF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from persona_rag.freet.adapter import FreetAdapter, FreetAdapterConfig
from persona_rag.freet.analysis import EncodedFeature
from persona_rag.freet.corpus import CorpusExample
from persona_rag.freet.tokenizer import TokenizerBundle


def load_adapter_checkpoint(
    path: Path, device: torch.device
) -> tuple[FreetAdapter, dict[str, Any], TokenizerBundle]:
    """Load an adapter checkpoint: re-instantiates the backbone + restores trainable weights."""
    raw = torch.load(path, map_location="cpu")
    backbone_id = raw.get("backbone_model_id")
    if backbone_id is None:
        raise ValueError("checkpoint missing backbone_model_id; cannot reload backbone")

    from persona_rag.freet.adapter_train import _load_backbone

    backbone, tokenizer = _load_backbone(backbone_id)
    adapter_cfg = FreetAdapterConfig(**raw["adapter_cfg"])
    adapter = FreetAdapter(backbone, adapter_cfg).to(device)

    state = raw["trainable_state_dict"]
    adapter.encoder.load_state_dict(state["encoder"])
    adapter.persona_head.load_state_dict(state["persona_head"])
    adapter.z_to_residual.load_state_dict(state["z_to_residual"])
    if device.type == "cuda":
        adapter.encoder = adapter.encoder.to(torch.float16)
        adapter.persona_head = adapter.persona_head.to(torch.float16)
        adapter.z_to_residual = adapter.z_to_residual.to(torch.float16)
    adapter.eval()
    return adapter, raw, tokenizer


def encode_dataset(
    adapter: FreetAdapter,
    examples: list[CorpusExample],
    tokenizer: TokenizerBundle,
    seq_len: int,
    device: torch.device,
) -> list[EncodedFeature]:
    """Per-sequence encoder feature: mean-pool encoder logits over valid tokens."""
    out: list[EncodedFeature] = []
    for ex in examples:
        ids, attn = tokenizer.encode(ex.text, seq_len)
        ids_b = ids.unsqueeze(0).to(device)
        mask_b = attn.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = adapter.encode(ids_b, mask_b)  # (1, T, n_personas)
        logits = logits.float().squeeze(0)
        keep = mask_b.squeeze(0).bool()
        feat = (
            logits[keep].mean(dim=0).cpu().numpy()
            if keep.sum() > 0
            else logits.mean(dim=0).cpu().numpy()
        )
        out.append(EncodedFeature(ex.persona_id, ex.source, feat))
    return out


def encoder_classification_accuracy(
    features: list[EncodedFeature], pid_map: dict[str, int]
) -> dict[str, Any]:
    inv = {v: k for k, v in pid_map.items()}
    correct = 0
    per_correct: dict[str, int] = {pid: 0 for pid in pid_map}
    per_total: dict[str, int] = {pid: 0 for pid in pid_map}
    for f in features:
        pred = inv.get(int(np.argmax(f.feature)), "<oov>")
        per_total[f.persona_id] = per_total.get(f.persona_id, 0) + 1
        if pred == f.persona_id:
            correct += 1
            per_correct[f.persona_id] = per_correct.get(f.persona_id, 0) + 1
    overall = correct / max(len(features), 1)
    per_recall = {pid: per_correct.get(pid, 0) / t if t > 0 else 0.0 for pid, t in per_total.items()}
    return {"overall_accuracy": overall, "per_persona_recall": per_recall}
