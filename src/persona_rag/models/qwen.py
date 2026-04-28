"""Qwen2.5-7B-Instruct backend.

`Qwen/Qwen2.5-7B-Instruct` is a strong instruction-following 7B model from
a different family than Llama / Mistral / Gemma — useful as a cross-family
judge to break self-bias in evaluation panels. Standard chat template with
native ``system`` role support; no special handling beyond the shared
:class:`HFBackend` plumbing.

V100 + fp16 + 4-bit NF4 with **SDPA attention**, deviating from the
project's eager-attention default. Eager + Qwen2.5 + fp16 + 4-bit on V100
produces NaN logits in the load-time warm-up guard; Qwen2.5 was trained
in bfloat16 (V100-unsupported) and exhibits attention-numerics overflow
under eager + fp16 downcast. SDPA's fused kernel is numerically more
stable on this combination. The project's eager-attention rule covers the
responder backbones (Gemma2 softcap + Llama kernel symmetry) — Qwen is a
judge, so the symmetry rationale does not apply here.
"""

from __future__ import annotations

from persona_rag.models._hf_base import HFBackend, HFBackendConfig


class QwenBackend(HFBackend):
    """Qwen2.5-7B-Instruct wrapper. Defaults match the V100 + fp16 deployment."""

    @classmethod
    def default_config(cls, **overrides: object) -> HFBackendConfig:
        base: dict[str, object] = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "qwen2.5-7b-instruct",
            "compute_dtype": "float16",
            "attn_implementation": "sdpa",
        }
        base.update(overrides)
        return HFBackendConfig(**base)  # type: ignore[arg-type]
