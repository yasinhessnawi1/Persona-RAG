"""Qwen2.5-7B-Instruct backend.

`Qwen/Qwen2.5-7B-Instruct` is a strong instruction-following 7B model from
a different family than Llama / Mistral / Gemma — useful as a cross-family
judge to break self-bias in evaluation panels. Standard chat template with
native ``system`` role support; no special handling beyond the shared
:class:`HFBackend` plumbing.

V100 + fp16 + 4-bit NF4 with eager attention, matching the project's
standard quantization recipe.
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
            "attn_implementation": "eager",
        }
        base.update(overrides)
        return HFBackendConfig(**base)  # type: ignore[arg-type]
