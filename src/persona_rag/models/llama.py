"""Llama 3.1 8B-Instruct backend.

Llama-3.1-specific points:

1. **Two valid stop tokens.** The canonical end-of-turn is ``<|eot_id|>`` (not
   the base ``<|end_of_text|>``). We pass both to ``generate`` so generation
   stops at either — matching Meta's reference.

2. **No pad token.** Handled in :class:`HFBackend` by aliasing ``pad_token`` to
   ``eos_token`` at load.

3. **``attn_implementation="eager"``** is applied here too, even though Llama
   doesn't strictly need it, so that Gemma 2 and Llama 3.1 share an attention-
   kernel path. Removes "kernel differences" as a confound in downstream
   evaluation.

Llama 3.1 supports the native ``system`` role in its chat template, so we use
:class:`HFBackend`'s default chat renderer unchanged.
"""

from __future__ import annotations

from persona_rag.models._hf_base import HFBackend, HFBackendConfig

_LLAMA_EOT_TOKEN = "<|eot_id|>"


class LlamaBackend(HFBackend):
    """Llama-3.1-8B-Instruct wrapper. Defaults match our V100 + fp16 deployment."""

    @classmethod
    def default_config(cls, **overrides: object) -> HFBackendConfig:
        base = {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "name": "llama3.1-8b-instruct",
            "compute_dtype": "float16",
            "attn_implementation": "eager",
        }
        base.update(overrides)
        return HFBackendConfig(**base)  # type: ignore[arg-type]

    def _stop_token_ids(self) -> list[int]:
        """Include both the base EOS and ``<|eot_id|>`` as valid stops."""
        ids: list[int] = []
        eos = self._tokenizer.eos_token_id
        if eos is not None:
            ids.append(eos)
        eot = self._tokenizer.convert_tokens_to_ids(_LLAMA_EOT_TOKEN)
        # ``convert_tokens_to_ids`` returns ``unk_token_id`` (or the unk int) when the
        # token is missing. Filter defensively.
        if eot is not None and eot != self._tokenizer.unk_token_id and eot not in ids:
            ids.append(eot)
        return ids
