"""Prometheus-2-7B-v2.0 backend (Mistral-Instruct base, evaluator-tuned).

`prometheus-eval/prometheus-7b-v2.0` is a Mistral-Instruct fine-tune
specialised for LLM-as-judge evaluation. The model card emphasises a
specific direct-assessment / pairwise-ranking rubric format with
``[INST]/[/INST]`` markers, but those markers are produced by Mistral-
Instruct's standard chat template — calling
``tokenizer.apply_chat_template`` (the default :class:`HFBackend`
behaviour) is correct here.

Mistral-Instruct does not natively accept a separate ``system`` role. The
chat template raises if a ``system`` message is supplied. This backend
folds any ``system`` content into the first user turn (mirroring the
Gemma 2 fold pattern), so callers stay role-agnostic.

V100 + fp16 + 4-bit NF4 with eager attention, matching the project's
standard quantization recipe.
"""

from __future__ import annotations

from persona_rag.models._hf_base import HFBackend, HFBackendConfig
from persona_rag.models.base import ChatMessage
from persona_rag.models.gemma import _fold_system_into_user


class PrometheusBackend(HFBackend):
    """Prometheus-2-7B v2.0 wrapper. Defaults match the V100 + fp16 deployment."""

    @classmethod
    def default_config(cls, **overrides: object) -> HFBackendConfig:
        base: dict[str, object] = {
            "model_id": "prometheus-eval/prometheus-7b-v2.0",
            "name": "prometheus-2-7b",
            "compute_dtype": "float16",
            "attn_implementation": "eager",
        }
        base.update(overrides)
        return HFBackendConfig(**base)  # type: ignore[arg-type]

    def _render_chat(self, messages: list[ChatMessage]) -> str:
        """Render with Mistral-Instruct's chat template after folding ``system`` turns.

        Mistral-Instruct's template does not accept a ``system`` role; we
        fold any ``system`` content into the first user turn (same pattern
        as the Gemma 2 backend).
        """
        folded = _fold_system_into_user(messages)
        return self._tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in folded],
            tokenize=False,
            add_generation_prompt=True,
        )

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Inline-prepend ``system_text`` to the first user turn.

        Mirrors the Gemma 2 backend — Mistral-Instruct has no ``system``
        role, so the system content folds into user.
        """
        messages: list[ChatMessage] = []
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_text))
        return self._render_chat(messages)
