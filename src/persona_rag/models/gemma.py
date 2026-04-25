"""Gemma 2 9B-Instruct backend.

Two Gemma-2-specific points the shared :class:`HFBackend` cannot express:

1. **No ``system`` role.** Gemma 2's chat template raises
   ``TemplateError: System role not supported``. We fold any ``system`` messages
   into the first user turn as a prefix, separated by a blank line. The public
   surface of this behaviour is :meth:`format_persona_prompt`, overridden here
   to inline-prepend the system text.

2. **``attn_implementation="eager"`` is mandatory.** Gemma 2 uses attention-
   logit softcap, which SDPA/flash-attention don't implement. Without eager the
   softcap is silently skipped and the model deviates from Google's reference.
   We mirror eager on the Llama backend too so the attention-kernel path is
   identical across backbones.

The fp16 compute dtype required on V100 (CC 7.0, no bf16) is handled in
:class:`persona_rag.models._hf_base.HFBackend` via the ``compute_dtype`` config
knob.
"""

from __future__ import annotations

from persona_rag.models._hf_base import HFBackend, HFBackendConfig
from persona_rag.models.base import ChatMessage


class GemmaBackend(HFBackend):
    """Gemma 2 9B-it wrapper. Defaults match our V100 + fp16 deployment."""

    @classmethod
    def default_config(cls, **overrides: object) -> HFBackendConfig:
        base: dict[str, object] = {
            "model_id": "google/gemma-2-9b-it",
            "name": "gemma2-9b-it",
            "compute_dtype": "float16",
            "attn_implementation": "eager",
        }
        base.update(overrides)
        return HFBackendConfig(**base)  # type: ignore[arg-type]

    def _render_chat(self, messages: list[ChatMessage]) -> str:
        """Render with Gemma 2's chat template after folding any ``system`` turns."""
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

        If ``history`` is provided, the system prefix lands on the earliest user
        turn in the full conversation — callers expect system-level guidance to
        be seen from the start, not just on the final turn.
        """
        messages: list[ChatMessage] = []
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_text))
        return self._render_chat(messages)


def _fold_system_into_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return a new list where every ``system`` turn is prepended to the next user turn.

    - All consecutive ``system`` messages at the start are concatenated.
    - Their text is prefixed (separated by a blank line) onto the first ``user`` turn.
    - A trailing ``system`` with no following user turn is turned into a user turn itself
      (rare, but keeps the template legal).
    - Any ``system`` messages that appear *after* the first user turn are treated the
      same way: folded into the next user turn.
    """
    if not any(m.role == "system" for m in messages):
        return list(messages)

    out: list[ChatMessage] = []
    buffered_system: list[str] = []
    for m in messages:
        if m.role == "system":
            buffered_system.append(m.content)
            continue
        if m.role == "user" and buffered_system:
            prefix = "\n\n".join(buffered_system).strip()
            out.append(ChatMessage(role="user", content=f"{prefix}\n\n{m.content}"))
            buffered_system = []
        else:
            out.append(m)

    if buffered_system:
        out.append(ChatMessage(role="user", content="\n\n".join(buffered_system).strip()))
    return out
