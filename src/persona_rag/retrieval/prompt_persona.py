"""Prompt-persona baseline: structured persona system block + dialogue few-shots.

Hybrid knowledge retrieval feeds the user block; the persona lives in the
system block. The persona block goes through the backend's
``format_persona_prompt`` so Gemma 2 (no native system role) and Llama 3.1
(native system) compose identically at the call site.

Authoring contract: each persona ships a `personas/examples/<persona_id>.yaml`
with hand-authored ``FewShotExchange``s; at least one should have
``is_constraint_case=True`` to demonstrate the persona's no-go zones.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Response, Turn
from persona_rag.retrieval.prompt_templates import (
    FewShotBundle,
    estimate_token_count,
    render_b2_persona_block,
    render_b2_simple_system,
    render_b2_user_block,
    trim_chunks_to_token_budget,
)
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_store import KnowledgeStore


@dataclass
class PromptPersonaRAG:
    """Prompt-persona pipeline.

    `b2_variant` flips between the structured persona block (default, used in
    headline results) and the one-liner persona prefix (audit-only, used to
    validate that the structured variant is measurably stronger). Both
    variants use the same hybrid knowledge retrieval; only the system block
    differs.
    """

    backend: LLMBackend
    knowledge_store: KnowledgeStore
    few_shots: FewShotBundle
    top_k: int = 3  # smaller than the vanilla pipeline — more system-prompt budget needed
    candidate_pool: int = 20
    alpha: float | None = None
    max_new_tokens: int = 256
    max_input_tokens: int = 4096
    b2_variant: str = (
        "v03"  # "v03" → structured persona block (headline); "v02_one_liner" → audit only
    )
    name: str = "prompt_persona"

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Turn] | None = None,
        *,
        seed: int | None = None,
    ) -> Response:
        """Run the prompt-persona pipeline end-to-end."""
        if self.b2_variant not in ("v03", "v02_one_liner"):
            raise ValueError(
                f"b2_variant must be 'v03' or 'v02_one_liner', got {self.b2_variant!r}"
            )
        if self.b2_variant == "v03" and self.few_shots.persona_id != persona.persona_id:
            raise ValueError(
                f"few-shot bundle persona_id={self.few_shots.persona_id!r} does not match "
                f"persona persona_id={persona.persona_id!r}"
            )

        # Build system block first so we can size the token budget for retrieval.
        if self.b2_variant == "v03":
            system_text = render_b2_persona_block(persona, self.few_shots)
        else:
            system_text = render_b2_simple_system(persona)

        knowledge_chunks = self.knowledge_store.query_hybrid(
            query, top_k=self.top_k, candidate_pool=self.candidate_pool, alpha=self.alpha
        )

        fixed_overhead = estimate_token_count(system_text) + estimate_token_count(query) + 64
        kept_chunks, dropped = trim_chunks_to_token_budget(
            knowledge_chunks,
            fixed_overhead_tokens=fixed_overhead,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
        )
        if dropped:
            logger.warning(
                "prompt_persona ({}) trimmed {} retrieved chunks", self.b2_variant, dropped
            )

        user_block = render_b2_user_block(query, kept_chunks)
        history_msgs = self._history_to_chat_messages(history) if history else None
        prompt = self.backend.format_persona_prompt(
            system_text=system_text, user_text=user_block, history=history_msgs
        )
        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        if seed is not None:
            gen_kwargs["seed"] = seed
        text = self.backend.generate(prompt, **gen_kwargs)

        return Response(
            text=text,
            retrieved_knowledge=kept_chunks,
            retrieved_persona={},  # persona lives in the system block, not retrieval
            prompt_used=prompt,
            steering_applied=False,
            drift_signal=None,
            metadata={
                "baseline": self.name,
                "backend": self.backend.name,
                "persona_id": persona.persona_id,
                "b2_variant": self.b2_variant,
                "few_shot_count": len(self.few_shots.exchanges),
                "constraint_case_count": sum(
                    1 for ex in self.few_shots.exchanges if ex.is_constraint_case
                ),
                "fusion_mode": "rrf" if self.alpha is None else f"weighted_sum(alpha={self.alpha})",
                "top_k": self.top_k,
                "trimmed_chunks": dropped,
                "seed": seed,
            },
        )

    @staticmethod
    def _history_to_chat_messages(history: list[Turn]):
        """Translate `Turn` items to the backend's `ChatMessage` shape."""
        from persona_rag.models.base import ChatMessage

        return [ChatMessage(role=t.role, content=t.content) for t in history]
