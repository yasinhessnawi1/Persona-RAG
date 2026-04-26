"""Vanilla RAG pipeline: hybrid knowledge retrieval, no persona conditioning."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Response, Turn
from persona_rag.retrieval.prompt_templates import (
    B1_VANILLA_RAG_SYSTEM,
    estimate_token_count,
    render_b1_user_block,
    trim_chunks_to_token_budget,
)
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_store import KnowledgeStore


@dataclass
class VanillaRAG:
    """Hybrid retrieval + neutral RAG instructions; persona ignored.

    The `persona` argument to `respond` is accepted for interface symmetry with
    persona-aware pipelines but is *not* used in prompt assembly.
    `Response.metadata` records `persona_ignored=True` so eval harnesses can
    confirm.
    """

    backend: LLMBackend
    knowledge_store: KnowledgeStore
    top_k: int = 5
    candidate_pool: int = 20
    alpha: float | None = None  # None → RRF (default); float → weighted-sum ablation
    max_new_tokens: int = 256
    max_input_tokens: int = 4096
    name: str = "vanilla_rag"

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Turn] | None = None,
    ) -> Response:
        """Run the vanilla RAG pipeline end-to-end."""
        knowledge_chunks = self.knowledge_store.query_hybrid(
            query, top_k=self.top_k, candidate_pool=self.candidate_pool, alpha=self.alpha
        )

        # Token-budget guard: trim retrieved chunks if they exceed the budget.
        fixed_overhead = (
            estimate_token_count(B1_VANILLA_RAG_SYSTEM) + estimate_token_count(query) + 64
        )
        kept_chunks, dropped = trim_chunks_to_token_budget(
            knowledge_chunks,
            fixed_overhead_tokens=fixed_overhead,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
        )
        if dropped:
            logger.warning(
                "vanilla_rag trimmed {} retrieved chunks to fit token budget", dropped
            )

        user_block = render_b1_user_block(query, kept_chunks)
        prompt = self.backend.format_persona_prompt(
            system_text=B1_VANILLA_RAG_SYSTEM, user_text=user_block, history=None
        )
        text = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)

        return Response(
            text=text,
            retrieved_knowledge=kept_chunks,
            retrieved_persona={},
            prompt_used=prompt,
            steering_applied=False,
            drift_signal=None,
            metadata={
                "baseline": self.name,
                "backend": self.backend.name,
                "persona_id": persona.persona_id,
                "persona_ignored": True,
                "fusion_mode": "rrf" if self.alpha is None else f"weighted_sum(alpha={self.alpha})",
                "top_k": self.top_k,
                "trimmed_chunks": dropped,
            },
        )
