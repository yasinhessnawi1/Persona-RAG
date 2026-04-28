"""Typed-retrieval pipeline with per-turn identity re-grounding (ID-RAG pattern).

This pipeline retrieves persona content from the four typed memory stores at
every turn (identity always; self_facts / worldview / episodic semantically),
plus knowledge from the hybrid knowledge store, and assembles them into a
structured system block. The architectural difference from the prompt-persona
pipeline is that persona content is *retrieved per turn* rather than baked
once into the system block at conversation start — and identity in particular
is re-grounded every turn to suppress drift in long conversations.

The block is rendered via Python string concatenation (no Jinja dependency),
following the same style as ``render_b2_persona_block``. Ablation switches are
surfaced as dataclass fields so a Hydra config or a test can flip them
individually:

- ``use_identity_every_turn`` — drop identity + constraints from turn 1+ when
  False (the ID-RAG ablation).
- ``use_epistemic_tags`` — strip the ``(belief)`` / ``(fact)`` annotation from
  worldview claims when False, isolating the tag's contribution to voice.
- ``use_episodic`` / ``write_episodic`` — episodic retrieval / write-back; off
  by default per the spec's semester scope.
- ``epistemic_allowlist`` — restricts which tags participate in worldview
  retrieval at the store-filter layer.
- ``top_k_*`` — per-store retrieval depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Response, Turn
from persona_rag.retrieval.prompt_templates import (
    estimate_token_count,
    trim_chunks_to_token_budget,
)
from persona_rag.schema.chunker import ChunkKind, PersonaChunk
from persona_rag.schema.persona import Persona
from persona_rag.stores.episodic_store import EpisodicStore
from persona_rag.stores.identity_store import IdentityStore
from persona_rag.stores.knowledge_chunk import KnowledgeChunk
from persona_rag.stores.knowledge_store import KnowledgeStore
from persona_rag.stores.self_facts_store import SelfFactsStore
from persona_rag.stores.worldview_store import WorldviewStore

DEFAULT_EPISTEMIC_ALLOWLIST: tuple[str, ...] = (
    "fact",
    "belief",
    "hypothesis",
    "contested",
)


@dataclass
class TypedRetrievalRAG:
    """Typed retrieval + per-turn identity re-grounding (ID-RAG) pipeline.

    Construct with the four typed memory stores from Spec 03, the knowledge
    store from Spec 04, and an `LLMBackend`. Personas must already be indexed
    in the four stores (the runner is responsible for registering them).
    """

    backend: LLMBackend
    knowledge_store: KnowledgeStore
    identity_store: IdentityStore
    self_facts_store: SelfFactsStore
    worldview_store: WorldviewStore
    episodic_store: EpisodicStore

    # Ablation switches.
    use_identity_every_turn: bool = True
    use_epistemic_tags: bool = True
    use_episodic: bool = False
    write_episodic: bool = False
    epistemic_allowlist: tuple[str, ...] = DEFAULT_EPISTEMIC_ALLOWLIST

    # Top-k per store.
    top_k_self_facts: int = 3
    top_k_worldview: int = 3
    top_k_episodic: int = 0
    top_k_knowledge: int = 5

    # Knowledge-retrieval fusion knobs (mirror PromptPersonaRAG's defaults).
    candidate_pool: int = 20
    alpha: float | None = None  # None → RRF.

    # Generation budget.
    max_new_tokens: int = 256
    max_input_tokens: int = 4096

    # Identifier the runner / wandb display.
    name: str = "typed_retrieval"

    # Cached per-instance counter for episodic write-back; used as a stable
    # turn_id when the caller does not pass conversation history.
    _episodic_write_counter: int = field(default=0, init=False, repr=False)

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Turn] | None = None,
        *,
        seed: int | None = None,
    ) -> Response:
        """Run the typed-retrieval pipeline end-to-end for one query."""
        if persona.persona_id is None:
            raise ValueError(
                "Persona.persona_id is None — cannot route typed-store retrieval without it."
            )
        history = history or []
        turn_ix = len(history) // 2  # one round-trip = user + assistant turn

        # 1. Identity + constraints (always retrieved when ID-RAG on; only
        #    on turn 0 otherwise — the ID-RAG ablation).
        id_rag_fired = self.use_identity_every_turn or turn_ix == 0
        if id_rag_fired:
            id_chunks = self.identity_store.query(query, top_k=1, persona_id=persona.persona_id)
        else:
            id_chunks = []
        identity_chunks = [c for c in id_chunks if c.kind == "identity"]
        constraint_chunks = [c for c in id_chunks if c.kind == "constraint"]

        # 2. Self_facts via top-k semantic retrieval.
        self_fact_chunks: list[PersonaChunk] = []
        if self.top_k_self_facts > 0:
            self_fact_chunks = self.self_facts_store.query(
                query, top_k=self.top_k_self_facts, persona_id=persona.persona_id
            )

        # 3. Worldview with epistemic-tag filter.
        worldview_chunks: list[PersonaChunk] = []
        if self.top_k_worldview > 0:
            worldview_chunks = self.worldview_store.query(
                query,
                top_k=self.top_k_worldview,
                persona_id=persona.persona_id,
                epistemic=list(self.epistemic_allowlist) if self.epistemic_allowlist else None,
            )

        # 4. Episodic (off by default).
        episodic_chunks: list[PersonaChunk] = []
        if self.use_episodic and self.top_k_episodic > 0:
            episodic_chunks = self.episodic_store.query(
                query, top_k=self.top_k_episodic, persona_id=persona.persona_id
            )

        # 5. Knowledge — hybrid retrieval (unchanged from B1/B2).
        knowledge_chunks = self.knowledge_store.query_hybrid(
            query,
            top_k=self.top_k_knowledge,
            candidate_pool=self.candidate_pool,
            alpha=self.alpha,
        )

        # 6. Build the system block, sized to the prompt budget. Persona-side
        #    chunks are short enough that we trim only the knowledge leg.
        system_text_no_knowledge = render_typed_system_block(
            identity_chunks=identity_chunks,
            constraint_chunks=constraint_chunks,
            self_fact_chunks=self_fact_chunks,
            worldview_chunks=worldview_chunks,
            episodic_chunks=episodic_chunks,
            knowledge_chunks=[],  # placeholder for sizing
            use_epistemic_tags=self.use_epistemic_tags,
        )
        fixed_overhead = (
            estimate_token_count(system_text_no_knowledge) + estimate_token_count(query) + 64
        )
        kept_knowledge, dropped = trim_chunks_to_token_budget(
            knowledge_chunks,
            fixed_overhead_tokens=fixed_overhead,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
        )
        if dropped:
            logger.warning("typed_retrieval trimmed {} retrieved knowledge chunk(s)", dropped)

        system_text = render_typed_system_block(
            identity_chunks=identity_chunks,
            constraint_chunks=constraint_chunks,
            self_fact_chunks=self_fact_chunks,
            worldview_chunks=worldview_chunks,
            episodic_chunks=episodic_chunks,
            knowledge_chunks=kept_knowledge,
            use_epistemic_tags=self.use_epistemic_tags,
        )

        # 7. Render via the backend's role-aware formatter.
        history_msgs = self._history_to_chat_messages(history) if history else None
        prompt = self.backend.format_persona_prompt(
            system_text=system_text, user_text=query, history=history_msgs
        )

        # 8. Generate.
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if seed is not None:
            gen_kwargs["seed"] = seed
        text = self.backend.generate(prompt, **gen_kwargs)

        # 9. Optional episodic write-back.
        if self.write_episodic:
            self._write_episodic(persona.persona_id, query=query, text=text, turn_ix=turn_ix)

        # 10. Per-store diagnostic logging.
        epi_mix = _count_epistemic_tags(worldview_chunks)
        logger.info(
            "typed_retrieval[{}] turn={} id_rag={} self_facts={} worldview={} "
            "epi_mix={} episodic={} knowledge={} knowledge_dropped={}",
            persona.persona_id,
            turn_ix,
            id_rag_fired,
            len(self_fact_chunks),
            len(worldview_chunks),
            epi_mix,
            len(episodic_chunks),
            len(kept_knowledge),
            dropped,
        )

        retrieved_persona: dict[ChunkKind, list[PersonaChunk]] = {
            "identity": identity_chunks,
            "constraint": constraint_chunks,
            "self_fact": self_fact_chunks,
            "worldview": worldview_chunks,
            "episodic": episodic_chunks,
        }

        return Response(
            text=text,
            retrieved_knowledge=kept_knowledge,
            retrieved_persona=retrieved_persona,
            prompt_used=prompt,
            steering_applied=False,
            drift_signal=None,
            metadata={
                "mechanism": self.name,
                "backend": self.backend.name,
                "persona_id": persona.persona_id,
                "turn_id": turn_ix,
                "id_rag_fired": id_rag_fired,
                "use_identity_every_turn": self.use_identity_every_turn,
                "use_epistemic_tags": self.use_epistemic_tags,
                "use_episodic": self.use_episodic,
                "write_episodic": self.write_episodic,
                "epistemic_allowlist": list(self.epistemic_allowlist),
                "top_k_self_facts": self.top_k_self_facts,
                "top_k_worldview": self.top_k_worldview,
                "top_k_episodic": self.top_k_episodic,
                "top_k_knowledge": self.top_k_knowledge,
                "fusion_mode": "rrf" if self.alpha is None else f"weighted_sum(alpha={self.alpha})",
                "trimmed_chunks": dropped,
                "worldview_epistemic_mix": dict(epi_mix),
                "self_fact_chunk_ids": [c.id for c in self_fact_chunks],
                "worldview_chunk_ids": [c.id for c in worldview_chunks],
                "episodic_chunk_ids": [c.id for c in episodic_chunks],
                "knowledge_chunk_ids": [c.id for c in kept_knowledge],
                "seed": seed,
            },
        )

    # ----------------------------------------------------------- helpers

    def _write_episodic(
        self,
        persona_id: str,
        *,
        query: str,
        text: str,
        turn_ix: int,
    ) -> None:
        """Append one episodic memory entry summarising this turn."""
        now = datetime.now(UTC)
        # Stable id — collisions across turns/runs are explicit upsert overwrites.
        self._episodic_write_counter += 1
        chunk_id = f"{persona_id}:episodic:runtime:{now.strftime('%Y%m%dT%H%M%S')}_{self._episodic_write_counter:04d}"
        chunk = PersonaChunk(
            id=chunk_id,
            text=f"User: {query}\nAssistant: {text}",
            kind="episodic",
            metadata={
                "persona_id": persona_id,
                "kind": "episodic",
                "timestamp": now.isoformat(),
                "decay_t0": now.isoformat(),
                "turn_id": str(turn_ix),
            },
        )
        try:
            self.episodic_store.write(chunk)
        except Exception as exc:  # pragma: no cover — log and re-raise per CLAUDE.md
            logger.error("episodic write_back failed for {}: {}", persona_id, exc)
            raise

    @staticmethod
    def _history_to_chat_messages(history: list[Turn]):
        """Translate `Turn` items to the backend's `ChatMessage` shape."""
        from persona_rag.models.base import ChatMessage

        return [ChatMessage(role=t.role, content=t.content) for t in history]


# ---------------------------------------------------------------------------
# system-block renderer (mirrors prompt_templates.py style — no Jinja dep)
# ---------------------------------------------------------------------------


def render_typed_system_block(
    *,
    identity_chunks: list[PersonaChunk],
    constraint_chunks: list[PersonaChunk],
    self_fact_chunks: list[PersonaChunk],
    worldview_chunks: list[PersonaChunk],
    episodic_chunks: list[PersonaChunk],
    knowledge_chunks: list[KnowledgeChunk],
    use_epistemic_tags: bool = True,
) -> str:
    """Assemble the M1 typed-retrieval system block.

    Concat order (rationale in research note): identity opener → constraints
    early → self_facts → worldview (epistemic-tagged) → episodic (optional)
    → knowledge passages. The user query is appended by the caller via the
    backend's role-aware formatter, not inside this block.
    """
    lines: list[str] = []

    # 1. Identity opener.
    if identity_chunks:
        for chunk in identity_chunks:
            lines.append(chunk.text.strip())
        lines.append("")

    # 2. Constraints.
    if constraint_chunks:
        lines.append("You must NOT:")
        for i, c in enumerate(constraint_chunks, start=1):
            lines.append(f"  {i}. {c.text.strip()}")
        lines.append("")

    # 3. Self-facts (retrieved).
    if self_fact_chunks:
        lines.append("Relevant facts about yourself:")
        for c in self_fact_chunks:
            lines.append(f"  - {c.text.strip()}")
        lines.append("")

    # 4. Worldview (epistemic-tagged).
    if worldview_chunks:
        if use_epistemic_tags:
            lines.append("Your views (epistemic status in parentheses):")
        else:
            lines.append("Your views:")
        for c in worldview_chunks:
            tag = c.metadata.get("epistemic", "")
            if use_epistemic_tags and tag:
                lines.append(f"  - {c.text.strip()} ({tag})")
            else:
                lines.append(f"  - {c.text.strip()}")
        lines.append("")

    # 5. Episodic (optional).
    if episodic_chunks:
        lines.append("Relevant prior discussion:")
        for c in episodic_chunks:
            lines.append(f"  - {c.text.strip()}")
        lines.append("")

    # 6. Knowledge.
    if knowledge_chunks:
        lines.append("Retrieved passages:")
        for i, kc in enumerate(knowledge_chunks, start=1):
            source = kc.metadata.get("source") or kc.metadata.get("doc_id") or "unknown"
            lines.append(f"[{i}] (source: {source}) {kc.text.strip()}")
        lines.append("")

    # 7. RAG instruction footer — ties retrieved content to the response.
    lines.append(
        "Stay in character. When the user asks a question, ground your answer "
        "in the retrieved passages above and cite passage numbers in square "
        "brackets when you rely on a passage. If the passages do not cover "
        "the question, say so rather than inventing detail."
    )

    return "\n".join(lines).rstrip() + "\n"


def _count_epistemic_tags(chunks: list[PersonaChunk]) -> dict[str, int]:
    """Count epistemic tags across a worldview chunk list. Used in metadata."""
    out: dict[str, int] = {}
    for c in chunks:
        tag = c.metadata.get("epistemic", "unknown")
        out[tag] = out.get(tag, 0) + 1
    return out


__all__ = [
    "DEFAULT_EPISTEMIC_ALLOWLIST",
    "TypedRetrievalRAG",
    "render_typed_system_block",
]
