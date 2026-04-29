"""Probe-aware multi-turn evaluation runner.

Generates assistant turns by replaying user turns through a
``RetrievalPipeline``. For ``counterfactual`` probes, the runner adds the
designated counter-evidence chunk to the knowledge store immediately
before the probe turn and removes it immediately after — leaving the
store byte-equivalent to its pre-injection state for the next turn.

This module is decoupled from the evaluation runner in ``runner.py``: it
produces ``EvalConversation`` objects (the same shape every metric
consumes) so the same Spec-08 metric stack scores probe transcripts
without modification. The probe-injection lifecycle and the per-turn
metadata (was the chunk in top-k, at what rank) live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from persona_rag.benchmarks.base import (
    BenchmarkConversation,
    CounterfactualChunk,
)
from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.schema.persona import Persona


@runtime_checkable
class _RespondingPipeline(Protocol):
    """Subset of ``RetrievalPipeline`` we depend on (avoids importing retrieval here)."""

    def respond(
        self,
        query: str,
        persona: Persona,
        history: list[Any] | None = None,
        *,
        seed: int | None = None,
    ) -> Any: ...


@runtime_checkable
class _InjectableStore(Protocol):
    """Subset of ``KnowledgeStore`` we depend on for probe injection."""

    def add_documents(self, documents: list[Any]) -> int: ...

    def remove_documents(self, doc_ids: list[str]) -> int: ...


@dataclass
class ProbeInjectionLog:
    """Per-turn record of one probe injection. Used for verification + reporting."""

    conversation_id: str
    probe_id: str
    probe_type: str
    probe_turn_index: int
    injected_chunk_id: str | None
    injected_chunk_in_topk: bool | None = None
    injected_chunk_rank: int | None = None  # 0-indexed; None if not retrieved


def _estimate_tokens(text: str) -> int:
    """Cheap ``ceil(len/4)`` token estimate. Mirrors `prompt_templates.estimate_token_count`.

    Inlined here so this module stays decoupled from `persona_rag.retrieval`
    at import time (avoids the same cycle as the local Turn import).
    """
    return max(1, (len(text) + 3) // 4)


def _trim_history_to_budget(
    history: list[Any],
    *,
    max_tokens: int,
) -> list[Any]:
    """Keep the most recent (user, assistant) pairs that fit within ``max_tokens``.

    Trims from the front, in turn-pair units so the user/assistant alternation
    invariant is preserved. The mechanism's prompt builder still does its own
    retrieval-budget trimming downstream; this guard exists so the threaded
    history alone does not push the prompt past the model's context limit on
    long multi-turn conversations.

    With Gemma-2-9B's 4096-token budget and a typical persona/system block
    around ~600 tokens plus retrieved chunks plus generation budget, ~1800
    tokens is a defensible default for history headroom. Callers can override
    via ``ProbeRunner.max_history_tokens``.
    """
    if max_tokens <= 0 or not history:
        return list(history)
    # Pair-walk from the end so we always keep the most-recent context.
    kept_reversed: list[Any] = []
    spent = 0
    # Walk in turn-pair units. ``history`` alternates user / assistant.
    # If the list ends in ``assistant``, that is one "open" pair to add first
    # (1 message); subsequent pairs are 2 messages each.
    i = len(history) - 1
    while i >= 0:
        # Identify the current pair span.
        msg_b = history[i]
        msg_a = history[i - 1] if i - 1 >= 0 else None
        text_b = getattr(msg_b, "content", "")
        text_a = getattr(msg_a, "content", "") if msg_a is not None else ""
        cost = _estimate_tokens(text_b) + _estimate_tokens(text_a) + 8  # 8 tokens role/sep tax
        if kept_reversed and spent + cost > max_tokens:
            break
        if msg_a is not None:
            kept_reversed.append(msg_b)
            kept_reversed.append(msg_a)
            spent += cost
            i -= 2
        else:
            kept_reversed.append(msg_b)
            spent += _estimate_tokens(text_b) + 4
            i -= 1
    kept_reversed.reverse()
    return kept_reversed


@dataclass
class ProbeRunner:
    """Replay a list of ``BenchmarkConversation`` objects through one pipeline.

    ``Turn`` (the retrieval-pipeline history class) is constructed locally
    so this module doesn't take a hard dependency on
    ``persona_rag.retrieval.base`` (which imports stores transitively).

    ``max_history_tokens`` caps the threaded conversation history's contribution
    to the prompt. Default 1800 tokens leaves headroom on a 4096-token Gemma-2
    context for the persona/system block, retrieval payload, and generation
    budget. The mechanism's own prompt builder is still responsible for
    trimming retrieval-side content; this guard prevents the *history* alone
    from pushing the prompt past the model's context limit on long
    conversations.
    """

    pipeline: _RespondingPipeline
    knowledge_store: _InjectableStore | None = None
    chunks: dict[str, CounterfactualChunk] = field(default_factory=dict)
    seed: int = 42
    mechanism_label: str = "unknown"
    max_history_tokens: int = 1800

    def replay(
        self,
        persona: Persona,
        conversations: list[BenchmarkConversation],
    ) -> tuple[list[EvalConversation], list[ProbeInjectionLog]]:
        """Generate one ``EvalConversation`` per input. Return (transcripts, injection logs)."""
        from persona_rag.retrieval.base import Turn  # local import to avoid cycle
        from persona_rag.stores.knowledge_store import KnowledgeDocument

        transcripts: list[EvalConversation] = []
        injection_logs: list[ProbeInjectionLog] = []
        for conv in conversations:
            history: list[Any] = []
            scored_turns: list[ScoredTurn] = []
            per_turn_meta: list[dict[str, Any]] = []
            for turn_ix, user_text in enumerate(conv.user_turns):
                injected_doc_id: str | None = None
                injection_log: ProbeInjectionLog | None = None
                if (
                    conv.probe is not None
                    and conv.probe.probe_turn_index == turn_ix
                    and conv.probe.probe_type == "counterfactual"
                    and conv.probe.injected_chunk_id is not None
                ):
                    chunk = self.chunks.get(conv.probe.injected_chunk_id)
                    if chunk is None:
                        raise KeyError(
                            f"probe {conv.probe.probe_id!r}: chunk "
                            f"{conv.probe.injected_chunk_id!r} not loaded; "
                            "pass it via ProbeRunner.chunks"
                        )
                    if self.knowledge_store is None:
                        raise RuntimeError(
                            f"probe {conv.probe.probe_id!r}: counterfactual "
                            "injection needs a knowledge_store"
                        )
                    injected_doc_id = chunk.chunk_id
                    self.knowledge_store.add_documents(
                        [
                            KnowledgeDocument(
                                doc_id=chunk.chunk_id,
                                text=chunk.text,
                                source=chunk.source_label,
                                metadata={
                                    "injected_for_probe": conv.probe.probe_id,
                                    "contradicts": chunk.contradicts,
                                },
                            )
                        ]
                    )
                    injection_log = ProbeInjectionLog(
                        conversation_id=conv.conversation_id,
                        probe_id=conv.probe.probe_id,
                        probe_type=conv.probe.probe_type,
                        probe_turn_index=turn_ix,
                        injected_chunk_id=chunk.chunk_id,
                    )

                threaded_history = _trim_history_to_budget(
                    history, max_tokens=self.max_history_tokens
                )
                try:
                    response = self.pipeline.respond(
                        user_text,
                        persona,
                        history=threaded_history,
                        seed=self.seed,
                    )
                finally:
                    if injected_doc_id is not None and self.knowledge_store is not None:
                        # Always eject, even if generation raised, so the next
                        # conversation does not see residue from this one.
                        self.knowledge_store.remove_documents([injected_doc_id])

                if injection_log is not None and getattr(response, "retrieved_knowledge", None):
                    retrieved_ids = [getattr(c, "id", None) for c in response.retrieved_knowledge]
                    # ``KnowledgeStore`` chunk ids are ``"<doc_id>:<chunk_ix>"`` —
                    # the injected doc is never re-chunked in our authoring
                    # convention (≤ 100 words ⇒ one chunk), so the chunk id is
                    # ``"<chunk_id>:0"``. Match either the prefix or full id.
                    chunk_prefix = f"{injection_log.injected_chunk_id}:"
                    rank: int | None = None
                    for ix, cid in enumerate(retrieved_ids):
                        if cid is None:
                            continue
                        if cid == injection_log.injected_chunk_id or cid.startswith(chunk_prefix):
                            rank = ix
                            break
                    injection_log.injected_chunk_in_topk = rank is not None
                    injection_log.injected_chunk_rank = rank
                    injection_logs.append(injection_log)
                elif injection_log is not None:
                    injection_log.injected_chunk_in_topk = False
                    injection_logs.append(injection_log)

                assistant_text = getattr(response, "text", "")
                scored_turns.append(
                    ScoredTurn(
                        turn_index=turn_ix,
                        user_text=user_text,
                        assistant_text=assistant_text,
                    )
                )
                meta: dict[str, Any] = {
                    "is_probe_turn": (
                        conv.probe is not None and conv.probe.probe_turn_index == turn_ix
                    ),
                    "probe_type": (conv.probe.probe_type if conv.probe is not None else None),
                    "history_messages_threaded": len(threaded_history),
                    "history_messages_total": len(history),
                }
                pipeline_meta = getattr(response, "metadata", None)
                if isinstance(pipeline_meta, dict):
                    meta["pipeline_metadata"] = pipeline_meta
                if injection_log is not None:
                    meta["injected_chunk_id"] = injection_log.injected_chunk_id
                    meta["injected_chunk_in_topk"] = injection_log.injected_chunk_in_topk
                    meta["injected_chunk_rank"] = injection_log.injected_chunk_rank
                per_turn_meta.append(meta)

                history.append(Turn(role="user", content=user_text))
                history.append(Turn(role="assistant", content=assistant_text))

            transcripts.append(
                EvalConversation(
                    conversation_id=(
                        f"{conv.persona_id}::{self.mechanism_label}::{conv.conversation_id}"
                    ),
                    mechanism=self.mechanism_label,
                    persona_id=conv.persona_id,
                    turns=tuple(scored_turns),
                    per_turn_metadata=tuple(per_turn_meta),
                )
            )
        logger.info(
            "ProbeRunner: replayed {} conversations through {} (seed={}); "
            "{} probe injections logged",
            len(conversations),
            self.mechanism_label,
            self.seed,
            len(injection_logs),
        )
        return transcripts, injection_logs


__all__ = ["ProbeInjectionLog", "ProbeRunner"]
