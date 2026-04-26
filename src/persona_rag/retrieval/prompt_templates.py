"""Prompt templates: vanilla-RAG instructions and structured persona block.

Templates are committed as code so the wording is reviewable, diffable across
revisions, and testable. Both backbones call
``LLMBackend.format_persona_prompt(system_text, user_text, history)`` to render
the final string — Gemma folds the system block into the first user turn (it
has no native `system` role); Llama uses the native `system` role.

The structured persona block follows the prompt-engineered role-play pattern:

    1. Identity opener: "You are {name}, {role}. {background}"
    2. Self-facts as bulleted list, with confidence in parens for high-bar facts.
    3. Worldview claims as bulleted list, with epistemic flag in parens.
    4. Constraints as a numbered "You must not" block.
    5. Few-shot dialogue exchanges (one demonstrating a constraint case).
    6. The current retrieval-augmented user turn.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from persona_rag.schema.chunker import PersonaChunk
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_chunk import KnowledgeChunk

# ---------------------------------------------------------------------------
# few-shot model
# ---------------------------------------------------------------------------


class FewShotTurn(BaseModel):
    """One turn inside a few-shot exchange. Role is `"user"` or `"assistant"`."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1)


class FewShotExchange(BaseModel):
    """One ``user → assistant`` few-shot exchange.

    `is_constraint_case=True` flags the example as the per-persona constraint
    demonstration (off-domain deflection, contested-topic flag, etc.). At least
    one such example per persona is recommended. The persona-block renderer
    orders constraint-case few-shots last so their boundary-setting effect is
    freshest in the model's context.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    is_constraint_case: bool = False
    turns: list[FewShotTurn] = Field(..., min_length=2)


class FewShotBundle(BaseModel):
    """All few-shots for one persona, loaded from ``personas/examples/<id>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str = Field(..., min_length=1)
    exchanges: list[FewShotExchange] = Field(..., min_length=1)

    @classmethod
    def from_yaml(cls, path: Path) -> FewShotBundle:
        """Load and validate few-shots for one persona."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        if "persona_id" not in raw:
            raw["persona_id"] = path.stem
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# B1 — vanilla RAG (no persona)
# ---------------------------------------------------------------------------

# Neutral, literature-standard RAG instructions. Committed verbatim so future
# readers can audit the wording. Do NOT tune persona-specifically — B1 is the
# floor baseline.
B1_VANILLA_RAG_SYSTEM = (
    "You are a helpful assistant. Use the following retrieved passages to answer "
    "the user's question. If the passages do not contain the answer, say so "
    "rather than guessing. Cite the passage number in square brackets when you "
    "rely on it."
)


def render_b1_user_block(query: str, knowledge_chunks: list[KnowledgeChunk]) -> str:
    """B1 user block: numbered retrieved passages, then the user's question."""
    parts: list[str] = []
    if knowledge_chunks:
        parts.append("Retrieved passages:")
        for i, chunk in enumerate(knowledge_chunks, start=1):
            source = chunk.metadata.get("source") or chunk.metadata.get("doc_id") or "unknown"
            parts.append(f"[{i}] (source: {source}) {chunk.text}")
        parts.append("")
    parts.append(f"User question: {query}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Structured persona block + few-shots
# ---------------------------------------------------------------------------


def render_b2_persona_block(persona: Persona, few_shots: FewShotBundle) -> str:
    """Assemble the structured persona system prompt (identity / facts / views / constraints / few-shots)."""
    p = persona
    lines: list[str] = []

    # 1. Identity opener.
    lines.append(f"You are {p.identity.name}, {p.identity.role}.")
    lines.append(p.identity.background.strip())
    lines.append("")

    # 2. Self-facts.
    if p.self_facts:
        lines.append("About yourself:")
        for sf in p.self_facts:
            tag = "" if sf.confidence >= 0.95 else f"  (confidence ≈ {sf.confidence:.2f})"
            lines.append(f"  - {sf.fact}{tag}")
        lines.append("")

    # 3. Worldview, epistemically tagged.
    if p.worldview:
        lines.append("Your views (epistemic status in parentheses where relevant):")
        for wv in p.worldview:
            tag = f"  ({wv.epistemic})" if wv.epistemic != "fact" else ""
            lines.append(f"  - {wv.claim}{tag}")
        lines.append("")

    # 4. Constraints.
    if p.identity.constraints:
        lines.append("You must NOT:")
        for i, c in enumerate(p.identity.constraints, start=1):
            lines.append(f"  {i}. {c}")
        lines.append("")

    # 5. Few-shots — non-constraint first, constraint cases last so they are most recent.
    sorted_exchanges = sorted(few_shots.exchanges, key=lambda ex: ex.is_constraint_case)
    if sorted_exchanges:
        lines.append("Example exchanges that show how you respond:")
        for ex in sorted_exchanges:
            lines.append(f"  -- Example: {ex.title} --")
            for turn in ex.turns:
                role_label = "User" if turn.role == "user" else "You"
                lines.append(f"  {role_label}: {turn.content}")
        lines.append("")

    # 6. RAG instruction footer (so the prompt persona still uses retrieved knowledge).
    lines.append(
        "When the user asks a question, use the retrieved passages below as your "
        "factual grounding. Stay in character. Cite passage numbers in square "
        "brackets when you rely on a passage."
    )

    return "\n".join(lines).rstrip() + "\n"


def render_b2_user_block(
    query: str,
    knowledge_chunks: list[KnowledgeChunk],
    persona_chunks_by_kind: dict[str, list[PersonaChunk]] | None = None,
) -> str:
    """User block for the prompt-persona pipeline: retrieved passages + the user's question.

    Persona chunks are *not* injected into the user block — the persona lives
    in the system prompt. The argument is accepted for interface symmetry with
    pipelines that DO inject persona chunks per-turn; this renderer ignores it.
    """
    del persona_chunks_by_kind  # persona lives in the system block, not user-side context
    parts: list[str] = []
    if knowledge_chunks:
        parts.append("Retrieved passages:")
        for i, chunk in enumerate(knowledge_chunks, start=1):
            source = chunk.metadata.get("source") or chunk.metadata.get("doc_id") or "unknown"
            parts.append(f"[{i}] (source: {source}) {chunk.text}")
        parts.append("")
    parts.append(f"User question: {query}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# audit-only variant: one-liner persona prefix
# ---------------------------------------------------------------------------


def render_b2_simple_system(persona: Persona) -> str:
    """One-liner persona prefix.

    EXISTS ONLY for strength-comparison auditing — never appears in headline
    results. Used to validate that the structured persona block is measurably
    stronger than a naïve one-liner persona prompt.
    """
    return (
        f"You are {persona.identity.name}, {persona.identity.role}. "
        "Use the retrieved passages to answer the user's question."
    )


# ---------------------------------------------------------------------------
# token-budget helpers
# ---------------------------------------------------------------------------


def estimate_token_count(text: str) -> int:
    """Cheap character-quartet token estimate.

    Avoids loading the actual tokenizer at every prompt-assembly call. Worst-
    case under-counts on dense English (real ratio ~4 chars/token but Gemma /
    Llama tokenizers occasionally split aggressively); good enough for the
    truncation guard's headroom.
    """
    return max(1, (len(text) + 3) // 4)


def trim_chunks_to_token_budget(
    chunks: list[KnowledgeChunk],
    *,
    fixed_overhead_tokens: int,
    max_input_tokens: int,
    max_new_tokens: int,
    safety_margin: int = 128,
) -> tuple[list[KnowledgeChunk], int]:
    """Drop lowest-rank chunks until the assembled prompt fits.

    Returns (kept_chunks, dropped_count). `chunks` must be ordered with the
    most-relevant first; we drop from the tail. The safety margin is the
    headroom we leave for the token-count estimate's worst-case under-count.
    """
    budget = max_input_tokens - max_new_tokens - safety_margin - fixed_overhead_tokens
    if budget <= 0:
        return [], len(chunks)
    kept: list[KnowledgeChunk] = []
    spent = 0
    for chunk in chunks:
        cost = estimate_token_count(chunk.text) + 16  # 16 tokens for citation header
        if spent + cost > budget:
            break
        kept.append(chunk)
        spent += cost
    dropped = len(chunks) - len(kept)
    return kept, dropped


__all__ = [
    "B1_VANILLA_RAG_SYSTEM",
    "FewShotBundle",
    "FewShotExchange",
    "FewShotTurn",
    "estimate_token_count",
    "render_b1_user_block",
    "render_b2_persona_block",
    "render_b2_simple_system",
    "render_b2_user_block",
    "trim_chunks_to_token_budget",
]
