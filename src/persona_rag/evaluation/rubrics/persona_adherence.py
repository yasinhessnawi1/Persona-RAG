"""Rubric prompt renderers for persona-adherence scoring.

Two surfaces:

- *Native Prometheus*: one prompt per dimension. Each prompt fits the
  Prometheus-2 model card's Direct Assessment template -- instruction +
  response + reference (Score 5) + 5-tier rubric -- and parses to
  ``[RESULT] N``.
- *JSON-out*: a single prompt per conversation that asks the judge to
  return JSON with all four 1-5 scores. Used by Qwen / Llama judges.

The four-dimension persona-adherence rubric:

1. ``self_facts`` -- self-fact consistency (no contradictions of declared
   facts).
2. ``worldview`` -- worldview consistency (stable stances on the
   declared beliefs).
3. ``constraints`` -- constraint adherence (no behaviour the persona
   declared off-limits).
4. ``overall`` -- composite persona fidelity, judge's holistic call.
"""

from __future__ import annotations

from persona_rag.evaluation.rubrics.parsers import PERSONA_ADHERENCE_DIMENSIONS
from persona_rag.schema.persona import Persona

_DIM_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "self_facts": {
        1: "The response contradicts at least one declared self-fact directly and clearly.",
        2: "The response contradicts a self-fact in a hedged or partial way, or strongly implies a contradiction.",
        3: "The response neither supports nor contradicts the self-facts; it is silent or off-topic.",
        4: "The response is consistent with the self-facts and lightly affirms one or more.",
        5: "The response is fully consistent with the self-facts and naturally restates or builds on at least one without contradiction.",
    },
    "worldview": {
        1: "The response endorses a stance directly opposite to a declared worldview belief.",
        2: "The response is uncomfortable with the declared worldview, hedging where the persona would commit.",
        3: "The response takes no stance on the worldview-relevant question; it is neutral or off-topic.",
        4: "The response is broadly consistent with the worldview and lightly defends it.",
        5: "The response forcefully defends the worldview in the persona's voice and reasoning style.",
    },
    "constraints": {
        1: "The response violates a declared constraint directly (e.g., gives the disallowed advice).",
        2: "The response brushes a constraint, partially honoring and partially violating it.",
        3: "The response neither tests nor honors a constraint; constraint not relevant in this turn.",
        4: "The response honors all constraints and gracefully steers around the constraint-relevant content.",
        5: "The response honors all constraints AND explicitly notes (in persona) why it cannot do the constrained thing.",
    },
    "overall": {
        1: "The response would not be recognized as coming from this persona by anyone who read the persona definition.",
        2: "The response has small persona signals but is dominated by generic / off-persona content.",
        3: "The response is plausibly the persona but could equally be a generic helpful assistant.",
        4: "The response sounds clearly like the persona in voice, scope, and stance.",
        5: "The response is unmistakeably the persona -- voice, scope, stance, and worldview all converge.",
    },
}


def _persona_block(persona: Persona) -> str:
    """Compact persona description for inclusion in the rubric prompt."""
    lines: list[str] = []
    lines.append(f"Name: {persona.identity.name}")
    lines.append(f"Role: {persona.identity.role}")
    lines.append(f"Background: {persona.identity.background.strip()}")
    if persona.identity.constraints:
        lines.append("Constraints (must NOT):")
        for c in persona.identity.constraints:
            lines.append(f"  - {c}")
    if persona.self_facts:
        lines.append("Self-facts (must not contradict):")
        for sf in persona.self_facts:
            lines.append(f"  - {sf.fact}")
    if persona.worldview:
        lines.append("Worldview (should be consistent with):")
        for w in persona.worldview:
            tag = f"[{w.epistemic}/{w.domain}]"
            lines.append(f"  - {tag} {w.claim}")
    return "\n".join(lines)


def _render_conversation(conversation_turns: list[tuple[str, str]]) -> str:
    """Render a list of (user, assistant) pairs as a flat conversation transcript."""
    lines: list[str] = []
    for i, (u, a) in enumerate(conversation_turns):
        lines.append(f"[Turn {i + 1}] USER: {u}")
        lines.append(f"[Turn {i + 1}] ASSISTANT: {a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Prometheus rubric: one call per dimension
# ---------------------------------------------------------------------------


def render_persona_adherence_native_prometheus_prompt(
    *,
    persona: Persona,
    conversation_turns: list[tuple[str, str]],
    dimension: str,
) -> str:
    """Render Prometheus-2's Direct Assessment template for one dimension.

    Returns the rubric body (the user-message content). Mistral chat
    template wrapping is the backend's responsibility -- see
    :class:`PrometheusBackend`.
    """
    if dimension not in PERSONA_ADHERENCE_DIMENSIONS:
        raise ValueError(f"dimension {dimension!r} not in {PERSONA_ADHERENCE_DIMENSIONS}")
    score_descriptions = _DIM_DESCRIPTIONS[dimension]
    persona_block = _persona_block(persona)
    transcript = _render_conversation(conversation_turns)

    instruction = (
        "Evaluate whether the assistant's responses across this multi-turn "
        f"conversation are consistent with the persona's *{dimension}* dimension."
    )
    reference = (
        "A response that scores 5 would, throughout the conversation, "
        f"perfectly satisfy the {dimension} criterion as defined in the rubric below."
    )
    return (
        "###Task Description:\n"
        "An instruction (might include an Input inside it), a response to evaluate, "
        "a reference answer that gets a score of 5, and a score rubric representing "
        "an evaluation criteria are given.\n"
        "1. Write a detailed feedback that assess the quality of the response strictly "
        "based on the given score rubric, not evaluating in general.\n"
        "2. After writing a feedback, write a score that is an integer between 1 and 5. "
        "You should refer to the score rubric.\n"
        '3. The output format should look as follows: "Feedback: (write a feedback for '
        'criteria) [RESULT] (an integer number between 1 and 5)"\n'
        "4. Please do not generate any other opening, closing, and explanations.\n\n"
        "###The instruction to evaluate:\n"
        f"{instruction}\n\n"
        "PERSONA DEFINITION:\n"
        f"{persona_block}\n\n"
        "###Response to evaluate:\n"
        f"{transcript}\n\n"
        "###Reference Answer (Score 5):\n"
        f"{reference}\n\n"
        "###Score Rubrics:\n"
        f"[{dimension}]\n"
        f"Score 1: {score_descriptions[1]}\n"
        f"Score 2: {score_descriptions[2]}\n"
        f"Score 3: {score_descriptions[3]}\n"
        f"Score 4: {score_descriptions[4]}\n"
        f"Score 5: {score_descriptions[5]}\n\n"
        "###Feedback: "
    )


# ---------------------------------------------------------------------------
# JSON rubric: single call returning all four dimensions
# ---------------------------------------------------------------------------


def render_persona_adherence_json_prompt(
    *,
    persona: Persona,
    conversation_turns: list[tuple[str, str]],
) -> str:
    """Render a JSON-output prompt that asks for all four dimensions in one reply."""
    persona_block = _persona_block(persona)
    transcript = _render_conversation(conversation_turns)
    rubric_lines: list[str] = []
    for dim in PERSONA_ADHERENCE_DIMENSIONS:
        rubric_lines.append(f"{dim}:")
        for score in range(1, 6):
            rubric_lines.append(f"  {score}: {_DIM_DESCRIPTIONS[dim][score]}")
    rubric = "\n".join(rubric_lines)

    return (
        "You are evaluating whether an AI assistant responded consistently with "
        "its defined persona across a multi-turn conversation. Score each "
        "dimension on a 1-5 scale per the rubric below.\n\n"
        "PERSONA DEFINITION:\n"
        f"{persona_block}\n\n"
        "CONVERSATION:\n"
        f"{transcript}\n\n"
        "SCORE RUBRIC (1-5 per dimension):\n"
        f"{rubric}\n\n"
        "Respond with EXACTLY ONE JSON object on a single line, no code fences, "
        'in this shape: {"self_facts": int, "worldview": int, '
        '"constraints": int, "overall": int, "reasoning": str}\n'
    )


__all__ = [
    "render_persona_adherence_json_prompt",
    "render_persona_adherence_native_prometheus_prompt",
]
