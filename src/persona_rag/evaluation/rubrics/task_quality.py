"""Task-quality rubric: did the assistant correctly answer the user's question?

Persona-independent. The same conversation that scores high on the
persona-adherence rubric can score low here if the assistant stayed in
voice but answered incorrectly. The two rubrics are scored independently
to keep the per-dimension reading clean.
"""

from __future__ import annotations

from persona_rag.schema.persona import Persona

_TASK_QUALITY_DESCRIPTIONS: dict[int, str] = {
    1: "The response is wrong, hallucinated, or fundamentally fails to answer the user's question.",
    2: "The response is mostly wrong or addresses only a small part of the user's question.",
    3: "The response is partially correct and partially missing or incorrect details.",
    4: "The response correctly answers the user's question with minor omissions or imprecisions.",
    5: "The response correctly and completely answers the user's question with relevant detail.",
}


def _render_conversation(conversation_turns: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for i, (u, a) in enumerate(conversation_turns):
        lines.append(f"[Turn {i + 1}] USER: {u}")
        lines.append(f"[Turn {i + 1}] ASSISTANT: {a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Prometheus rubric: one call covering the whole conversation
# ---------------------------------------------------------------------------


def render_task_quality_native_prometheus_prompt(
    *,
    persona: Persona,
    conversation_turns: list[tuple[str, str]],
) -> str:
    """Render Prometheus's Direct Assessment template for task quality.

    The persona definition is included only as context; the rubric
    explicitly tells the judge to score *task quality independent of
    persona*. The persona block helps the judge avoid penalising
    persona-driven content choices that are otherwise correct.
    """
    transcript = _render_conversation(conversation_turns)
    instruction = (
        "Evaluate whether the assistant correctly answered the user's knowledge "
        "questions across this multi-turn conversation, INDEPENDENT of persona "
        "consistency."
    )
    reference = (
        "A response that scores 5 would correctly and completely answer every "
        "knowledge question raised by the user, with relevant detail and no "
        "hallucinated facts."
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
        f"{instruction}\n"
        f"(persona context, not the criterion: {persona.identity.name}, "
        f"{persona.identity.role})\n\n"
        "###Response to evaluate:\n"
        f"{transcript}\n\n"
        "###Reference Answer (Score 5):\n"
        f"{reference}\n\n"
        "###Score Rubrics:\n"
        "[task_quality]\n"
        f"Score 1: {_TASK_QUALITY_DESCRIPTIONS[1]}\n"
        f"Score 2: {_TASK_QUALITY_DESCRIPTIONS[2]}\n"
        f"Score 3: {_TASK_QUALITY_DESCRIPTIONS[3]}\n"
        f"Score 4: {_TASK_QUALITY_DESCRIPTIONS[4]}\n"
        f"Score 5: {_TASK_QUALITY_DESCRIPTIONS[5]}\n\n"
        "###Feedback: "
    )


# ---------------------------------------------------------------------------
# JSON rubric
# ---------------------------------------------------------------------------


def render_task_quality_json_prompt(
    *,
    persona: Persona,
    conversation_turns: list[tuple[str, str]],
) -> str:
    """Render a JSON-output task-quality prompt."""
    transcript = _render_conversation(conversation_turns)
    rubric_lines = "\n".join(
        f"  {score}: {desc}" for score, desc in _TASK_QUALITY_DESCRIPTIONS.items()
    )
    return (
        "You are evaluating whether an AI assistant correctly answered the "
        "user's knowledge questions in a multi-turn conversation. "
        "Score INDEPENDENT of persona -- focus only on factual correctness "
        "and completeness.\n\n"
        f"PERSONA CONTEXT (not part of the criterion): {persona.identity.name}, "
        f"{persona.identity.role}\n\n"
        "CONVERSATION:\n"
        f"{transcript}\n\n"
        "SCORE RUBRIC (1-5):\n"
        f"{rubric_lines}\n\n"
        "Respond with EXACTLY ONE JSON object on a single line, no code fences, "
        'in this shape: {"score": int, "reasoning": str}\n'
    )


__all__ = [
    "render_task_quality_json_prompt",
    "render_task_quality_native_prometheus_prompt",
]
