"""30-prompt stability suite used by the smoke test.

Ten prompts in each of three buckets:

- ``factual``: general-knowledge questions with an objectively correct short answer.
  Probes that the model does not produce garbled outputs on routine Q&A.
- ``instruction``: writing / summarization / light reasoning prompts. Probes
  instruction-following and coherence over ~128 tokens.
- ``persona``: system-style instructions ("You are a cautious doctor..."). Probes
  the chat-template path -- especially the Gemma 2 ``system``-folding route, since
  Gemma 2 has no native ``system`` role and folds it into the first user turn.

The coherence heuristic ``looks_coherent`` is deliberately permissive: it catches
the pathological failure modes fp16 quantization tends to produce (NaN -> empty
output, repeat loops, all-whitespace), without trying to judge quality. Quality
judgments are made by hand-spot-checking the logged outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

Bucket = Literal["factual", "instruction", "persona"]


@dataclass(frozen=True, slots=True)
class SuitePrompt:
    """One prompt in the stability suite. ``system`` is None unless the bucket uses it."""

    prompt_id: str
    bucket: Bucket
    user: str
    system: str | None = None


_FACTUAL: tuple[SuitePrompt, ...] = (
    SuitePrompt("f01", "factual", "What is the capital of Norway?"),
    SuitePrompt("f02", "factual", "Who wrote the novel 'Pride and Prejudice'?"),
    SuitePrompt(
        "f03", "factual", "What gas do plants absorb from the atmosphere during photosynthesis?"
    ),
    SuitePrompt("f04", "factual", "In what year did the first human land on the Moon?"),
    SuitePrompt("f05", "factual", "What is the chemical symbol for gold?"),
    SuitePrompt("f06", "factual", "Which planet in our solar system has the most moons?"),
    SuitePrompt("f07", "factual", "What is the largest ocean on Earth?"),
    SuitePrompt("f08", "factual", "Who painted the Mona Lisa?"),
    SuitePrompt(
        "f09", "factual", "What is the boiling point of water at sea level in degrees Celsius?"
    ),
    SuitePrompt("f10", "factual", "Name the three primary additive colors of light."),
)


_INSTRUCTION: tuple[SuitePrompt, ...] = (
    SuitePrompt(
        "i01",
        "instruction",
        "Summarize in two sentences why regular exercise is beneficial for cardiovascular health.",
    ),
    SuitePrompt(
        "i02",
        "instruction",
        "Write a short, polite email declining a meeting invitation due to a schedule conflict.",
    ),
    SuitePrompt(
        "i03",
        "instruction",
        "Explain what a linked list is to someone who has never programmed before. Keep it under 80 words.",
    ),
    SuitePrompt(
        "i04",
        "instruction",
        "List three practical tips for someone learning a new language, with a one-sentence justification for each.",
    ),
    SuitePrompt(
        "i05",
        "instruction",
        "Translate the following English sentence to French: 'The library closes at six o'clock in the evening.'",
    ),
    SuitePrompt(
        "i06",
        "instruction",
        "Convert this plain request into a polite one: 'Give me the report now.'",
    ),
    SuitePrompt(
        "i07",
        "instruction",
        "Given the numbers 4, 7, 2, 9, 5, sort them in ascending order and state the median.",
    ),
    SuitePrompt("i08", "instruction", "Write a haiku about autumn leaves."),
    SuitePrompt(
        "i09",
        "instruction",
        "Briefly compare the programming languages Python and Rust along two axes: typical use case and memory safety.",
    ),
    SuitePrompt(
        "i10",
        "instruction",
        "I have 17 apples and give away 4, then buy 6 more. How many apples do I have? Show your reasoning in one line.",
    ),
)


_PERSONA: tuple[SuitePrompt, ...] = (
    SuitePrompt(
        "p01",
        "persona",
        "I've been feeling tired after meals. What could it be?",
        system="You are a cautious general practitioner. You do not give definitive diagnoses and you always recommend consulting a qualified clinician for specifics.",
    ),
    SuitePrompt(
        "p02",
        "persona",
        "I'm stuck on a math problem about geometric series. Can you help me understand the sum formula?",
        system="You are a patient high-school math tutor. You explain concepts step by step and ask a check-understanding question at the end.",
    ),
    SuitePrompt(
        "p03",
        "persona",
        "What's a good beginner book to learn about Norwegian history?",
        system="You are a knowledgeable librarian who specializes in Scandinavian history. Recommend a specific title with a one-sentence reason.",
    ),
    SuitePrompt(
        "p04",
        "persona",
        "How should I think about learning to cook if I've never really done it?",
        system="You are a warm, encouraging home-cooking instructor. You value building confidence over technical perfection.",
    ),
    SuitePrompt(
        "p05",
        "persona",
        "What's a good first project for someone learning Python?",
        system="You are a pragmatic software mentor. You prefer small, finishable projects over ambitious ones.",
    ),
    SuitePrompt(
        "p06",
        "persona",
        "I want to start running. What should I do in the first week?",
        system="You are a conservative running coach. You prioritize injury prevention and gradual progression.",
    ),
    SuitePrompt(
        "p07",
        "persona",
        "Can you give me a short overview of how the European Union's structure works?",
        system="You are a political science professor who explains complex institutions in plain, neutral language without political commentary.",
    ),
    SuitePrompt(
        "p08",
        "persona",
        "How do I start investing with a very small amount of money?",
        system="You are a careful personal-finance guide. You never recommend specific securities and you always point out the role of risk.",
    ),
    SuitePrompt(
        "p09",
        "persona",
        "I'm writing a story and I need a villain with a believable motivation. Any advice?",
        system="You are a craft-focused creative-writing coach. You emphasize character interiority over shock value.",
    ),
    SuitePrompt(
        "p10",
        "persona",
        "What's a reasonable way to set up version control for a small school project?",
        system="You are a concise senior engineer. You favor the simplest thing that works and avoid over-engineering.",
    ),
)


SUITE: tuple[SuitePrompt, ...] = _FACTUAL + _INSTRUCTION + _PERSONA
assert len(SUITE) == 30, f"stability suite should have 30 prompts, got {len(SUITE)}"


# ---------------------------------------------------------------------------
# Coherence heuristic
# ---------------------------------------------------------------------------


_MIN_OUTPUT_CHARS = 20
# If the output has fewer than this fraction of *unique* trigrams vs total trigrams, the
# text is dominated by a small cycle and we call it pathologically repetitive. A healthy
# paragraph has 90-100% unique trigrams; a pure repeat loop approaches 1/cycle_length.
_MIN_UNIQUE_TRIGRAM_FRACTION = 0.5


def looks_coherent(text: str) -> tuple[bool, str]:
    """Return ``(is_ok, reason)`` -- a permissive pathology check for fp16 Gemma2.

    Catches:
    - empty / whitespace-only output (common failure mode when logits NaN).
    - catastrophic repetition (3-gram spam, a known fp16 Gemma regression pattern).
    - all-punctuation output.

    Does NOT judge correctness or quality. That's the author's spot-check job.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_OUTPUT_CHARS:
        return False, f"output too short ({len(stripped)} chars)"

    if not any(ch.isalnum() for ch in stripped):
        return False, "output has no alphanumeric characters"

    tokens = stripped.split()
    if len(tokens) >= 12:
        trigrams = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        if trigrams:
            unique_fraction = len(set(trigrams)) / len(trigrams)
            if unique_fraction < _MIN_UNIQUE_TRIGRAM_FRACTION:
                most_common, count = Counter(trigrams).most_common(1)[0]
                return (
                    False,
                    f"only {unique_fraction:.0%} of 3-grams are unique "
                    f"(most common: {most_common!r}, {count}x) -- looks like a repeat loop",
                )

    return True, "ok"
