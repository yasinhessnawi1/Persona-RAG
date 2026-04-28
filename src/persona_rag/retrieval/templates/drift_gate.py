"""Drift-gate prompt template (versioned) and response parser.

The drift gate asks an LLM judge whether the assistant's most recent turn
drifted from the declared persona. Output is a flag + confidence + a one-
sentence rationale. The flag * threshold-confidence pair is what the
mechanism uses to branch between the cheap path and the gated path.

Template style follows the prompt-template-as-code pattern set by
``render_b2_persona_block`` and ``render_typed_system_block``: Python concat,
no Jinja, no runtime template engine. Any change to the template body must
bump ``DRIFT_GATE_TEMPLATE_VERSION`` and land a new decisions-log entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from persona_rag.retrieval.base import Turn
from persona_rag.schema.persona import Persona

# Bump on any change to the rendered prompt body. Every gate decision logs
# this so cross-run results stay interpretable across template revisions.
DRIFT_GATE_TEMPLATE_VERSION = "v2"

# Default history window threaded into the gate prompt. 4 turns covers a
# user-assistant pair plus the most-recent user turn the gate is judging.
DEFAULT_HISTORY_WINDOW = 4

DriftFlag = Literal["drift", "ok"]


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """Outcome of one gate call.

    `should_gate` collapses (flag, confidence, threshold) into the binary
    branch decision. The mechanism's responder reads `should_gate` only;
    `flag` / `confidence` / `rationale` / `raw_response` are kept for
    diagnostics and the gate-calibration sweep.
    """

    flag: DriftFlag
    confidence: float
    rationale: str
    should_gate: bool
    raw_response: str
    template_version: str = DRIFT_GATE_TEMPLATE_VERSION


def render_drift_gate_prompt(
    *,
    persona: Persona,
    history: list[Turn],
    current_user_turn: str,
    last_assistant_turn: str | None,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> str:
    """Render the gate-judge prompt for the most recent assistant turn.

    The prompt mirrors M1's persona block formatting (identity / constraints
    / self_facts / worldview) so the judge sees the same persona content the
    responder saw. Recent history is windowed to the configured length to
    keep latency bounded.

    Args:
        persona: Pydantic persona model loaded from YAML.
        history: full conversation history threaded so far. The gate evaluates
            the *last* assistant turn in this history (or
            ``last_assistant_turn`` if explicitly passed).
        current_user_turn: the user turn that prompted the assistant's most
            recent reply. The gate frames its judgment around this.
        last_assistant_turn: the assistant turn the gate is judging. If
            ``None``, the gate is being called at turn 0 (no prior assistant)
            — the rendered prompt notes the absence and the parser will
            return ``flag=ok`` by construction (no drift to detect on turn 0).
        history_window: how many of the most-recent turns to thread into the
            prompt before the current pair.
    """
    p = persona
    lines: list[str] = []

    lines.append(
        "You are evaluating whether an AI assistant kept its declared persona "
        "in the most recent turn of a conversation. Read the persona, the "
        "recent context, and the assistant's most recent reply, then judge "
        "that single reply against the persona."
    )
    lines.append("")

    lines.append("Persona:")
    lines.append(f"You are {p.identity.name}, {p.identity.role}.")
    lines.append(p.identity.background.strip())
    lines.append("")

    if p.identity.constraints:
        lines.append("You must NOT:")
        for i, c in enumerate(p.identity.constraints, start=1):
            lines.append(f"  {i}. {c}")
        lines.append("")

    if p.self_facts:
        lines.append("Relevant facts about yourself:")
        for sf in p.self_facts:
            lines.append(f"  - {sf.fact}")
        lines.append("")

    if p.worldview:
        lines.append("Your views (epistemic status in parentheses):")
        for wv in p.worldview:
            tag = f" ({wv.epistemic})" if wv.epistemic else ""
            lines.append(f"  - {wv.claim}{tag}")
        lines.append("")

    # Window the recent history. We exclude the current pair (most-recent user
    # turn + last_assistant_turn) from this block since they get their own
    # explicit framing below. The rest of `history` is the older context.
    older_history = history[:-1] if last_assistant_turn is not None and history else history
    if last_assistant_turn is not None and older_history and older_history[-1].role == "user":
        # The last user message in `history` is the current pair's user turn —
        # already surfaced explicitly below. Drop it here too.
        older_history = older_history[:-1]
    windowed = older_history[-history_window:] if older_history else []
    if windowed:
        lines.append(f"Recent conversation (last {len(windowed)} turns, oldest first):")
        for turn in windowed:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content.strip()}")
        lines.append("")

    lines.append("Most recent user turn:")
    lines.append(current_user_turn.strip())
    lines.append("")

    if last_assistant_turn is not None:
        lines.append("Most recent assistant turn:")
        lines.append(last_assistant_turn.strip())
        lines.append("")
    else:
        lines.append("Most recent assistant turn: (none — this is turn 0)")
        lines.append("")

    lines.append(
        "A turn is 'ok' when the reply is consistent with the persona's "
        "voice, stated facts, stated views, and constraints. A turn is "
        "'drift' when the reply contradicts or ignores the persona — for "
        "example, ignoring a constraint, asserting a view opposite to a "
        "stated worldview, claiming experience the persona does not have, "
        "or breaking the established voice toward generic-assistant prose."
    )
    lines.append("")
    lines.append("Examples (illustrative — do not copy verbatim):")
    lines.append(
        "  - Assistant gives a confident answer in a domain the persona's "
        "constraints exclude (e.g. financial advice from a CS tutor): drift."
    )
    lines.append(
        "  - Assistant directly endorses a claim the persona's worldview marks as contested: drift."
    )
    lines.append(
        "  - Assistant stays on-topic, reasons within the persona's voice, "
        "and respects the constraints: ok."
    )
    lines.append("")
    lines.append(
        "Calibrate confidence honestly: use values near 1.0 for clear-cut "
        "cases (clear ok OR clear drift), values near 0.5 when the call is "
        "borderline, and values near 0.0 only when you cannot tell at all."
    )
    lines.append("")
    lines.append("Now judge the most recent assistant turn above.")
    lines.append("Answer in exactly this format on three lines:")
    lines.append("flag: ok|drift")
    lines.append("confidence: <decimal between 0.0 and 1.0>")
    lines.append("rationale: <one short sentence>")

    return "\n".join(lines).rstrip() + "\n"


_FLAG_RE = re.compile(r"^\s*flag\s*:\s*(drift|ok)\s*$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"^\s*confidence\s*:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_RATIONALE_RE = re.compile(r"^\s*rationale\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def parse_drift_gate_response(
    raw: str,
    *,
    confidence_threshold: float,
) -> DriftCheck:
    """Parse the judge's raw text response into a `DriftCheck`.

    Tolerates whitespace and case variation around the structured fields. On
    malformed output, returns ``flag=ok, confidence=0.0, should_gate=False`` —
    a defensive default that favours the cheap path rather than triggering an
    expensive false-positive gated branch. The raw response is preserved on
    the `DriftCheck` regardless so debugging stays possible.
    """
    flag_match = _FLAG_RE.search(raw)
    confidence_match = _CONFIDENCE_RE.search(raw)
    rationale_match = _RATIONALE_RE.search(raw)

    if flag_match is None or confidence_match is None:
        return DriftCheck(
            flag="ok",
            confidence=0.0,
            rationale="(malformed gate response — defaulting to ok)",
            should_gate=False,
            raw_response=raw,
        )

    flag_value = flag_match.group(1).strip().lower()
    flag: DriftFlag = "drift" if flag_value == "drift" else "ok"

    try:
        confidence = float(confidence_match.group(1))
    except ValueError:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = rationale_match.group(1).strip() if rationale_match else ""
    should_gate = flag == "drift" and confidence >= confidence_threshold

    return DriftCheck(
        flag=flag,
        confidence=confidence,
        rationale=rationale,
        should_gate=should_gate,
        raw_response=raw,
    )


__all__ = [
    "DEFAULT_HISTORY_WINDOW",
    "DRIFT_GATE_TEMPLATE_VERSION",
    "DriftCheck",
    "DriftFlag",
    "parse_drift_gate_response",
    "render_drift_gate_prompt",
]
