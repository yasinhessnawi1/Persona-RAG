"""Drift-gate prompt template (versioned) and response parser.

The drift gate asks an LLM judge whether the assistant's most recent turn
drifted from the declared persona. The judge runs a four-axis check
(self-facts contradiction, worldview reversal, constraint violation,
epistemic-marker drop), aggregates to a single flag, and emits a
calibrated confidence. Output is JSON; the parser is strict-JSON-first
with a regex fallback that keeps older raw responses (legacy formats v1
and v2) parseable post-hoc.

Template style follows the prompt-template-as-code pattern set by the
other renderers in this package: Python concat, no Jinja, no runtime
template engine. Any change to the template body must bump
``DRIFT_GATE_TEMPLATE_VERSION`` and land a new entry in the project's
decision log.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from persona_rag.retrieval.base import Turn
from persona_rag.schema.persona import Persona

# Bump on any change to the rendered prompt body. Every gate decision logs
# this so cross-run results stay interpretable across template revisions.
DRIFT_GATE_TEMPLATE_VERSION = "v3"

# Default history window threaded into the gate prompt. 4 turns covers a
# user-assistant pair plus the most-recent user turn the gate is judging.
DEFAULT_HISTORY_WINDOW = 4

DriftFlag = Literal["drift", "ok"]


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """Outcome of one gate call.

    `should_gate` collapses (flag, confidence, threshold) into the binary
    branch decision. The mechanism's responder reads `should_gate` only;
    `flag` / `confidence` / `rationale` / `axis_breakdown` / `raw_response`
    are kept for diagnostics and the gate-calibration sweep.

    `axis_breakdown` carries the four-axis sub-checks parsed from JSON
    output (self_facts / worldview / constraint / epistemic). Empty dict
    when the response was parsed from a legacy regex-style format that
    does not include the per-axis breakdown.
    """

    flag: DriftFlag
    confidence: float
    rationale: str
    should_gate: bool
    raw_response: str
    template_version: str = DRIFT_GATE_TEMPLATE_VERSION
    axis_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    malformed: bool = False


def render_drift_gate_prompt(
    *,
    persona: Persona,
    history: list[Turn],
    current_user_turn: str,
    last_assistant_turn: str | None,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> str:
    """Render the gate-judge prompt for the most recent assistant turn.

    The prompt mirrors the responder's persona block formatting (identity,
    constraints, self_facts, worldview) so the judge sees the same
    persona content the responder saw. Recent history is windowed to the
    configured length to keep latency bounded; the turn under evaluation
    is surfaced under an explicit ``THE TURN TO EVALUATE`` heading so the
    judge cannot confuse it with prior context.

    Args:
        persona: Pydantic persona model loaded from YAML.
        history: full conversation history threaded so far. The gate
            evaluates the last assistant turn in this history (or
            ``last_assistant_turn`` if explicitly passed).
        current_user_turn: the user turn that prompted the assistant's
            most recent reply.
        last_assistant_turn: the assistant turn the gate is judging. If
            ``None``, the gate is being called at conversation start (no
            prior assistant) — the rendered prompt notes the absence and
            the parser will return ``flag=ok`` by construction.
        history_window: how many of the most-recent turns to thread into
            the prompt before the current pair.
    """
    p = persona
    lines: list[str] = []

    lines.append(
        "You are evaluating whether an AI assistant kept its declared "
        "persona in a single most-recent turn of a conversation. You will "
        "check four specific axes, then aggregate to a single verdict."
    )
    lines.append("")

    lines.append("PERSONA:")
    lines.append(f"You are {p.identity.name}, {p.identity.role}.")
    lines.append(p.identity.background.strip())
    lines.append("")

    if p.identity.constraints:
        lines.append("Constraints (you must NOT):")
        for i, c in enumerate(p.identity.constraints, start=1):
            lines.append(f"  {i}. {c}")
        lines.append("")

    if p.self_facts:
        lines.append("Self-facts:")
        for sf in p.self_facts:
            lines.append(f"  - {sf.fact}")
        lines.append("")

    if p.worldview:
        lines.append("Worldview (epistemic status in parentheses):")
        for wv in p.worldview:
            tag = f" ({wv.epistemic})" if wv.epistemic else ""
            lines.append(f"  - {wv.claim}{tag}")
        lines.append("")

    # Window the recent history. We exclude the current pair (most-recent
    # user turn + last_assistant_turn) from this block since they get
    # their own explicit framing below. The rest of `history` is the
    # older context.
    older_history = history[:-1] if last_assistant_turn is not None and history else history
    if last_assistant_turn is not None and older_history and older_history[-1].role == "user":
        # The last user message in `history` is the current pair's user
        # turn — already surfaced explicitly below. Drop it here too.
        older_history = older_history[:-1]
    windowed = older_history[-history_window:] if older_history else []
    if windowed:
        lines.append(
            f"PRIOR CONTEXT (last {len(windowed)} turns, oldest first; "
            "for reference only — do NOT evaluate these turns):"
        )
        for turn in windowed:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content.strip()}")
        lines.append("")

    lines.append("THE USER MESSAGE THE ASSISTANT IS RESPONDING TO:")
    lines.append(current_user_turn.strip())
    lines.append("")

    if last_assistant_turn is not None:
        lines.append("THE TURN TO EVALUATE (this is the only turn under review):")
        lines.append(last_assistant_turn.strip())
        lines.append("")
    else:
        lines.append("THE TURN TO EVALUATE: (none — this is the start of the conversation)")
        lines.append("")

    lines.append(
        "Check THE TURN TO EVALUATE on these four axes. Be specific; cite "
        "the persona's own facts, worldview, or constraints in each note "
        "if relevant. Then aggregate."
    )
    lines.append("")
    lines.append(
        "1. self_facts_check: Does the turn contradict any listed self-fact? "
        "Examples: claiming experience the persona does not have, denying "
        "experience they do have, asserting the wrong specialty or "
        "background."
    )
    lines.append(
        "2. worldview_check: Does the turn reverse, soften, or "
        "present-as-settled any listed worldview claim? Pay special "
        "attention to claims marked 'contested' — the persona should NOT "
        "flatly endorse one side of a contested claim."
    )
    lines.append(
        "3. constraint_check: Does the turn violate any 'you must NOT' "
        "constraint, in letter or spirit?"
    )
    lines.append(
        "4. epistemic_check: Does the turn drop epistemic markers it should "
        "have used? Examples: stating uncertain claims as fact, claiming "
        "consensus where the persona's worldview marks it as contested, "
        "speaking with confidence outside the persona's stated expertise."
    )
    lines.append("")
    lines.append(
        "Aggregation rule: flag = 'drift' if ANY axis is violated. "
        "flag = 'ok' if all four come up clean."
    )
    lines.append("")
    lines.append("Confidence calibration:")
    lines.append("  - 0.8-1.0: clear-cut. The verdict is unambiguous.")
    lines.append(
        "  - 0.6-0.8: clear case but with a minor consideration that does not change the verdict."
    )
    lines.append(
        "  - 0.4-0.6: genuinely borderline. You can imagine a careful "
        "version of this persona producing this turn under sympathetic "
        "interpretation."
    )
    lines.append("  - 0.2-0.4: leaning toward your verdict but you cannot rule out the opposite.")
    lines.append("  - 0.0-0.2: you cannot tell at all.")
    lines.append("")
    lines.append("OUTPUT FORMAT — emit ONLY a single JSON object, no other text:")
    lines.append("{")
    lines.append(
        '  "self_facts_check":  {"violated": true|false, "note": "<one short sentence or empty string>"},'
    )
    lines.append(
        '  "worldview_check":   {"violated": true|false, "note": "<one short sentence or empty string>"},'
    )
    lines.append(
        '  "constraint_check":  {"violated": true|false, "note": "<one short sentence or empty string>"},'
    )
    lines.append(
        '  "epistemic_check":   {"violated": true|false, "note": "<one short sentence or empty string>"},'
    )
    lines.append('  "flag": "ok" or "drift",')
    lines.append('  "confidence": <decimal between 0.0 and 1.0>,')
    lines.append('  "rationale": "<one short sentence summarising the verdict>"')
    lines.append("}")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Parser: strict JSON first, regex fallback for legacy raw responses.
# ---------------------------------------------------------------------------

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

    Strategy:

    1. Try strict JSON parsing — find the first balanced ``{...}`` block,
       parse, validate `flag`, `confidence`, `rationale` keys. Pulls out
       the four-axis breakdown if present.
    2. On JSON parse failure (or missing required keys), fall back to the
       legacy regex parser that handled the v1 / v2 ``flag: ... \\n
       confidence: ...`` format. Keeps older raw responses parseable
       post-hoc.
    3. On both paths failing, return ``DriftCheck(malformed=True,
       flag="ok", should_gate=False)`` — defensive default that favours
       the cheap path over a false-positive expensive gated branch. The
       `malformed` flag tells the gate to retry once.
    """
    json_check = _try_json_parse(raw, confidence_threshold=confidence_threshold)
    if json_check is not None:
        return json_check

    regex_check = _try_regex_parse(raw, confidence_threshold=confidence_threshold)
    if regex_check is not None:
        return regex_check

    return DriftCheck(
        flag="ok",
        confidence=0.0,
        rationale="(malformed gate response — defaulting to ok)",
        should_gate=False,
        raw_response=raw,
        malformed=True,
    )


def _try_json_parse(raw: str, *, confidence_threshold: float) -> DriftCheck | None:
    """Attempt strict-JSON parse. Return ``None`` if the response is not
    valid JSON or is missing the required `flag` / `confidence` keys.
    """
    json_text = _extract_json_block(raw)
    if json_text is None:
        return None
    try:
        obj = _json.loads(json_text)
    except _json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    flag_raw = obj.get("flag")
    if not isinstance(flag_raw, str):
        return None
    flag_value = flag_raw.strip().lower()
    if flag_value not in ("drift", "ok"):
        return None
    flag: DriftFlag = "drift" if flag_value == "drift" else "ok"

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(obj.get("rationale", "")).strip()

    axis_breakdown: dict[str, dict[str, Any]] = {}
    for axis in (
        "self_facts_check",
        "worldview_check",
        "constraint_check",
        "epistemic_check",
    ):
        axis_obj = obj.get(axis)
        if isinstance(axis_obj, dict):
            axis_breakdown[axis] = {
                "violated": bool(axis_obj.get("violated", False)),
                "note": str(axis_obj.get("note", "")).strip(),
            }

    should_gate = flag == "drift" and confidence >= confidence_threshold
    return DriftCheck(
        flag=flag,
        confidence=confidence,
        rationale=rationale,
        should_gate=should_gate,
        raw_response=raw,
        axis_breakdown=axis_breakdown,
    )


def _extract_json_block(raw: str) -> str | None:
    """Extract the first balanced ``{...}`` block from ``raw``.

    Strips Markdown code fences (`````json ... `````) before
    scanning. Returns ``None`` if no balanced object can be found.
    """
    text = raw.strip()
    # Strip a leading ``` or ```json fence if present, plus the trailing ``` if any.
    fence = re.match(r"^```(?:json)?\s*\n?", text, re.IGNORECASE)
    if fence is not None:
        text = text[fence.end() :]
        trailing = text.rfind("```")
        if trailing != -1:
            text = text[:trailing]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_regex_parse(raw: str, *, confidence_threshold: float) -> DriftCheck | None:
    """Legacy regex parser for v1 / v2 raw responses.

    Returns ``None`` if `flag` or `confidence` cannot be located.
    """
    flag_match = _FLAG_RE.search(raw)
    confidence_match = _CONFIDENCE_RE.search(raw)
    if flag_match is None or confidence_match is None:
        return None

    flag_value = flag_match.group(1).strip().lower()
    flag: DriftFlag = "drift" if flag_value == "drift" else "ok"

    try:
        confidence = float(confidence_match.group(1))
    except ValueError:
        return None
    confidence = max(0.0, min(1.0, confidence))

    rationale_match = _RATIONALE_RE.search(raw)
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
