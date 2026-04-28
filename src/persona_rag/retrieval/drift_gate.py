"""LLM-as-judge drift gate.

One LLM call per turn. The judge sees the declared persona block, a window
of recent history, and the most-recent (user, assistant) pair, and answers
whether the assistant's last turn drifted away from the persona's voice,
worldview, or constraints. The judge returns a flag plus a confidence
scalar; the gate combines (flag == "drift") with (confidence >= threshold)
to decide whether the responder uses the cheap path (typed retrieval +
greedy responder) or the gated path (augmented retrieval + N candidates +
hybrid rerank).

This is the only drift detector in the architecture. The cheap path runs
no post-hoc check on the responder's output, by design — gate-reliability
is tested directly via a calibration sweep over the hand-authored
drift-trajectory corpus, and a post-hoc verification pass is recorded as
future-scope work rather than shipped here.

The gate judge is intentionally a different model from the responder
backbone (cross-family, to avoid self-bias). It is also a different model
from the rerank judge inside the hybrid ranker (cross-family, to avoid
double-counting the same model's preferences in both gating and ranking).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.templates import (
    DEFAULT_HISTORY_WINDOW,
    DriftCheck,
    parse_drift_gate_response,
    render_drift_gate_prompt,
)
from persona_rag.schema.persona import Persona


@dataclass
class LlmJudgeDriftGate:
    """One LLM call per turn returning ``flag`` + ``confidence``.

    `confidence_threshold` is the operational knob — at threshold ``0.5`` a
    flag of ``drift`` with confidence ``>= 0.5`` triggers the gated path.
    The threshold sweep at evaluation time reports gate behaviour at a
    range of thresholds.

    `retry_on_malformed` controls a single-retry path: when the parser
    fails on both JSON and the regex fallback, the gate re-issues the
    same prompt once (no template change). Helps the model recover from
    a one-off generation glitch (multi-output answers, code-fence noise,
    early stop) without paying the cost on every turn.
    """

    judge: LLMBackend
    confidence_threshold: float = 0.5
    history_window: int = DEFAULT_HISTORY_WINDOW
    max_new_tokens: int = 256
    temperature: float = 0.0
    retry_on_malformed: bool = True

    @property
    def name(self) -> str:
        return f"drift_gate(judge={self.judge.name})"

    def check(
        self,
        *,
        persona: Persona,
        query: str,
        history: list[Turn] | None = None,
    ) -> DriftCheck:
        """Run the gate for the current turn.

        Args:
            persona: Pydantic persona model loaded from YAML.
            query: the user's most-recent turn (the one the responder is
                about to answer).
            history: full conversation history threaded so far. The gate
                evaluates the last assistant turn in this history; if there
                is no prior assistant turn, the gate returns
                ``flag=ok, should_gate=False`` by construction.
        """
        history = history or []
        last_assistant = self._last_assistant_text(history)

        if last_assistant is None:
            # No prior assistant content for the gate to judge — the cheap
            # path is always taken at conversation start, and the cheap-path
            # responder sets the conversational baseline for subsequent
            # turns.
            logger.debug("drift_gate: no prior assistant turn — should_gate=False")
            return DriftCheck(
                flag="ok",
                confidence=0.0,
                rationale="(no prior assistant turn to judge)",
                should_gate=False,
                raw_response="",
            )

        prompt = render_drift_gate_prompt(
            persona=persona,
            history=history,
            current_user_turn=query,
            last_assistant_turn=last_assistant,
            history_window=self.history_window,
        )
        raw = self.judge.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        check = parse_drift_gate_response(
            raw,
            confidence_threshold=self.confidence_threshold,
        )
        if check.malformed and self.retry_on_malformed:
            logger.warning("drift_gate: malformed response; retrying once")
            raw = self.judge.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            check = parse_drift_gate_response(
                raw,
                confidence_threshold=self.confidence_threshold,
            )
        logger.info(
            "drift_gate: flag={} confidence={:.2f} should_gate={} (threshold={:.2f}){}",
            check.flag,
            check.confidence,
            check.should_gate,
            self.confidence_threshold,
            " [malformed-after-retry]" if check.malformed else "",
        )
        return check

    @staticmethod
    def _last_assistant_text(history: list[Turn]) -> str | None:
        """Most-recent assistant turn text, or ``None`` if none in history."""
        for turn in reversed(history):
            if turn.role == "assistant":
                return turn.content
        return None


__all__ = ["LlmJudgeDriftGate"]
