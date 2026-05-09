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

from dataclasses import dataclass, field

from loguru import logger

from persona_rag.models.base import LLMBackend
from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.templates import (
    DEFAULT_HISTORY_WINDOW,
    DriftCheck,
    parse_drift_gate_response,
    render_drift_gate_prompt,
)
from persona_rag.retrieval.templates.drift_gate import DriftFlag
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


# ---------------------------------------------------------------------------
# Oracle gate (table-driven, no LLM call)
# ---------------------------------------------------------------------------


# Bumped on any change to the oracle's firing rule. Mirrors the LLM gate's
# template-version field so per-turn metadata stays interpretable across runs.
ORACLE_GATE_TEMPLATE_VERSION = "oracle_v1"


class _OracleJudgeStub:
    """Object exposing ``.name`` so ``DriftGatedMechanism`` reads ``oracle``.

    The mechanism's ``_cheap_path`` / ``_gated_path`` populate
    ``gate_judge_model`` from ``self.drift_gate.judge.name``. The oracle
    issues no LLM call, but the field still needs a stable string for
    cost-tracking + diagnostic readouts.
    """

    name = "oracle"


@dataclass
class OracleDriftGate:
    """Probe-aware oracle gate. No LLM call.

    The gate's decision for any ``(conversation_id, turn_index)`` pair is
    pre-computed from the probe metadata: fire on ``probe_turn_index`` and
    ``probe_turn_index + 1`` for Type-A (``self_fact_challenge``) and
    Type-B (``counterfactual``) probes; never fire on Type-C
    (``constraint_bait``) — Type-C tests constraint deflection, not drift
    recovery.

    The gate is stateful: ``ProbeRunner`` (or its subclass) advances
    ``current_conversation_id`` + ``current_turn_index`` before each call
    to the host mechanism's ``respond(...)``, so ``check(...)`` can read
    the cursor and emit the right ``DriftCheck`` without needing the
    benchmark schema piped through every layer.

    The exposed attribute names (``judge``, ``confidence_threshold``,
    ``name``) mirror :class:`LlmJudgeDriftGate` so
    :class:`DriftGatedMechanism` reads the right metadata fields without
    branching on gate type.
    """

    # Map ``conversation_id -> {turn_index: should_gate}``.  The runner
    # populates this once per replay() call from the loaded benchmark
    # conversations; the gate only reads it.
    decisions: dict[str, dict[int, bool]] = field(default_factory=dict)

    # Per-turn diagnostic strings the gate logs into ``rationale``.
    rationales: dict[str, dict[int, str]] = field(default_factory=dict)

    # Cursor advanced by the runner before each ``respond(...)`` call.
    current_conversation_id: str | None = None
    current_turn_index: int | None = None

    # Sentinel values mirroring the LLM-gate's metadata schema.
    confidence_threshold: float = 1.0  # not meaningful for the oracle
    judge: _OracleJudgeStub = field(default_factory=_OracleJudgeStub)

    @property
    def name(self) -> str:
        return f"oracle_drift_gate({ORACLE_GATE_TEMPLATE_VERSION})"

    def set_cursor(self, conversation_id: str, turn_index: int) -> None:
        """Advance the cursor before ``respond(...)``. Called by the runner."""
        self.current_conversation_id = conversation_id
        self.current_turn_index = turn_index

    def register_conversation(
        self,
        *,
        conversation_id: str,
        n_turns: int,
        probe_type: str | None,
        probe_turn_index: int | None,
    ) -> None:
        """Pre-compute the per-turn ``should_gate`` table for one conversation.

        Type A (``self_fact_challenge``) and Type B (``counterfactual``)
        probes fire the oracle on the probe turn and the immediate
        follow-up (when in range). Type C (``constraint_bait``) and
        no-probe conversations leave every turn at ``should_gate=False``.
        """
        per_turn: dict[int, bool] = {ix: False for ix in range(n_turns)}
        per_rat: dict[int, str] = {ix: "" for ix in range(n_turns)}

        if probe_type in {"self_fact_challenge", "counterfactual"} and probe_turn_index is not None:
            for offset in (0, 1):
                ix = probe_turn_index + offset
                if 0 <= ix < n_turns:
                    per_turn[ix] = True
                    label = "probe-turn" if offset == 0 else "follow-up"
                    per_rat[ix] = (
                        f"oracle: probe_type={probe_type} "
                        f"probe_turn={probe_turn_index} "
                        f"current_turn={ix} ({label})"
                    )

        for ix in range(n_turns):
            if not per_turn[ix]:
                if probe_type == "constraint_bait":
                    per_rat[ix] = (
                        f"oracle: probe_type=constraint_bait "
                        f"probe_turn={probe_turn_index} current_turn={ix} "
                        "(Type-C never fires)"
                    )
                elif probe_type is None:
                    per_rat[ix] = "oracle: no probe — cheap path"
                else:
                    per_rat[ix] = (
                        f"oracle: probe_type={probe_type} "
                        f"probe_turn={probe_turn_index} current_turn={ix} "
                        "(outside probe-window)"
                    )

        self.decisions[conversation_id] = per_turn
        self.rationales[conversation_id] = per_rat

    def check(
        self,
        *,
        persona: Persona,
        query: str,
        history: list[Turn] | None = None,
    ) -> DriftCheck:
        """Look up the pre-computed decision for the runner's current cursor.

        ``persona``, ``query``, and ``history`` are accepted only to keep
        the signature compatible with :class:`LlmJudgeDriftGate` — the
        oracle ignores them (decision rests entirely on probe metadata).
        ``DriftCheck`` is populated so ``DriftGatedMechanism``'s metadata
        flow and the harness's ``DriftQualityMetric`` see the same field
        shape they get from the LLM gate.
        """
        del persona, query, history  # unused; cursor-driven decision

        if self.current_conversation_id is None or self.current_turn_index is None:
            # Defensive default. The runner is responsible for advancing the
            # cursor; an unset cursor means the gate fell out of sync.
            logger.warning("oracle_drift_gate: cursor unset — defaulting to should_gate=False")
            return DriftCheck(
                flag="ok",
                confidence=0.0,
                rationale="oracle: cursor unset",
                should_gate=False,
                raw_response="",
                template_version=ORACLE_GATE_TEMPLATE_VERSION,
            )

        conv = self.decisions.get(self.current_conversation_id)
        if conv is None:
            logger.warning(
                "oracle_drift_gate: conversation_id={} not registered — should_gate=False",
                self.current_conversation_id,
            )
            return DriftCheck(
                flag="ok",
                confidence=0.0,
                rationale=f"oracle: conversation {self.current_conversation_id} not registered",
                should_gate=False,
                raw_response="",
                template_version=ORACLE_GATE_TEMPLATE_VERSION,
            )

        should_gate = bool(conv.get(self.current_turn_index, False))
        rationale = self.rationales.get(self.current_conversation_id, {}).get(
            self.current_turn_index, ""
        )
        flag: DriftFlag = "drift" if should_gate else "ok"
        confidence = 1.0 if should_gate else 0.0
        logger.info(
            "oracle_drift_gate: conv={} turn={} should_gate={}",
            self.current_conversation_id,
            self.current_turn_index,
            should_gate,
        )
        return DriftCheck(
            flag=flag,
            confidence=confidence,
            rationale=rationale,
            should_gate=should_gate,
            raw_response="",
            template_version=ORACLE_GATE_TEMPLATE_VERSION,
        )


__all__ = ["ORACLE_GATE_TEMPLATE_VERSION", "LlmJudgeDriftGate", "OracleDriftGate"]
