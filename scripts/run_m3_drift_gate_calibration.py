"""Replay the LLM-judge drift gate over the hand-authored drift-trajectory corpus.

Loads the in_persona + drifting conversations for each persona, replays
the gate on every (turn >= 1) of every conversation, and reports the
gate's flag-rate per condition at each candidate threshold. The corpus
shape (3 personas x 2 conditions x 6 user-assistant pairs, with user
turns identical between conditions) makes this a clean
flag-rate-differential test for gate quality.

Usage:
    uv run python scripts/run_m3_drift_gate_calibration.py \\
        --judge llama \\
        --thresholds 0.3,0.4,0.5,0.6,0.7 \\
        --output-dir results/m3_drift_gate_calibration/$(date +%Y%m%d_%H%M%S)

V100-only: loads Llama-3.1-8B (4-bit) as the gate judge.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from persona_rag.models import HFBackendConfig, LlamaBackend
from persona_rag.retrieval.base import Turn
from persona_rag.retrieval.drift_gate import LlmJudgeDriftGate
from persona_rag.schema.conversation import (
    DriftTrajectoryConversation,
    assert_user_turns_match,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIFT_TRAJECTORY_ROOT = REPO_ROOT / "benchmarks_data" / "drift_trajectory"


def _build_judge(name: str) -> Any:
    """Build the gate judge backend."""
    if name != "llama":
        raise ValueError(f"--judge must be 'llama' for the canonical sweep, got {name!r}")
    cfg = HFBackendConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        name="llama3.1-8b-instruct",
        revision=None,
        compute_dtype="float16",
        attn_implementation="eager",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        max_input_tokens=4096,
        trust_remote_code=False,
        warmup_nan_guard=True,
    )
    return LlamaBackend(cfg)


def _load_conversations(persona_id: str) -> dict[str, DriftTrajectoryConversation]:
    """Load the in_persona + drifting conversations for one persona."""
    base = DRIFT_TRAJECTORY_ROOT / persona_id
    convs = {
        "in_persona": DriftTrajectoryConversation.from_yaml(base / "in_persona.yaml"),
        "drifting": DriftTrajectoryConversation.from_yaml(base / "drifting.yaml"),
    }
    assert_user_turns_match(convs.values())
    return convs


def _gate_each_turn(
    *,
    persona: Persona,
    conv: DriftTrajectoryConversation,
    judge: Any,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Replay the conversation and call the gate after each assistant turn.

    Skips the first turn (no prior assistant for the gate to judge).
    Returns one record per turn keyed by threshold so flag-rates can be
    computed in one pass.
    """
    history: list[Turn] = []
    out: list[dict[str, Any]] = []
    pairs = list(zip(conv.user_turn_texts(), conv.assistant_turn_texts(), strict=True))
    for turn_ix, (user_text, assistant_text) in enumerate(pairs):
        if turn_ix == 0:
            history.extend(
                [
                    Turn(role="user", content=user_text),
                    Turn(role="assistant", content=assistant_text),
                ]
            )
            continue
        history.extend(
            [
                Turn(role="user", content=user_text),
                Turn(role="assistant", content=assistant_text),
            ]
        )
        gate_default = LlmJudgeDriftGate(judge=judge, confidence_threshold=0.5)
        check = gate_default.check(persona=persona, query=user_text, history=history[:-1])
        record = {
            "persona_id": persona.persona_id,
            "condition": conv.condition,
            "turn_ix": turn_ix,
            "user_turn": user_text,
            "assistant_turn": assistant_text,
            "gate_flag": check.flag,
            "gate_confidence": check.confidence,
            "gate_rationale": check.rationale,
            "gate_template_version": check.template_version,
            "raw_response": check.raw_response,
            "drift_level_authored": _authored_drift(conv, turn_ix),
        }
        for t in thresholds:
            record[f"should_gate@{t}"] = check.flag == "drift" and check.confidence >= t
        out.append(record)
    return out


def _authored_drift(conv: DriftTrajectoryConversation, turn_ix: int) -> str | None:
    """Return the author-annotated drift_level for a given turn (drifting only)."""
    if conv.condition != "drifting":
        return None
    assistant_pos = turn_ix * 2 + 1
    if 0 <= assistant_pos < len(conv.turns):
        return conv.turns[assistant_pos].drift_level
    return None


def _summarise(records: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    """Compute flag-rate per (condition, threshold) and the differential."""
    rates: dict[str, dict[str, float]] = {}
    for cond in ("in_persona", "drifting"):
        rates[cond] = {}
        cond_records = [r for r in records if r["condition"] == cond]
        n = len(cond_records)
        for t in thresholds:
            triggered = sum(1 for r in cond_records if r[f"should_gate@{t}"])
            rates[cond][f"{t}"] = triggered / n if n else 0.0
    differentials = {
        f"{t}": rates["drifting"][f"{t}"] - rates["in_persona"][f"{t}"] for t in thresholds
    }
    verdict = {
        f"{t}": "differential >= 0.5" if differentials[f"{t}"] >= 0.5 else "below threshold"
        for t in thresholds
    }
    return {"flag_rates": rates, "differential": differentials, "verdict": verdict}


def main() -> int:
    """Calibration sweep entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--judge", default="llama", choices=["llama"])
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated list of gate thresholds to sweep.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--personas",
        default="cs_tutor,historian,climate_scientist",
        help="Comma-separated persona ids; default = all three shipped personas.",
    )
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    persona_ids = [p.strip() for p in args.personas.split(",") if p.strip()]
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    judge = _build_judge(args.judge)
    all_records: list[dict[str, Any]] = []

    for pid in persona_ids:
        persona = Persona.from_yaml(REPO_ROOT / "personas" / f"{pid}.yaml")
        convs = _load_conversations(pid)
        for cond, conv in convs.items():
            logger.info("running gate on persona={} condition={}", pid, cond)
            records = _gate_each_turn(
                persona=persona, conv=conv, judge=judge, thresholds=thresholds
            )
            all_records.extend(records)

    summary = _summarise(all_records, thresholds)

    (out_root / "raw.json").write_text(
        json.dumps({"records": all_records, "thresholds": thresholds}, indent=2),
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        "| threshold | in_persona flag-rate | drifting flag-rate | differential | verdict |",
        "|---|---|---|---|---|",
    ]
    for t in thresholds:
        ip = summary["flag_rates"]["in_persona"][f"{t}"]
        dr = summary["flag_rates"]["drifting"][f"{t}"]
        diff = summary["differential"][f"{t}"]
        verdict = summary["verdict"][f"{t}"]
        rows.append(f"| {t:.2f} | {ip:.2%} | {dr:.2%} | {diff:+.2%} | {verdict} |")
    (out_root / "report.md").write_text(
        "# Drift-gate calibration sweep\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    logger.info("calibration artefacts written to {}", out_root)

    per_persona = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        per_persona[rec["persona_id"]][rec["condition"]].append(rec)
    (out_root / "per_persona.json").write_text(
        json.dumps(
            {
                pid: {cond: len(records) for cond, records in conds.items()}
                for pid, conds in per_persona.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
