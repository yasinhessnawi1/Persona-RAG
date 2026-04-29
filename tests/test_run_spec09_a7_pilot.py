"""Smoke test for the A7 pilot driver.

The script's value-add over `ProbeRunner` is its on-disk artefact layout
(transcripts/, injection_logs.json, scoring_template.csv, run_config.json,
report.md). This test exercises the artefact writers with a fake responder so
we can run it on the dev Mac without loading Gemma.

It does NOT test the ``main()`` entry point end-to-end (that requires a real
Gemma backend). It tests the helper functions that produce the artefacts so a
regression in the layout would surface locally.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from persona_rag.benchmarks.base import (
    BenchmarkConversation,
    DriftProbe,
)
from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.probe_runner import ProbeInjectionLog

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_spec09_a7_pilot.py"


@pytest.fixture(scope="module")
def script_module() -> Any:
    """Import the script as a module so we can call its private helpers."""
    spec = importlib.util.spec_from_file_location("run_spec09_a7_pilot", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_spec09_a7_pilot", module)
    spec.loader.exec_module(module)
    return module


def _make_conv(
    persona_id: str, probe_id: str, probe_type: str = "self_fact_challenge"
) -> BenchmarkConversation:
    probe = DriftProbe(
        probe_id=probe_id,
        probe_type=probe_type,  # type: ignore[arg-type]
        probe_turn_index=1,
        injected_chunk_id="some_chunk" if probe_type == "counterfactual" else None,
        targets=f"{persona_id}::self_facts[0]",
    )
    return BenchmarkConversation(
        conversation_id=f"{persona_id}::{probe_id}",
        persona_id=persona_id,
        benchmark="counterfactual_probes",
        user_turns=["t0", "t1", "t2"],
        probe=probe,
        notes="test",
    )


def _make_transcript(conv: BenchmarkConversation) -> EvalConversation:
    runner_id = f"{conv.persona_id}::B2::{conv.conversation_id}"
    return EvalConversation(
        conversation_id=runner_id,
        mechanism="B2",
        persona_id=conv.persona_id,
        turns=tuple(
            ScoredTurn(turn_index=ix, user_text=t, assistant_text=f"asst {ix}")
            for ix, t in enumerate(conv.user_turns)
        ),
        per_turn_metadata=tuple(
            {"is_probe_turn": ix == 1, "probe_type": conv.probe.probe_type if conv.probe else None}
            for ix in range(len(conv.user_turns))
        ),
    )


def test_write_transcript_uses_safe_filename(script_module: Any, tmp_path: Path) -> None:
    """Filename must be ``::``-free so it stays cross-platform safe."""
    conv = _make_conv("cs_tutor", "probe_a_01")
    transcript = _make_transcript(conv)
    path = script_module._write_transcript(tmp_path, transcript)
    assert "::" not in path.name
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mechanism"] == "B2"
    assert payload["persona_id"] == "cs_tutor"
    assert len(payload["turns"]) == 3
    assert payload["per_turn_metadata"][1]["is_probe_turn"] is True


def test_write_scoring_template_one_row_per_conversation(
    script_module: Any, tmp_path: Path
) -> None:
    convs = [
        _make_conv("cs_tutor", "probe_a_01"),
        _make_conv("cs_tutor", "probe_b_01", probe_type="counterfactual"),
        _make_conv("historian", "probe_c_01", probe_type="constraint_bait"),
    ]
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript_paths = {}
    for conv in convs:
        t = _make_transcript(conv)
        transcript_paths[conv.conversation_id] = script_module._write_transcript(transcripts_dir, t)

    out = tmp_path / "scoring_template.csv"
    script_module._write_scoring_template(out, convs, transcript_paths)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["probe_id"] == "probe_a_01"
    assert rows[0]["persona_id"] == "cs_tutor"
    assert rows[0]["probe_type"] == "self_fact_challenge"
    # transcript_path is relative, points at an existing file.
    full = (out.parent / rows[0]["transcript_path"]).resolve()
    assert full.exists()
    # fail_pass and notes columns are blank (author fills them).
    assert rows[0]["fail_pass"] == ""
    assert rows[0]["notes"] == ""


def test_write_report_includes_persona_and_injection_summary(
    script_module: Any, tmp_path: Path
) -> None:
    convs = [
        _make_conv("cs_tutor", "probe_a_01"),
        _make_conv("cs_tutor", "probe_b_01", probe_type="counterfactual"),
    ]
    injection_logs = [
        ProbeInjectionLog(
            conversation_id="cs_tutor::B2::cs_tutor::probe_b_01",
            probe_id="probe_b_01",
            probe_type="counterfactual",
            probe_turn_index=1,
            injected_chunk_id="some_chunk",
            injected_chunk_in_topk=True,
            injected_chunk_rank=2,
        ),
    ]
    out = tmp_path / "report.md"
    script_module._write_report(
        out,
        run_id="20260429_222200",
        conversations=convs,
        injection_logs=injection_logs,
        transcript_paths={},
    )
    body = out.read_text(encoding="utf-8")
    assert "Spec-09 A7 calibration pilot" in body
    assert "cs_tutor" in body
    assert "probe_b_01" in body
    assert "[30%, 70%]" in body  # gate spelled out for the author
    # per-persona breakdown shows probe-type counts.
    assert "self_fact_challenge=1" in body
    assert "counterfactual=1" in body


def test_git_rev_returns_string(script_module: Any) -> None:
    rev = script_module._git_rev()
    assert isinstance(rev, str)
    assert len(rev) > 0  # either real SHA or "unknown"
