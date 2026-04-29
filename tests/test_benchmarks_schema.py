"""Schema tests for the benchmark base types.

Pure-pydantic tests; no external IO. Cover:

- ``DriftProbe`` invariants: counterfactual must carry an
  ``injected_chunk_id``; non-counterfactual must not.
- ``BenchmarkConversation``: probe_turn_index must be in range; user_turns
  must be non-empty.
- ``CounterfactualChunk``: required fields parse.
- Round-trip: ``BenchmarkConversation.to_yaml`` then re-load yields equal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from persona_rag.benchmarks.base import (
    BenchmarkConversation,
    CounterfactualChunk,
    DriftProbe,
)
from persona_rag.benchmarks.counterfactual_probes import (
    load_counterfactual_chunk,
    load_probe_yaml,
)


def test_drift_probe_counterfactual_requires_chunk_id() -> None:
    with pytest.raises(ValueError, match="counterfactual probes require"):
        DriftProbe(probe_id="p1", probe_type="counterfactual", probe_turn_index=0)


def test_drift_probe_non_counterfactual_rejects_chunk_id() -> None:
    with pytest.raises(ValueError, match="injected_chunk_id is only valid"):
        DriftProbe(
            probe_id="p1",
            probe_type="self_fact_challenge",
            probe_turn_index=0,
            injected_chunk_id="some_chunk",
        )


def test_drift_probe_constraint_bait_no_chunk() -> None:
    probe = DriftProbe(
        probe_id="c1",
        probe_type="constraint_bait",
        probe_turn_index=2,
    )
    assert probe.injected_chunk_id is None


def test_benchmark_conversation_probe_turn_in_bounds() -> None:
    probe = DriftProbe(probe_id="x", probe_type="constraint_bait", probe_turn_index=5)
    with pytest.raises(ValueError, match="out of range"):
        BenchmarkConversation(
            conversation_id="c1",
            persona_id="p",
            benchmark="x",
            user_turns=["only one turn"],
            probe=probe,
        )


def test_benchmark_conversation_rejects_empty_user_turn() -> None:
    with pytest.raises(ValueError, match="is empty"):
        BenchmarkConversation(
            conversation_id="c1",
            persona_id="p",
            benchmark="x",
            user_turns=["good turn", "   "],
        )


def test_benchmark_conversation_no_probe_is_valid() -> None:
    conv = BenchmarkConversation(
        conversation_id="c1",
        persona_id="p",
        benchmark="personachat",
        user_turns=["hello"],
    )
    assert conv.probe is None


def test_counterfactual_chunk_required_fields() -> None:
    chunk = CounterfactualChunk(
        chunk_id="x",
        persona_id="p",
        contradicts="p::worldview[0]",
        text="some counter-evidence",
    )
    assert chunk.source_label == "counter_evidence"


def test_conversation_yaml_round_trip(tmp_path: Path) -> None:
    probe = DriftProbe(
        probe_id="r",
        probe_type="counterfactual",
        probe_turn_index=1,
        injected_chunk_id="some_chunk",
    )
    conv = BenchmarkConversation(
        conversation_id="c::r",
        persona_id="p",
        benchmark="counterfactual_probes",
        user_turns=["t1", "t2", "t3"],
        probe=probe,
        notes="round-trip",
    )
    out = tmp_path / "conv.yaml"
    conv.to_yaml(out)
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    rebuilt = BenchmarkConversation.model_validate(raw)
    assert rebuilt == conv


def test_load_counterfactual_chunk_parses_frontmatter(tmp_path: Path) -> None:
    body = (
        "---\n"
        "chunk_id: t1\n"
        "persona_id: p\n"
        "contradicts: p::w[0]\n"
        "source_label: test_source\n"
        "---\n"
        "Body of the chunk.\n"
    )
    path = tmp_path / "c.md"
    path.write_text(body, encoding="utf-8")
    chunk = load_counterfactual_chunk(path)
    assert chunk.chunk_id == "t1"
    assert chunk.text == "Body of the chunk."
    assert chunk.source_label == "test_source"


def test_load_counterfactual_chunk_rejects_missing_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        load_counterfactual_chunk(path)


def test_load_probe_yaml_round_trip(tmp_path: Path) -> None:
    raw = {
        "persona_id": "p",
        "probe_id": "x",
        "probe_type": "self_fact_challenge",
        "probe_turn_index": 1,
        "user_turns": ["a", "b"],
        "notes": "sanity",
    }
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    conv = load_probe_yaml(path)
    assert conv.persona_id == "p"
    assert conv.probe is not None
    assert conv.probe.probe_type == "self_fact_challenge"
    assert conv.user_turns == ["a", "b"]
