"""Tests for EvaluationRunner + transcripts + human-validation pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from persona_rag.evaluation.human_validation import (
    HUMAN_RUBRIC_DIMENSIONS,
    HumanScoreRow,
    alpha_against_panel,
    export_csv,
    load_human_csv,
    stratified_sample,
)
from persona_rag.evaluation.metrics import EvalConversation, MetricResult, ScoredTurn
from persona_rag.evaluation.poll_panel import (
    JudgeCheckpoint,
    PerJudgeConversationScore,
)
from persona_rag.evaluation.rubrics import PersonaAdherenceScore, TaskQualityScore
from persona_rag.evaluation.runner import (
    AGGREGATE_HEADER,
    LONG_FORM_HEADER,
    EvaluationRunner,
    MechanismCell,
)
from persona_rag.evaluation.transcripts import (
    conversation_yaml_to_eval,
    load_baseline_response_dir,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


# ----- transcripts loader -----


def test_load_baseline_response_dir(tmp_path: Path) -> None:
    """Per-query response JSONs become 1-turn EvalConversations."""
    for i in range(3):
        path = tmp_path / f"response_{i:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "query": f"q{i}",
                    "text": f"a{i}",
                    "bucket": "knowledge_grounded",
                    "seed": 42,
                }
            )
        )
    convs = load_baseline_response_dir(tmp_path, mechanism="b1", persona_id="cs_tutor")
    assert len(convs) == 3
    assert convs[0].mechanism == "b1"
    assert convs[0].persona_id == "cs_tutor"
    assert convs[0].turns[0].user_text == "q0"
    assert convs[0].turns[0].assistant_text == "a0"
    assert convs[0].per_turn_metadata[0]["bucket"] == "knowledge_grounded"


def test_load_baseline_skips_missing_text_or_query(tmp_path: Path) -> None:
    (tmp_path / "response_00.json").write_text(json.dumps({"query": "q0", "text": "a0"}))
    (tmp_path / "response_01.json").write_text(json.dumps({"query": "", "text": "a1"}))
    convs = load_baseline_response_dir(tmp_path, mechanism="b1", persona_id="cs_tutor")
    assert len(convs) == 1


def test_conversation_yaml_to_eval(tmp_path: Path) -> None:
    """A drift-trajectory YAML becomes a multi-turn EvalConversation."""
    yaml_text = """\
persona_id: cs_tutor
condition: in_persona
n_pairs: 2
turns:
  - role: user
    text: hi
  - role: assistant
    text: hello
  - role: user
    text: bye
  - role: assistant
    text: see you
"""
    path = tmp_path / "conv.yaml"
    path.write_text(yaml_text)
    conv = conversation_yaml_to_eval(path, mechanism="b1")
    assert len(conv.turns) == 2
    assert conv.turns[0].user_text == "hi"
    assert conv.turns[0].assistant_text == "hello"
    assert conv.turns[1].user_text == "bye"
    assert conv.persona_id == "cs_tutor"
    assert conv.mechanism == "b1"


# ----- EvaluationRunner -----


class _ConstMetric:
    """Metric returning a fixed per-conversation list and aggregate."""

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value

    def score(
        self,
        conversations: list[EvalConversation],
        persona: Persona,
    ) -> MetricResult:
        per = [self._value for _ in conversations]
        return MetricResult(
            name=self.name,
            value=self._value,
            per_conversation=per,
            per_conversation_ids=[c.conversation_id for c in conversations],
            per_persona={persona.persona_id or "<unknown>": self._value},
            metadata={},
        )


def _make_cell(mechanism: str, persona: Persona, n: int = 2) -> MechanismCell:
    convs = [
        EvalConversation(
            conversation_id=f"{mechanism}_c{i}",
            mechanism=mechanism,
            persona_id=persona.persona_id or "<unknown>",
            turns=(ScoredTurn(turn_index=0, user_text=f"q{i}", assistant_text=f"a{i}"),),
        )
        for i in range(n)
    ]
    return MechanismCell(
        mechanism=mechanism,
        model="gemma2-9b-it",
        benchmark="test_bench",
        persona=persona,
        conversations=convs,
        seed=42,
    )


def test_runner_writes_long_and_aggregate_csvs(cs_tutor: Persona, tmp_path: Path) -> None:
    runner = EvaluationRunner(
        output_dir=tmp_path / "run",
        metrics=[_ConstMetric("m1", 0.9), _ConstMetric("m2", 0.6)],
        run_id="20260429_120000",
    )
    cells = [_make_cell("b1", cs_tutor), _make_cell("m3", cs_tutor)]
    runner.run(cells)

    long_path = tmp_path / "run" / "results.csv"
    agg_path = tmp_path / "run" / "results_aggregate.csv"
    assert long_path.exists()
    assert agg_path.exists()

    with long_path.open() as f:
        rows = list(csv.DictReader(f))
    # 2 mechanisms x 2 metrics x 2 conversations = 8 rows
    assert len(rows) == 8
    assert set(rows[0].keys()) == set(LONG_FORM_HEADER)
    assert all(r["run_id"] == "20260429_120000" for r in rows)
    # Distinct mechanisms present.
    assert {r["mechanism"] for r in rows} == {"b1", "m3"}

    with agg_path.open() as f:
        agg_rows = list(csv.DictReader(f))
    # 2 mechanisms x 2 metrics = 4 rows
    assert len(agg_rows) == 4
    assert set(agg_rows[0].keys()) == set(AGGREGATE_HEADER)


def test_runner_writes_run_config_and_json_bundle(cs_tutor: Persona, tmp_path: Path) -> None:
    runner = EvaluationRunner(
        output_dir=tmp_path / "rcfg",
        metrics=[_ConstMetric("m1", 0.5)],
        run_id="rid",
    )
    runner.run([_make_cell("b1", cs_tutor)])
    cfg = json.loads((tmp_path / "rcfg" / "run_config.json").read_text())
    assert cfg["run_id"] == "rid"
    assert cfg["metrics"] == ["m1"]
    bundle = json.loads((tmp_path / "rcfg" / "results.json").read_text())
    assert len(bundle["cells"]) == 1


def test_runner_rejects_no_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        EvaluationRunner(output_dir=tmp_path, metrics=[])


def test_runner_reproducibility_two_runs_same_seed(cs_tutor: Persona, tmp_path: Path) -> None:
    """Same metrics + same conversations + same seed -> identical CSV outputs."""
    cell = _make_cell("b1", cs_tutor)
    runner_a = EvaluationRunner(
        output_dir=tmp_path / "a",
        metrics=[_ConstMetric("m", 0.7)],
        run_id="rid",
        seed=42,
    )
    runner_b = EvaluationRunner(
        output_dir=tmp_path / "b",
        metrics=[_ConstMetric("m", 0.7)],
        run_id="rid",
        seed=42,
    )
    runner_a.run([cell])
    runner_b.run([cell])

    a_text = (tmp_path / "a" / "results.csv").read_text()
    b_text = (tmp_path / "b" / "results.csv").read_text()
    assert a_text == b_text


# ----- Human-validation pipeline -----


def _conv_for_human(mechanism: str, persona_id: str, conv_id: str) -> EvalConversation:
    return EvalConversation(
        conversation_id=conv_id,
        mechanism=mechanism,
        persona_id=persona_id,
        turns=(ScoredTurn(turn_index=0, user_text="q", assistant_text=f"a-{conv_id}"),),
    )


def test_stratified_sample_picks_balanced_per_mechanism() -> None:
    convs = {
        mech: [_conv_for_human(mech, "cs_tutor", f"{mech}_{i}") for i in range(10)]
        for mech in ("b1", "b2", "m1", "m3")
    }
    items = stratified_sample(convs, per_mechanism=5, seed=42)
    assert len(items) == 20  # 4 mechanisms x 5 each
    counts: dict[str, int] = {}
    for item in items:
        counts[item.mechanism] = counts.get(item.mechanism, 0) + 1
    assert counts == {"b1": 5, "b2": 5, "m1": 5, "m3": 5}
    # Round-robin: first 4 items span the mechanisms.
    first_four = {items[i].mechanism for i in range(4)}
    assert first_four == {"b1", "b2", "m1", "m3"}


def test_export_and_load_round_trip(tmp_path: Path) -> None:
    convs = {
        "b1": [_conv_for_human("b1", "cs_tutor", f"b1_{i}") for i in range(3)],
        "m3": [_conv_for_human("m3", "cs_tutor", f"m3_{i}") for i in range(3)],
    }
    items = stratified_sample(convs, per_mechanism=2, seed=7)
    csv_path = tmp_path / "export.csv"
    map_path = tmp_path / "mapping.json"
    export_csv(items, csv_path, map_path)

    # Simulate the human filling in the CSV.
    rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            for dim in HUMAN_RUBRIC_DIMENSIONS:
                r[f"score_{dim}"] = "4"
            rows.append(r)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_human_csv(csv_path, map_path)
    assert len(loaded) == len(items)
    # Mechanism rejoined from mapping.
    assert {row.mechanism for row in loaded} == {"b1", "m3"}
    assert all(row.scores["self_facts"] == 4 for row in loaded)


def test_alpha_against_panel_runs_with_two_judges() -> None:
    """Smoke: alpha computation returns finite numbers when raters agree."""
    human_rows = [
        HumanScoreRow(
            row_id=i,
            blinded_token=f"t{i}",
            mechanism="b1",
            persona_id="cs_tutor",
            conversation_id=f"c{i}",
            scores={dim: 4 for dim in HUMAN_RUBRIC_DIMENSIONS},
        )
        for i in range(5)
    ]
    panel = {
        "j1": JudgeCheckpoint(
            judge_name="j1",
            rubric_format="json",
            persona_id="cs_tutor",
            scores=[
                PerJudgeConversationScore(
                    conversation_id=f"c{i}",
                    persona_adherence=PersonaAdherenceScore(
                        self_facts=4, worldview=4, constraints=4, overall=4
                    ),
                    task_quality=TaskQualityScore(score=4),
                )
                for i in range(5)
            ],
        ),
        "j2": JudgeCheckpoint(
            judge_name="j2",
            rubric_format="json",
            persona_id="cs_tutor",
            scores=[
                PerJudgeConversationScore(
                    conversation_id=f"c{i}",
                    persona_adherence=PersonaAdherenceScore(
                        self_facts=4, worldview=4, constraints=4, overall=4
                    ),
                    task_quality=TaskQualityScore(score=4),
                )
                for i in range(5)
            ],
        ),
    }
    alphas = alpha_against_panel(human_rows, panel)
    assert "vs_j1" in alphas
    assert "vs_j2" in alphas
    assert "vs_panel" in alphas
