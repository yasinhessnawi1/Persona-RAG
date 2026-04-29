"""Tests for the PoLL panel + rubric parsers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.evaluation.poll_panel import (
    JudgeCheckpoint,
    JudgeSpec,
    PerJudgeConversationScore,
    PoLLPanel,
    load_checkpoints_from_dir,
    reliability_matrix_from_checkpoints,
)
from persona_rag.evaluation.rubrics import (
    PersonaAdherenceScore,
    TaskQualityScore,
    parse_persona_adherence_json,
    parse_persona_adherence_native_prometheus,
    parse_task_quality_json,
    parse_task_quality_native_prometheus,
    render_persona_adherence_json_prompt,
    render_persona_adherence_native_prometheus_prompt,
    render_task_quality_json_prompt,
    render_task_quality_native_prometheus_prompt,
)
from persona_rag.schema.persona import Persona

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


def _conv(persona_id: str, conv_id: str, n_turns: int = 3) -> EvalConversation:
    turns = tuple(
        ScoredTurn(
            turn_index=i,
            user_text=f"user q{i} ({conv_id})",
            assistant_text=f"assistant a{i} ({conv_id})",
        )
        for i in range(n_turns)
    )
    return EvalConversation(
        conversation_id=conv_id,
        mechanism="test",
        persona_id=persona_id,
        turns=turns,
    )


# ----- Native Prometheus parser -----


def test_parse_persona_adherence_native_aggregates_four_calls() -> None:
    raws = {
        "self_facts": "Feedback: Great consistency. [RESULT] 5",
        "worldview": "Feedback: Mild drift on systems-design view. [RESULT] 3",
        "constraints": "Feedback: All constraints honored. [RESULT] 5",
        "overall": "Feedback: Strong persona voice. [RESULT] 4",
    }
    score = parse_persona_adherence_native_prometheus(raws)
    assert score.self_facts == 5
    assert score.worldview == 3
    assert score.constraints == 5
    assert score.overall == 4
    assert score.malformed is False
    assert score.overall_mean == pytest.approx(4.25)


def test_parse_persona_adherence_native_handles_missing_dimension() -> None:
    """If one dimension's response is malformed, that dim falls back to 3 and malformed=True."""
    raws = {
        "self_facts": "Feedback: Good. [RESULT] 4",
        "worldview": "garbled output with no result marker",
        "constraints": "Feedback: Fine. [RESULT] 5",
        "overall": "Feedback: Great. [RESULT] 5",
    }
    score = parse_persona_adherence_native_prometheus(raws)
    assert score.malformed is True
    assert score.worldview == 3


def test_parse_task_quality_native() -> None:
    score = parse_task_quality_native_prometheus("Feedback: Correct. [RESULT] 4")
    assert score.score == 4
    assert score.malformed is False


def test_parse_task_quality_native_malformed() -> None:
    score = parse_task_quality_native_prometheus("nonsense")
    assert score.malformed is True
    assert score.score == 3  # fallback


# ----- JSON parsers -----


def test_parse_persona_adherence_json_clean() -> None:
    raw = '{"self_facts": 5, "worldview": 4, "constraints": 5, "overall": 5, "reasoning": "ok"}'
    score = parse_persona_adherence_json(raw)
    assert score.self_facts == 5
    assert score.malformed is False


def test_parse_persona_adherence_json_with_preamble_and_fences() -> None:
    raw = (
        "Sure. Here is the JSON:\n```json\n"
        '{"self_facts": 4, "worldview": 4, "constraints": 5, "overall": 4, "reasoning": "..."}'
        "\n```\nLet me know."
    )
    score = parse_persona_adherence_json(raw)
    assert score.self_facts == 4
    assert score.malformed is False


def test_parse_persona_adherence_json_missing_dim_falls_back() -> None:
    raw = '{"self_facts": 4, "worldview": 3, "overall": 4}'  # constraints missing
    score = parse_persona_adherence_json(raw)
    assert score.constraints == 3
    assert score.malformed is True


def test_parse_persona_adherence_json_clamps_out_of_range() -> None:
    raw = '{"self_facts": 7, "worldview": 0, "constraints": 5, "overall": 4}'
    score = parse_persona_adherence_json(raw)
    assert score.self_facts == 5
    assert score.worldview == 1


def test_parse_task_quality_json() -> None:
    score = parse_task_quality_json('{"score": 4, "reasoning": "ok"}')
    assert score.score == 4
    assert score.malformed is False


def test_parse_task_quality_json_malformed() -> None:
    score = parse_task_quality_json("not even json")
    assert score.malformed is True


# ----- Prompt renderers smoke-test -----


def test_render_persona_adherence_native_prompt_contains_required_sections(
    cs_tutor: Persona,
) -> None:
    prompt = render_persona_adherence_native_prometheus_prompt(
        persona=cs_tutor,
        conversation_turns=[("hi", "hello")],
        dimension="self_facts",
    )
    assert "###Score Rubrics:" in prompt
    assert "[self_facts]" in prompt
    assert "[RESULT]" in prompt
    assert cs_tutor.identity.name in prompt


def test_render_persona_adherence_native_prompt_rejects_unknown_dimension(
    cs_tutor: Persona,
) -> None:
    with pytest.raises(ValueError, match="dimension"):
        render_persona_adherence_native_prometheus_prompt(
            persona=cs_tutor,
            conversation_turns=[("hi", "hello")],
            dimension="not_a_dim",
        )


def test_render_persona_adherence_json_prompt_asks_for_json(cs_tutor: Persona) -> None:
    prompt = render_persona_adherence_json_prompt(
        persona=cs_tutor,
        conversation_turns=[("hi", "hello")],
    )
    assert "JSON" in prompt
    assert '"self_facts"' in prompt
    assert cs_tutor.identity.name in prompt


def test_render_task_quality_prompts(cs_tutor: Persona) -> None:
    native = render_task_quality_native_prometheus_prompt(
        persona=cs_tutor, conversation_turns=[("hi", "hello")]
    )
    json_prompt = render_task_quality_json_prompt(
        persona=cs_tutor, conversation_turns=[("hi", "hello")]
    )
    assert "[task_quality]" in native
    assert '"score"' in json_prompt


# ----- PoLL panel sequential loading + checkpointing -----


class _CannedJudge:
    """Returns a pre-set raw response per generate() call."""

    name = "canned-judge"
    model_id = "fake/canned"
    num_layers = 0
    hidden_dim = 0

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        self.call_count += 1
        return self.response


def _native_response_for_dim(score: int) -> str:
    return f"Feedback: pinned for test [RESULT] {score}"


def _json_response(
    self_facts: int, worldview: int, constraints: int, overall: int, task: int
) -> str:
    """Return a string containing the persona JSON; the task call uses a separate response."""
    return (
        f'{{"self_facts": {self_facts}, "worldview": {worldview}, '
        f'"constraints": {constraints}, "overall": {overall}, '
        '"reasoning": "ok"}'
    )


def test_panel_runs_two_judges_sequentially_and_aggregates(
    cs_tutor: Persona, tmp_path: Path
) -> None:
    """Two-judge panel: judge A scores 5 across the board, judge B scores 3. Mean = 4."""
    # Judge A: native Prometheus, always [RESULT] 5
    judge_a = _CannedJudge("Feedback: perfect [RESULT] 5")
    judge_b_response = (
        '{"self_facts": 3, "worldview": 3, "constraints": 3, "overall": 3, "reasoning": "ok"}'
    )
    judge_b = _CannedJudge(judge_b_response)

    spec_a = JudgeSpec(
        name="judge_a_native", builder=lambda: judge_a, rubric_format="native_prometheus"
    )
    spec_b = JudgeSpec(name="judge_b_json", builder=lambda: judge_b, rubric_format="json")

    panel = PoLLPanel(judges=[spec_a, spec_b], output_dir=tmp_path / "poll")
    convs = [_conv("cs_tutor", f"c{i}") for i in range(2)]
    results = panel.run(cs_tutor, convs)

    persona_result = results["poll_persona_adherence"]
    task_result = results["poll_task_quality"]

    # Per conversation: mean of (5.0 across A's 4 dims) and (3.0 across B's 4 dims) = 4.0
    assert persona_result.value == pytest.approx(4.0)
    # Task quality: A returns score 5, B returns... actually B's task call returns the same JSON
    # which has no "score" field -> falls back to 3 (malformed).
    # Average of 5 and 3 = 4.
    assert task_result.value == pytest.approx(4.0)

    assert sorted(persona_result.metadata["judge_names"]) == ["judge_a_native", "judge_b_json"]


def test_panel_per_judge_checkpoint_files_written(cs_tutor: Persona, tmp_path: Path) -> None:
    spec = JudgeSpec(
        name="ck_test",
        builder=lambda: _CannedJudge("Feedback: ok [RESULT] 4"),
        rubric_format="native_prometheus",
    )
    panel = PoLLPanel(judges=[spec], output_dir=tmp_path / "p")
    convs = [_conv("cs_tutor", "c0")]
    panel.run(cs_tutor, convs)
    ckpt_path = tmp_path / "p" / "judge_ck_test.json"
    assert ckpt_path.exists()
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert payload["judge_name"] == "ck_test"
    assert len(payload["scores"]) == 1


def test_panel_skips_already_checkpointed_judges(cs_tutor: Persona, tmp_path: Path) -> None:
    """If a checkpoint exists, the builder is not invoked again."""
    spec = JudgeSpec(
        name="skipme",
        builder=lambda: _CannedJudge("Feedback: ok [RESULT] 4"),
        rubric_format="native_prometheus",
    )
    panel = PoLLPanel(judges=[spec], output_dir=tmp_path / "q")
    convs = [_conv("cs_tutor", "c0")]
    panel.run(cs_tutor, convs)

    # Replace builder with one that throws.
    def bad_builder() -> _CannedJudge:
        raise RuntimeError("builder should not be called when checkpoint exists")

    spec_again = JudgeSpec(
        name="skipme",
        builder=bad_builder,
        rubric_format="native_prometheus",
    )
    panel_again = PoLLPanel(judges=[spec_again], output_dir=tmp_path / "q")
    panel_again.run(cs_tutor, convs)  # no exception


def test_panel_rejects_duplicate_judge_names() -> None:
    spec = JudgeSpec(name="dup", builder=lambda: _CannedJudge("ok"), rubric_format="json")
    with pytest.raises(ValueError, match="duplicate judge name"):
        PoLLPanel(judges=[spec, spec], output_dir=Path("/tmp"))


def test_load_checkpoints_from_dir(cs_tutor: Persona, tmp_path: Path) -> None:
    spec_a = JudgeSpec(
        name="a",
        builder=lambda: _CannedJudge("Feedback: [RESULT] 5"),
        rubric_format="native_prometheus",
    )
    spec_b = JudgeSpec(
        name="b",
        builder=lambda: _CannedJudge(
            '{"self_facts": 3, "worldview": 3, "constraints": 3, "overall": 3}'
        ),
        rubric_format="json",
    )
    panel = PoLLPanel(judges=[spec_a, spec_b], output_dir=tmp_path / "x")
    convs = [_conv("cs_tutor", "c0")]
    panel.run(cs_tutor, convs)

    loaded = load_checkpoints_from_dir(tmp_path / "x")
    assert set(loaded.keys()) == {"a", "b"}


def test_reliability_matrix_shape() -> None:
    ckpts = {
        "j1": JudgeCheckpoint(
            judge_name="j1",
            rubric_format="native_prometheus",
            persona_id="cs",
            scores=[
                PerJudgeConversationScore(
                    conversation_id="c0",
                    persona_adherence=PersonaAdherenceScore(
                        self_facts=4, worldview=4, constraints=5, overall=4
                    ),
                    task_quality=TaskQualityScore(score=4),
                ),
                PerJudgeConversationScore(
                    conversation_id="c1",
                    persona_adherence=PersonaAdherenceScore(
                        self_facts=3, worldview=3, constraints=3, overall=3
                    ),
                    task_quality=TaskQualityScore(score=3),
                ),
            ],
        ),
        "j2": JudgeCheckpoint(
            judge_name="j2",
            rubric_format="json",
            persona_id="cs",
            scores=[
                PerJudgeConversationScore(
                    conversation_id="c0",
                    persona_adherence=PersonaAdherenceScore(
                        self_facts=5, worldview=4, constraints=5, overall=4
                    ),
                    task_quality=TaskQualityScore(score=5),
                ),
                # Missing c1 -- judge B failed to score it.
            ],
        ),
    }
    matrix = reliability_matrix_from_checkpoints(ckpts, rubric="persona_adherence")
    assert len(matrix) == 2  # n_judges
    assert len(matrix[0]) == 2  # n_items (c0, c1)
    # Judge j2 missing c1 -> nan.
    import math

    assert math.isnan(matrix[1][1])
