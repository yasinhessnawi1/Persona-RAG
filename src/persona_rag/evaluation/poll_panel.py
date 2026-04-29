"""Three-judge self-hosted PoLL panel for persona-adherence + task-quality scoring.

Sequential loading pattern (V100 can hold one 7-9B 4-bit model at a
time alongside cached evaluation data):

1. Build judge A (e.g. Prometheus-2-7B).
2. Score every conversation in the work-list against both rubrics.
3. Checkpoint judge A's scores to disk.
4. Free judge A.
5. Build judge B (e.g. Qwen2.5-7B-Instruct).
6. ... repeat for judges B and C.
7. Aggregate across judges.

The panel takes *judge builders* (zero-arg factories), not pre-loaded
backends. The builder is invoked when it's that judge's turn; the
returned backend is freed (Python ``del`` + ``torch.cuda.empty_cache``)
between judges.

Per-judge checkpoints live under ``<run_dir>/judge_<judge_name>/``:

- ``persona_adherence.json`` -- list of one entry per (conversation_id,
  swap_id) -- each entry has the parsed ``PersonaAdherenceScore`` plus the
  raw response.
- ``task_quality.json`` -- same shape, single-dimension score.

The aggregator skips judges whose checkpoints already exist on disk --
re-runs of a partially-failed panel resume cleanly.

Format choice per judge:

- Prometheus-2 -> native Prometheus rubric (one call per dimension for
  persona-adherence, one call total for task-quality).
- Qwen / Llama / others -> JSON rubric (single call per rubric).

Position swap is intentionally NOT applied. The rubrics are
single-response direct assessment -- no second candidate to swap with.
Position bias is the failure mode of pairwise evaluation, not direct
assessment. (See the spec-08 research note for the reasoning.)
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from persona_rag.evaluation.metrics import EvalConversation, MetricResult
from persona_rag.evaluation.rubrics import (
    PERSONA_ADHERENCE_DIMENSIONS,
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
from persona_rag.models.base import LLMBackend
from persona_rag.schema.persona import Persona

JudgeFormat = Literal["native_prometheus", "json"]


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """One judge in the panel.

    ``builder`` is invoked when it's this judge's turn; the returned
    backend is freed afterwards. Keeping it a closure means the panel
    never holds more than one judge resident at a time.
    """

    name: str
    builder: Callable[[], LLMBackend]
    rubric_format: JudgeFormat
    max_new_tokens_persona: int = 512
    max_new_tokens_task: int = 384
    temperature: float = 0.0


class PerJudgeConversationScore(BaseModel):
    """One conversation's score under one judge."""

    model_config = ConfigDict(extra="allow")

    conversation_id: str
    persona_adherence: PersonaAdherenceScore
    task_quality: TaskQualityScore


class JudgeCheckpoint(BaseModel):
    """Persisted per-judge results, written once per judge sweep."""

    model_config = ConfigDict(extra="forbid")

    judge_name: str
    rubric_format: JudgeFormat
    persona_id: str
    scores: list[PerJudgeConversationScore] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Judge sweep
# ---------------------------------------------------------------------------


def _conv_to_pairs(conv: EvalConversation) -> list[tuple[str, str]]:
    return [(t.user_text, t.assistant_text) for t in conv.turns]


def _score_one_with_judge(
    judge: LLMBackend,
    judge_spec: JudgeSpec,
    persona: Persona,
    conv: EvalConversation,
) -> PerJudgeConversationScore:
    """Score one conversation under the currently-loaded judge."""
    pairs = _conv_to_pairs(conv)
    if judge_spec.rubric_format == "native_prometheus":
        # Persona adherence: one call per dimension.
        persona_raws: dict[str, str] = {}
        for dim in PERSONA_ADHERENCE_DIMENSIONS:
            prompt = render_persona_adherence_native_prometheus_prompt(
                persona=persona,
                conversation_turns=pairs,
                dimension=dim,
            )
            raw = judge.generate(
                prompt,
                max_new_tokens=judge_spec.max_new_tokens_persona,
                temperature=judge_spec.temperature,
            )
            persona_raws[dim] = raw
        persona_score = parse_persona_adherence_native_prometheus(persona_raws)

        task_prompt = render_task_quality_native_prometheus_prompt(
            persona=persona,
            conversation_turns=pairs,
        )
        task_raw = judge.generate(
            task_prompt,
            max_new_tokens=judge_spec.max_new_tokens_task,
            temperature=judge_spec.temperature,
        )
        task_score = parse_task_quality_native_prometheus(task_raw)
    else:
        # JSON: single call per rubric.
        persona_prompt = render_persona_adherence_json_prompt(
            persona=persona,
            conversation_turns=pairs,
        )
        persona_raw = judge.generate(
            persona_prompt,
            max_new_tokens=judge_spec.max_new_tokens_persona,
            temperature=judge_spec.temperature,
        )
        persona_score = parse_persona_adherence_json(persona_raw)

        task_prompt = render_task_quality_json_prompt(
            persona=persona,
            conversation_turns=pairs,
        )
        task_raw = judge.generate(
            task_prompt,
            max_new_tokens=judge_spec.max_new_tokens_task,
            temperature=judge_spec.temperature,
        )
        task_score = parse_task_quality_json(task_raw)

    return PerJudgeConversationScore(
        conversation_id=conv.conversation_id,
        persona_adherence=persona_score,
        task_quality=task_score,
    )


def _free_judge(judge: LLMBackend) -> None:
    """Best-effort: drop the judge backend and free GPU memory."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    del judge


def run_judge_sweep(
    judge_spec: JudgeSpec,
    persona: Persona,
    conversations: list[EvalConversation],
    checkpoint_path: Path,
    *,
    skip_if_checkpoint_exists: bool = True,
) -> JudgeCheckpoint:
    """Score every conversation under one judge and write the checkpoint.

    Returns the loaded checkpoint (re-loads from disk if it already
    exists and ``skip_if_checkpoint_exists=True``).
    """
    if skip_if_checkpoint_exists and checkpoint_path.exists():
        logger.info(
            "PoLL: judge {} checkpoint already at {} -- skipping sweep",
            judge_spec.name,
            checkpoint_path,
        )
        return JudgeCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))

    logger.info(
        "PoLL: building judge {} for {} conversations (format={})",
        judge_spec.name,
        len(conversations),
        judge_spec.rubric_format,
    )
    judge = judge_spec.builder()
    scores: list[PerJudgeConversationScore] = []
    malformed_persona = 0
    malformed_task = 0
    try:
        for i, conv in enumerate(conversations):
            try:
                score = _score_one_with_judge(judge, judge_spec, persona, conv)
            except Exception as exc:
                logger.warning(
                    "PoLL: judge {} failed on conversation {} ({}); retrying once",
                    judge_spec.name,
                    conv.conversation_id,
                    exc,
                )
                score = _score_one_with_judge(judge, judge_spec, persona, conv)
            scores.append(score)
            if score.persona_adherence.malformed:
                malformed_persona += 1
            if score.task_quality.malformed:
                malformed_task += 1
            if (i + 1) % 5 == 0:
                logger.info(
                    "PoLL: judge {} progress {}/{} conversations",
                    judge_spec.name,
                    i + 1,
                    len(conversations),
                )
    finally:
        _free_judge(judge)

    checkpoint = JudgeCheckpoint(
        judge_name=judge_spec.name,
        rubric_format=judge_spec.rubric_format,
        persona_id=persona.persona_id or "<unknown>",
        scores=scores,
        metadata={
            "malformed_persona_adherence": malformed_persona,
            "malformed_task_quality": malformed_task,
            "n_conversations": len(conversations),
        },
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        checkpoint.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("PoLL: wrote judge {} checkpoint at {}", judge_spec.name, checkpoint_path)
    return checkpoint


# ---------------------------------------------------------------------------
# Panel orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PoLLPanel:
    """Sequential 3-judge orchestrator with per-judge checkpointing.

    ``judges`` is a list of :class:`JudgeSpec`. The panel:

    1. For each judge in order: run :func:`run_judge_sweep`.
    2. After all sweeps complete: aggregate per-conversation scores
       across judges (mean across judges, per dimension).

    Returns a :class:`MetricResult` per rubric (persona-adherence headline =
    mean ``overall_mean`` across judges, then mean across conversations;
    task-quality headline = mean across judges, then mean across
    conversations).
    """

    judges: list[JudgeSpec]
    output_dir: Path
    name: str = field(default="poll_panel")

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("PoLLPanel needs at least one judge.")
        seen: set[str] = set()
        for j in self.judges:
            if j.name in seen:
                raise ValueError(f"duplicate judge name {j.name!r} in panel")
            seen.add(j.name)

    def run(
        self,
        persona: Persona,
        conversations: list[EvalConversation],
    ) -> dict[str, MetricResult]:
        """Run all judges sequentially and aggregate. Returns one MetricResult per rubric."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        per_judge: dict[str, JudgeCheckpoint] = {}
        for spec in self.judges:
            ckpt_path = self.output_dir / f"judge_{spec.name}.json"
            per_judge[spec.name] = run_judge_sweep(spec, persona, conversations, ckpt_path)
        return self._aggregate(persona, conversations, per_judge)

    @staticmethod
    def _aggregate(
        persona: Persona,
        conversations: list[EvalConversation],
        per_judge: dict[str, JudgeCheckpoint],
    ) -> dict[str, MetricResult]:
        """Mean-across-judges per conversation, then mean across conversations."""
        # Index by (judge_name, conversation_id).
        per_judge_lookup: dict[str, dict[str, PerJudgeConversationScore]] = {
            name: {s.conversation_id: s for s in ckpt.scores} for name, ckpt in per_judge.items()
        }

        persona_per_conv: list[float] = []
        task_per_conv: list[float] = []
        per_conv_ids: list[str] = []

        # Per-judge aggregate distributions for divergence reporting.
        per_judge_persona_aggregate: dict[str, float] = {}
        per_judge_task_aggregate: dict[str, float] = {}

        # Build per-judge per-conversation lists for the aggregate calc.
        per_judge_persona_lists: dict[str, list[float]] = {n: [] for n in per_judge_lookup}
        per_judge_task_lists: dict[str, list[float]] = {n: [] for n in per_judge_lookup}

        for conv in conversations:
            persona_means: list[float] = []
            task_means: list[float] = []
            for judge_name, lookup in per_judge_lookup.items():
                score = lookup.get(conv.conversation_id)
                if score is None:
                    continue  # Judge missed this conversation.
                persona_means.append(score.persona_adherence.overall_mean)
                task_means.append(float(score.task_quality.score))
                per_judge_persona_lists[judge_name].append(score.persona_adherence.overall_mean)
                per_judge_task_lists[judge_name].append(float(score.task_quality.score))
            persona_per_conv.append(statistics.fmean(persona_means) if persona_means else 3.0)
            task_per_conv.append(statistics.fmean(task_means) if task_means else 3.0)
            per_conv_ids.append(conv.conversation_id)

        for judge_name, lst in per_judge_persona_lists.items():
            per_judge_persona_aggregate[judge_name] = statistics.fmean(lst) if lst else float("nan")
        for judge_name, lst in per_judge_task_lists.items():
            per_judge_task_aggregate[judge_name] = statistics.fmean(lst) if lst else float("nan")

        persona_value = statistics.fmean(persona_per_conv) if persona_per_conv else 3.0
        task_value = statistics.fmean(task_per_conv) if task_per_conv else 3.0
        persona_id = persona.persona_id or "<unknown>"

        meta_common: dict[str, Any] = {
            "n_judges": len(per_judge_lookup),
            "judge_names": list(per_judge_lookup.keys()),
            "n_conversations": len(conversations),
        }

        persona_meta: dict[str, Any] = {
            **meta_common,
            "per_judge_aggregate": per_judge_persona_aggregate,
            "malformed_per_judge": {
                n: ckpt.metadata.get("malformed_persona_adherence", 0)
                for n, ckpt in per_judge.items()
            },
        }
        task_meta: dict[str, Any] = {
            **meta_common,
            "per_judge_aggregate": per_judge_task_aggregate,
            "malformed_per_judge": {
                n: ckpt.metadata.get("malformed_task_quality", 0) for n, ckpt in per_judge.items()
            },
        }

        return {
            "poll_persona_adherence": MetricResult(
                name="poll_persona_adherence",
                value=persona_value,
                per_conversation=persona_per_conv,
                per_conversation_ids=per_conv_ids,
                per_persona={persona_id: persona_value},
                metadata=persona_meta,
            ),
            "poll_task_quality": MetricResult(
                name="poll_task_quality",
                value=task_value,
                per_conversation=task_per_conv,
                per_conversation_ids=per_conv_ids,
                per_persona={persona_id: task_value},
                metadata=task_meta,
            ),
        }


# ---------------------------------------------------------------------------
# Krippendorff's alpha utilities
# ---------------------------------------------------------------------------


def reliability_matrix_from_checkpoints(
    per_judge: dict[str, JudgeCheckpoint],
    *,
    rubric: Literal["persona_adherence", "task_quality"],
) -> list[list[float]]:
    """Build the (n_judges, n_items) reliability matrix expected by `krippendorff.alpha`.

    Items are conversations. Order is the union of conversation ids
    across judges, sorted for determinism. Missing scores are
    represented by ``float('nan')``.

    For the persona-adherence rubric the per-judge per-conversation
    score is the ``overall_mean`` (mean across the 4 dimensions). For
    task quality it is the single 1-5 score.
    """
    item_ids = sorted({s.conversation_id for ckpt in per_judge.values() for s in ckpt.scores})
    out: list[list[float]] = []
    for _judge_name, ckpt in per_judge.items():
        lookup = {s.conversation_id: s for s in ckpt.scores}
        row: list[float] = []
        for item_id in item_ids:
            score = lookup.get(item_id)
            if score is None:
                row.append(float("nan"))
                continue
            if rubric == "persona_adherence":
                row.append(score.persona_adherence.overall_mean)
            else:
                row.append(float(score.task_quality.score))
        out.append(row)
    return out


def load_checkpoints_from_dir(output_dir: Path) -> dict[str, JudgeCheckpoint]:
    """Load all ``judge_*.json`` checkpoints under ``output_dir``."""
    out: dict[str, JudgeCheckpoint] = {}
    for path in sorted(output_dir.glob("judge_*.json")):
        ckpt = JudgeCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        out[ckpt.judge_name] = ckpt
    return out


def write_combined_summary(output_dir: Path, results: dict[str, MetricResult]) -> Path:
    """Write a single ``poll_summary.json`` at the panel output dir."""
    path = output_dir / "poll_summary.json"
    payload = {name: r.model_dump() for name, r in results.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "JudgeCheckpoint",
    "JudgeFormat",
    "JudgeSpec",
    "PerJudgeConversationScore",
    "PoLLPanel",
    "load_checkpoints_from_dir",
    "reliability_matrix_from_checkpoints",
    "run_judge_sweep",
    "write_combined_summary",
]
