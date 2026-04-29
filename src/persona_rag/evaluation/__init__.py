"""Evaluation harness: metrics, judges, runner, human validation.

Entry points:

- :class:`EvaluationRunner` -- drive a metric stack across (mechanism,
  persona) cells; emit long-form CSV.
- :class:`MiniCheckMetric` -- primary self-fact contradiction metric.
- :class:`SyconMetric` -- worldview-stance ToF/NoF flip rate.
- :class:`PoLLPanel` -- sequential 3-judge persona-adherence + task-quality
  scoring with per-judge checkpointing.
- :class:`CostTracker` -- per-mechanism LLM-call / latency aggregation
  from existing per-turn pipeline metadata.
- :class:`DriftQualityMetric` -- M3 gate precision / recall / F1 against
  MiniCheck-derived inconsistency labels.
- :class:`RefCheckerMetric` -- soft-optional triplet-level secondary
  metric.

The smoke-test stability suite (used by `smoke_test_models.py`) ships
alongside under :mod:`persona_rag.evaluation.smoke_suite`.
"""

from persona_rag.evaluation.cost import CostTracker
from persona_rag.evaluation.drift_quality import ConfusionCounts, DriftQualityMetric
from persona_rag.evaluation.metrics import (
    EvalConversation,
    Metric,
    MetricResult,
    ScoredTurn,
)
from persona_rag.evaluation.minicheck_metric import (
    DEFAULT_MINICHECK_MODEL_ID,
    HFMiniCheckScorer,
    MiniCheckMetric,
    MiniCheckScorer,
    split_sentences,
)
from persona_rag.evaluation.poll_panel import (
    JudgeCheckpoint,
    JudgeFormat,
    JudgeSpec,
    PerJudgeConversationScore,
    PoLLPanel,
    load_checkpoints_from_dir,
    reliability_matrix_from_checkpoints,
    run_judge_sweep,
    write_combined_summary,
)
from persona_rag.evaluation.refchecker_metric import (
    RefCheckerMetric,
    is_refchecker_available,
)
from persona_rag.evaluation.runner import (
    AGGREGATE_HEADER,
    LONG_FORM_HEADER,
    EvaluationRunner,
    MechanismCell,
)
from persona_rag.evaluation.sycon_metric import (
    FlipStats,
    LlmStanceClassifier,
    Stance,
    StanceCheck,
    StanceClassifier,
    SyconMetric,
    compute_flip_stats,
    parse_stance_response,
    render_stance_prompt,
)
from persona_rag.evaluation.transcripts import (
    conversation_yaml_to_eval,
    load_baseline_response_dir,
    load_conversation_yamls,
)

__all__ = [
    "AGGREGATE_HEADER",
    "DEFAULT_MINICHECK_MODEL_ID",
    "LONG_FORM_HEADER",
    "ConfusionCounts",
    "CostTracker",
    "DriftQualityMetric",
    "EvalConversation",
    "EvaluationRunner",
    "FlipStats",
    "HFMiniCheckScorer",
    "JudgeCheckpoint",
    "JudgeFormat",
    "JudgeSpec",
    "LlmStanceClassifier",
    "MechanismCell",
    "Metric",
    "MetricResult",
    "MiniCheckMetric",
    "MiniCheckScorer",
    "PerJudgeConversationScore",
    "PoLLPanel",
    "RefCheckerMetric",
    "ScoredTurn",
    "Stance",
    "StanceCheck",
    "StanceClassifier",
    "SyconMetric",
    "compute_flip_stats",
    "conversation_yaml_to_eval",
    "is_refchecker_available",
    "load_baseline_response_dir",
    "load_checkpoints_from_dir",
    "load_conversation_yamls",
    "parse_stance_response",
    "reliability_matrix_from_checkpoints",
    "render_stance_prompt",
    "run_judge_sweep",
    "split_sentences",
    "write_combined_summary",
]
