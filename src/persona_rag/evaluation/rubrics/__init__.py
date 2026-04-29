"""Rubric prompt templates and parsers for the PoLL panel.

Two rubric families ship:

- :func:`render_persona_adherence_native_prometheus_prompt` /
  :func:`render_persona_adherence_json_prompt` -- score self-fact /
  worldview / constraint adherence and overall persona fidelity on a
  1-5 scale across four dimensions.
- :func:`render_task_quality_native_prometheus_prompt` /
  :func:`render_task_quality_json_prompt` -- single-dimension 1-5 score
  on whether the assistant correctly answered the user's knowledge
  question, independent of persona.

Two surface formats exist for each rubric: Prometheus's *native*
rubric (5-tier score descriptions, ``[RESULT] N`` parsing) for the
Prometheus-2 judge, and a JSON-out variant for the Qwen / Llama
judges that were not trained on Prometheus's surface.
"""

from persona_rag.evaluation.rubrics.parsers import (
    PERSONA_ADHERENCE_DIMENSIONS,
    PersonaAdherenceScore,
    TaskQualityScore,
    parse_persona_adherence_json,
    parse_persona_adherence_native_prometheus,
    parse_task_quality_json,
    parse_task_quality_native_prometheus,
)
from persona_rag.evaluation.rubrics.persona_adherence import (
    render_persona_adherence_json_prompt,
    render_persona_adherence_native_prometheus_prompt,
)
from persona_rag.evaluation.rubrics.task_quality import (
    render_task_quality_json_prompt,
    render_task_quality_native_prometheus_prompt,
)

__all__ = [
    "PERSONA_ADHERENCE_DIMENSIONS",
    "PersonaAdherenceScore",
    "TaskQualityScore",
    "parse_persona_adherence_json",
    "parse_persona_adherence_native_prometheus",
    "parse_task_quality_json",
    "parse_task_quality_native_prometheus",
    "render_persona_adherence_json_prompt",
    "render_persona_adherence_native_prometheus_prompt",
    "render_task_quality_json_prompt",
    "render_task_quality_native_prometheus_prompt",
]
