"""Versioned prompt templates shared across retrieval pipelines.

Each template ships as a Python module exporting one or more concat-style
renderer functions plus a ``*_TEMPLATE_VERSION`` constant. Per-turn metadata
records the version string so cross-run results stay interpretable when a
template iterates. Any change to a template body bumps the version *and*
lands a new decisions-log entry.
"""

from persona_rag.retrieval.templates.drift_gate import (
    DRIFT_GATE_TEMPLATE_VERSION,
    DriftCheck,
    parse_drift_gate_response,
    render_drift_gate_prompt,
)

__all__ = [
    "DRIFT_GATE_TEMPLATE_VERSION",
    "DriftCheck",
    "parse_drift_gate_response",
    "render_drift_gate_prompt",
]
