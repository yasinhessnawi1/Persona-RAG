"""Retrieval pipelines: baselines and shared infrastructure.

This package ships two end-to-end retrieval pipelines built on a common
`RetrievalPipeline` Protocol + `Response` Pydantic contract:

- :class:`VanillaRAG` — hybrid knowledge retrieval, neutral system prompt, no
  persona conditioning.
- :class:`PromptPersonaRAG` — hybrid knowledge retrieval plus a structured
  persona system block and hand-authored dialogue few-shot exchanges (with an
  audit-only one-liner-shim variant for strength comparisons).

Hybrid-retrieval helpers (RRF + weighted-sum fusion, BM25 tokenisation) live
alongside so future pipelines that consume the same surfaces share one
implementation.
"""

from persona_rag.retrieval.base import Response, RetrievalPipeline, Turn
from persona_rag.retrieval.character_rm import (
    DEFAULT_CHARACTER_RM_MODEL_ID,
    CharacterRMScorer,
    CharacterScorer,
    FakeCharacterRMScorer,
)
from persona_rag.retrieval.drift_gate import LlmJudgeDriftGate
from persona_rag.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
    weighted_sum_fusion,
)
from persona_rag.retrieval.hybrid_ranker import (
    DEFAULT_ENABLED_SIGNALS,
    DEFAULT_HYBRID_WEIGHTS,
    HybridRanker,
    RankedCandidate,
)
from persona_rag.retrieval.mechanism_drift_gated import (
    DEFAULT_EXTRA_CANDIDATE_TEMPERATURES,
    DriftGatedMechanism,
)
from persona_rag.retrieval.prompt_persona import PromptPersonaRAG
from persona_rag.retrieval.prompt_templates import (
    B1_VANILLA_RAG_SYSTEM,
    FewShotBundle,
    FewShotExchange,
    FewShotTurn,
    render_b1_user_block,
    render_b2_persona_block,
    render_b2_simple_system,
    render_b2_user_block,
)
from persona_rag.retrieval.templates import (
    DRIFT_GATE_TEMPLATE_VERSION,
    DriftCheck,
    parse_drift_gate_response,
    render_drift_gate_prompt,
)
from persona_rag.retrieval.typed_retrieval import (
    DEFAULT_EPISTEMIC_ALLOWLIST,
    TypedRetrievalRAG,
    render_typed_system_block,
)
from persona_rag.retrieval.vanilla_rag import VanillaRAG

__all__ = [
    "B1_VANILLA_RAG_SYSTEM",
    "DEFAULT_CHARACTER_RM_MODEL_ID",
    "DEFAULT_ENABLED_SIGNALS",
    "DEFAULT_EPISTEMIC_ALLOWLIST",
    "DEFAULT_EXTRA_CANDIDATE_TEMPERATURES",
    "DEFAULT_HYBRID_WEIGHTS",
    "DEFAULT_RRF_K",
    "DRIFT_GATE_TEMPLATE_VERSION",
    "CharacterRMScorer",
    "CharacterScorer",
    "DriftCheck",
    "DriftGatedMechanism",
    "FakeCharacterRMScorer",
    "FewShotBundle",
    "FewShotExchange",
    "FewShotTurn",
    "HybridRanker",
    "LlmJudgeDriftGate",
    "PromptPersonaRAG",
    "RankedCandidate",
    "Response",
    "RetrievalPipeline",
    "Turn",
    "TypedRetrievalRAG",
    "VanillaRAG",
    "parse_drift_gate_response",
    "reciprocal_rank_fusion",
    "render_b1_user_block",
    "render_b2_persona_block",
    "render_b2_simple_system",
    "render_b2_user_block",
    "render_drift_gate_prompt",
    "render_typed_system_block",
    "weighted_sum_fusion",
]
