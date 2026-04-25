"""Persona-vector extraction, separability probing, and drift signal.

Replicates the persona-vectors methodology (Chen, Lindsey et al., arXiv
2507.21509) on Gemma-2-9B at 4-bit fp16 + eager attention. The validation
script ``scripts/validate_persona_vectors.py`` is the entry point that runs
the full pipeline and reports a separability verdict.

Primary user entry points:

- :class:`ContrastPromptGenerator` — builds ~50 in/out contrastive prompt
  pairs from a persona document.
- :class:`PersonaVectorExtractor` — implements the
  :class:`~persona_rag.schema.registry.PersonaVectorExtractorLike` protocol;
  computes mass-mean persona vectors per layer via
  :meth:`~persona_rag.models.base.LLMBackend.get_hidden_states`.
- :class:`SeparabilityProbe` — logistic-regression probe with shuffled-label
  and random-feature spurious-correlation controls; reports AUROC per layer.
- :class:`DriftSignal` — cosine similarity to the in-persona centroid,
  scaled to [-1, 1]. Consumed by drift-gated retrieval at inference time.
"""

from persona_rag.vectors.cache import (
    PersonaVectorCacheMeta,
    load_persona_vectors,
    save_persona_vectors,
)
from persona_rag.vectors.contrast_prompts import (
    ContrastPromptGenerator,
    ContrastSet,
)
from persona_rag.vectors.drift import DriftSignal
from persona_rag.vectors.extractor import PersonaVectorExtractor, PersonaVectors
from persona_rag.vectors.layer_selection import pick_global_best_layer
from persona_rag.vectors.probe import SeparabilityProbe, SeparabilityResult

__all__ = [
    "ContrastPromptGenerator",
    "ContrastSet",
    "DriftSignal",
    "PersonaVectorCacheMeta",
    "PersonaVectorExtractor",
    "PersonaVectors",
    "SeparabilityProbe",
    "SeparabilityResult",
    "load_persona_vectors",
    "pick_global_best_layer",
    "save_persona_vectors",
]
