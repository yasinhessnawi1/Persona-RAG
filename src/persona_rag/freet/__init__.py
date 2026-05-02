"""Off-spec branch experiment: Free Transformer adapted for persona-labeled data.

Reference: Fleuret 2025, arXiv 2510.17558.

This package is intentionally **isolated from the rest of the codebase**: no
other module imports from it, and nothing under `src/persona_rag/freet/` is
referenced by the persona-vector / RAG pipelines. The package exists so we
can train a small Free Transformer end-to-end on a single V100 and ask whether
its latent random tensor Z linearly separates the three personas.
"""

from persona_rag.freet.model import (
    BinaryMapper,
    FreeTransformer,
    FreeTransformerConfig,
    FreeTransformerEncoder,
)

__all__ = [
    "BinaryMapper",
    "FreeTransformer",
    "FreeTransformerConfig",
    "FreeTransformerEncoder",
]
