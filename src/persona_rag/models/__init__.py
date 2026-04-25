"""LLM backend wrappers. Shared interface in :mod:`persona_rag.models.base`."""

from persona_rag.models._hf_base import HFBackend, HFBackendConfig
from persona_rag.models.base import ChatMessage, GenerationConfig, LLMBackend
from persona_rag.models.gemma import GemmaBackend
from persona_rag.models.llama import LlamaBackend


def load_backend(name: str, **overrides: object) -> LLMBackend:
    """Factory: build a backend by short name.

    Names match the Hydra configs in ``src/persona_rag/config/model/``.
    """
    if name in {"gemma", "gemma2", "gemma2-9b-it"}:
        return GemmaBackend(GemmaBackend.default_config(**overrides))
    if name in {"llama", "llama3", "llama3.1", "llama3.1-8b-instruct"}:
        return LlamaBackend(LlamaBackend.default_config(**overrides))
    raise ValueError(f"Unknown backend {name!r}; expected one of: gemma, llama")


__all__ = [
    "ChatMessage",
    "GemmaBackend",
    "GenerationConfig",
    "HFBackend",
    "HFBackendConfig",
    "LLMBackend",
    "LlamaBackend",
    "load_backend",
]
