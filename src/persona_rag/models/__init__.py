"""LLM backend wrappers. Shared interface in :mod:`persona_rag.models.base`."""

from persona_rag.models._hf_base import HFBackend, HFBackendConfig
from persona_rag.models.base import ChatMessage, GenerationConfig, LLMBackend
from persona_rag.models.gemma import GemmaBackend
from persona_rag.models.glm_api import GlmApiBackend
from persona_rag.models.llama import LlamaBackend
from persona_rag.models.prometheus import PrometheusBackend
from persona_rag.models.qwen import QwenBackend


def load_backend(name: str, **overrides: object) -> LLMBackend:
    """Factory: build a backend by short name.

    Names match the Hydra configs in ``src/persona_rag/config/model/``.
    """
    if name in {"gemma", "gemma2", "gemma2-9b-it"}:
        return GemmaBackend(GemmaBackend.default_config(**overrides))
    if name in {"llama", "llama3", "llama3.1", "llama3.1-8b-instruct"}:
        return LlamaBackend(LlamaBackend.default_config(**overrides))
    if name in {"prometheus", "prometheus2", "prometheus-2-7b"}:
        return PrometheusBackend(PrometheusBackend.default_config(**overrides))
    if name in {"qwen", "qwen2.5", "qwen2.5-7b", "qwen2.5-7b-instruct"}:
        return QwenBackend(QwenBackend.default_config(**overrides))
    if name in {"glm", "glm-4.7", "glm4.7", "glm-api"}:
        return GlmApiBackend(**overrides)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown backend {name!r}; expected one of: gemma, llama, prometheus, qwen2.5, glm-api"
    )


__all__ = [
    "ChatMessage",
    "GemmaBackend",
    "GenerationConfig",
    "GlmApiBackend",
    "HFBackend",
    "HFBackendConfig",
    "LLMBackend",
    "LlamaBackend",
    "PrometheusBackend",
    "QwenBackend",
    "load_backend",
]
