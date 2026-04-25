"""Tests for the Llama 3.1 backend.

Same two-tier structure as ``test_gemma.py``:

- **Local / fast tests**: config surface, default-value sanity, no model load.
- **Slow + GPU tests** (``@pytest.mark.slow`` + ``@pytest.mark.gpu``): actual model
  load, generation, hidden-state capture, reproducibility, NaN guard.
"""

from __future__ import annotations

import os

import pytest
import torch

from persona_rag.models import (
    ChatMessage,
    GenerationConfig,
    LlamaBackend,
    LLMBackend,
    load_backend,
)

# ---------------------------------------------------------------------------
# Local tests (always run)
# ---------------------------------------------------------------------------


def test_llama_default_config_uses_eager_and_fp16():
    cfg = LlamaBackend.default_config()
    assert cfg.attn_implementation == "eager", (
        "Llama mirrors Gemma's eager attention path for kernel-path symmetry"
    )
    assert cfg.compute_dtype == "float16"
    assert cfg.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.warmup_nan_guard is True


def test_llama_config_overrides_work():
    cfg = LlamaBackend.default_config(max_input_tokens=2048)
    assert cfg.max_input_tokens == 2048
    assert cfg.attn_implementation == "eager"


# ---------------------------------------------------------------------------
# Slow + GPU tests (run only on the V100 servers)
# ---------------------------------------------------------------------------


_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required; run on V100 server",
)
_NEEDS_TOKEN = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="Gated models require HF_TOKEN",
)


@pytest.fixture(scope="module")
def llama() -> LLMBackend:
    """Module-scoped Llama backend to amortize the ~20s load across tests."""
    return load_backend("llama")


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_loads_and_generates(llama: LLMBackend) -> None:
    out = llama.generate("Paris is the capital of", max_new_tokens=16, seed=42)
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_hidden_states_shapes(llama: LLMBackend) -> None:
    probe_layers = [8, 12, 16, 20]
    hs_mean = llama.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="mean", over="prompt"
    )
    hs_last = llama.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="last", over="prompt"
    )
    hs_none = llama.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="none", over="prompt"
    )
    for idx in probe_layers:
        assert hs_mean[idx].shape == (llama.hidden_dim,)
        assert hs_last[idx].shape == (llama.hidden_dim,)
        assert hs_none[idx].dim() == 2
        assert hs_none[idx].shape[-1] == llama.hidden_dim
        assert hs_mean[idx].device.type == "cpu"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_greedy_is_reproducible(llama: LLMBackend) -> None:
    a = llama.generate("The best way to learn a new skill is to", max_new_tokens=32, seed=42)
    b = llama.generate("The best way to learn a new skill is to", max_new_tokens=32, seed=42)
    assert a == b, f"greedy output diverged across runs:\n a={a!r}\n b={b!r}"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_format_persona_prompt_uses_system_role(llama: LLMBackend) -> None:
    """Llama supports the native ``system`` role — the rendered prompt should contain
    the ``<|start_header_id|>system<|end_header_id|>`` block, not an inline prefix
    on the user turn."""
    rendered = llama.format_persona_prompt(
        system_text="You are a concise assistant.",
        user_text="Say hi.",
    )
    assert "concise assistant" in rendered
    assert "system" in rendered.lower()
    # Llama 3.1's header markers. We don't hardcode the exact tokens (version drift)
    # but the "system" header should appear as a distinct block before "user".
    assert rendered.lower().index("system") < rendered.lower().index("user")


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_chat_with_system_role(llama: LLMBackend) -> None:
    """Regression: Llama ``chat`` accepts a native system turn and produces output."""
    out = llama.chat(
        [
            ChatMessage("system", "You are a concise assistant."),
            ChatMessage("user", "Say hi."),
        ],
        cfg=GenerationConfig(max_new_tokens=8, do_sample=False, seed=42),
    )
    assert out.strip()


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_nan_guard_passes_on_normal_case(llama: LLMBackend) -> None:
    ok, stats = llama.check_logits_finite("The capital of France is Paris.")
    assert ok
    assert not stats["has_nan"]
    assert not stats["has_inf"]


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_llama_nan_guard_triggers_on_synthetic_case() -> None:
    """Poison the forward pass with NaN and verify the guard flags it."""
    backend = load_backend("llama")

    real_forward = backend._model.__call__  # type: ignore[attr-defined]

    class _NanOut:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    def _poisoned(*args, **kwargs):  # type: ignore[no-untyped-def]
        real = real_forward(*args, **kwargs)
        poisoned = real.logits.clone()
        poisoned[..., 0] = float("nan")
        return _NanOut(poisoned)

    backend._model.__call__ = _poisoned  # type: ignore[attr-defined]
    try:
        ok, stats = backend.check_logits_finite("anything")
        assert not ok
        assert stats["has_nan"] is True
    finally:
        backend._model.__call__ = real_forward  # type: ignore[attr-defined]
