"""Tests for the Gemma 2 backend.

Two tiers:

- **Local / fast tests** (no marker): static surface and config tests + the
  Gemma-specific ``_fold_system_into_user`` logic + the ``format_persona_prompt``
  shape for Gemma. Always runnable, no model load, no GPU.
- **Slow + GPU tests** (``@pytest.mark.slow`` AND ``@pytest.mark.gpu``): actual model
  load, generation, hidden-state capture, reproducibility, and the NaN guard
  exercised both ways. Skipped by default and in CI; only run on the V100 servers
  with ``pytest -m "slow and gpu"``.
"""

from __future__ import annotations

import os

import pytest
import torch

from persona_rag.models import (
    ChatMessage,
    GemmaBackend,
    GenerationConfig,
    LLMBackend,
    load_backend,
)
from persona_rag.models.gemma import _fold_system_into_user

# ---------------------------------------------------------------------------
# Local tests (always run)
# ---------------------------------------------------------------------------


def test_gemma_default_config_uses_eager_and_fp16():
    cfg = GemmaBackend.default_config()
    assert cfg.attn_implementation == "eager", (
        "Gemma 2 requires eager attention for softcap correctness"
    )
    assert cfg.compute_dtype == "float16", "V100 has no hardware bfloat16"
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.model_id == "google/gemma-2-9b-it"
    assert cfg.warmup_nan_guard is True, "NaN guard must be on by default per spec §7"


def test_gemma_config_overrides_work():
    cfg = GemmaBackend.default_config(compute_dtype="bfloat16", warmup_nan_guard=False)
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.warmup_nan_guard is False
    # But attn_implementation must not be overridden implicitly.
    assert cfg.attn_implementation == "eager"


def test_fold_no_system_passes_through_unchanged():
    msgs = [
        ChatMessage("user", "hello"),
        ChatMessage("assistant", "hi"),
        ChatMessage("user", "what's the weather"),
    ]
    assert _fold_system_into_user(msgs) == msgs


def test_fold_leading_system_merges_into_first_user():
    msgs = [
        ChatMessage("system", "You are a helpful assistant."),
        ChatMessage("user", "What is 2+2?"),
    ]
    out = _fold_system_into_user(msgs)
    assert len(out) == 1
    assert out[0].role == "user"
    assert out[0].content.startswith("You are a helpful assistant.")
    assert out[0].content.endswith("What is 2+2?")


def test_fold_multiple_leading_systems_concatenate():
    msgs = [
        ChatMessage("system", "You are a doctor."),
        ChatMessage("system", "You always recommend seeing a specialist."),
        ChatMessage("user", "I have a headache."),
    ]
    out = _fold_system_into_user(msgs)
    assert len(out) == 1
    assert "doctor" in out[0].content
    assert "specialist" in out[0].content
    assert "headache" in out[0].content


def test_fold_does_not_mutate_input():
    msgs = [ChatMessage("system", "foo"), ChatMessage("user", "bar")]
    original = list(msgs)
    _ = _fold_system_into_user(msgs)
    assert msgs == original


def test_fold_trailing_system_becomes_user_turn():
    msgs = [
        ChatMessage("user", "Hello"),
        ChatMessage("assistant", "Hi"),
        ChatMessage("system", "Be concise."),
    ]
    out = _fold_system_into_user(msgs)
    assert out[-1].role == "user"
    assert "Be concise" in out[-1].content


def test_format_persona_prompt_returns_string_without_model_load():
    """``format_persona_prompt`` requires a tokenizer, so we can't fully test it
    without loading the model. This test documents that the Gemma backend's
    override exists and has the right signature — full end-to-end check is in
    the slow/GPU tier.
    """
    assert hasattr(GemmaBackend, "format_persona_prompt")


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
def gemma() -> LLMBackend:
    """Module-scoped Gemma backend to amortize the ~20s load across tests."""
    return load_backend("gemma")


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_loads_and_generates(gemma: LLMBackend) -> None:
    out = gemma.generate("Paris is the capital of", max_new_tokens=16, seed=42)
    assert isinstance(out, str)
    assert out.strip(), "backend returned empty output"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_hidden_states_shapes(gemma: LLMBackend) -> None:
    probe_layers = [8, 12, 16, 20]
    # over="prompt" is deterministic and cheap; exercises all three pool modes.
    hs_mean = gemma.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="mean", over="prompt"
    )
    hs_last = gemma.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="last", over="prompt"
    )
    hs_none = gemma.get_hidden_states(
        "The quick brown fox", layers=probe_layers, pool="none", over="prompt"
    )
    for idx in probe_layers:
        assert hs_mean[idx].shape == (gemma.hidden_dim,)
        assert hs_last[idx].shape == (gemma.hidden_dim,)
        assert hs_none[idx].dim() == 2
        assert hs_none[idx].shape[-1] == gemma.hidden_dim
        assert not torch.isnan(hs_mean[idx]).any()
        assert hs_mean[idx].device.type == "cpu", "hidden states must be returned on CPU"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_hidden_states_generation_scope(gemma: LLMBackend) -> None:
    """``over="generation"`` must actually capture generated-token hidden states."""
    hs = gemma.get_hidden_states(
        "Describe a sunset in one sentence.",
        layers=[12],
        pool="none",
        over="generation",
        max_new_tokens=16,
    )
    t = hs[12]
    assert t.dim() == 2
    # Should have at least a couple of generated tokens before EOS.
    assert t.shape[0] >= 2, f"expected >=2 generated tokens, got shape {tuple(t.shape)}"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_greedy_is_reproducible(gemma: LLMBackend) -> None:
    """Same prompt + same seed + greedy → identical output."""
    a = gemma.generate("The best way to learn a new skill is to", max_new_tokens=32, seed=42)
    b = gemma.generate("The best way to learn a new skill is to", max_new_tokens=32, seed=42)
    assert a == b, f"greedy output diverged across runs:\n a={a!r}\n b={b!r}"


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_format_persona_prompt_inline_prepends(gemma: LLMBackend) -> None:
    """Gemma's override folds ``system_text`` into the first user turn."""
    rendered = gemma.format_persona_prompt(
        system_text="You are a concise assistant.",
        user_text="Say hi.",
    )
    # Gemma's chat template wraps user turns in <start_of_turn>user ... <end_of_turn>.
    # We don't assert on exact template output; we assert that the system text landed
    # inside the rendered prompt and no explicit "system" role marker appears.
    assert "concise assistant" in rendered
    assert "<start_of_turn>user" in rendered
    # Gemma template never emits "system" because the role is rejected — so the string
    # "<start_of_turn>system" must NOT appear.
    assert "<start_of_turn>system" not in rendered


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_chat_does_not_raise_template_error(gemma: LLMBackend) -> None:
    """Regression: ``chat`` must not raise Gemma's ``System role not supported`` error."""
    out = gemma.chat(
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
def test_gemma_nan_guard_passes_on_normal_case(gemma: LLMBackend) -> None:
    """Happy path: on a normal prompt, the guard reports OK."""
    ok, stats = gemma.check_logits_finite("The capital of France is Paris.")
    assert ok
    assert not stats["has_nan"]
    assert not stats["has_inf"]


@pytest.mark.slow
@pytest.mark.gpu
@_NEEDS_GPU
@_NEEDS_TOKEN
def test_gemma_nan_guard_triggers_on_synthetic_case() -> None:
    """Synthetic: feed NaN directly through the guard's check path and expect trigger.

    We can't easily produce a real NaN logit without breaking something on purpose, so
    we monkey-patch the model's forward once to return a NaN-poisoned tensor and
    verify :meth:`check_logits_finite` returns ``(False, {has_nan: True})``.
    """
    backend = load_backend("gemma")

    # Build a fake forward output. We reach into the backend briefly — this is the one
    # test path where touching internals is the point.
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
