"""Static tests for the 8-bit quantization path on ``HFBackendConfig``.

These tests do not load any model. They cover:

- the new ``load_in_8bit`` field exists and defaults to False;
- ``load_in_4bit`` and ``load_in_8bit`` are mutually exclusive at
  backend construction time;
- ``save_load_report`` includes the new field.

A separate slow + gpu test (under ``pytest -m "slow and gpu"`` on the
V100) exercises an actual 8-bit Gemma load — that test belongs with
the other GPU-marker'd backend tests and is intentionally not duplicated
here.
"""

from __future__ import annotations

import json

import pytest

from persona_rag.models import HFBackendConfig
from persona_rag.models._hf_base import HFBackend


def _base_cfg(**overrides):
    base = dict(
        model_id="google/gemma-2-9b-it",
        name="gemma2-9b-it-test",
        revision=None,
        compute_dtype="float16",
        attn_implementation="eager",
        max_input_tokens=3500,
        trust_remote_code=False,
        warmup_nan_guard=False,
    )
    base.update(overrides)
    return HFBackendConfig(**base)


def test_load_in_8bit_field_exists_and_defaults_false() -> None:
    cfg = _base_cfg()
    assert hasattr(cfg, "load_in_8bit")
    assert cfg.load_in_8bit is False
    # 4-bit remains the project default; 8-bit must be opt-in.
    assert cfg.load_in_4bit is True


def test_load_in_8bit_settable() -> None:
    cfg = _base_cfg(load_in_4bit=False, load_in_8bit=True)
    assert cfg.load_in_4bit is False
    assert cfg.load_in_8bit is True


def test_mutual_exclusion_raises_at_backend_init() -> None:
    """Both flags True is a configuration error; surfaced before any model load."""
    cfg = _base_cfg(load_in_4bit=True, load_in_8bit=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        HFBackend(cfg)  # raises *before* tokenizer/model load


def test_save_load_report_includes_load_in_8bit(tmp_path, monkeypatch) -> None:
    """The on-disk load report records both quantization flags."""
    # Build a minimal HFBackend stub without going through __init__ — we are
    # only testing the report writer, not the load path. Direct instantiation
    # of HFBackend would try to download weights; we sidestep that by setting
    # the private attrs the writer reads.
    backend = HFBackend.__new__(HFBackend)
    backend._cfg = _base_cfg(load_in_4bit=False, load_in_8bit=True)
    # Stub the properties the report calls (avoids requiring a loaded model).
    backend._model = type(
        "M",
        (),
        {"config": type("C", (), {"num_hidden_layers": 42, "hidden_size": 4096})()},
    )()

    report_path = tmp_path / "load_report.json"
    backend.save_load_report(report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["load_in_4bit"] is False
    assert data["load_in_8bit"] is True
    assert data["bnb_4bit_quant_type"] == "nf4"  # field present even when unused
