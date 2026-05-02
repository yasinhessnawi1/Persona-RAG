"""Smoke tests for the off-spec Free-Transformer experiment.

These cover only the Darwin-runnable surface: model forward, BinaryMapper
gradient pass-through, KL math, and the Hydra entry-point config. The full
training and Z-separability runs are V100-only.
"""

from __future__ import annotations

import math

import pytest
import torch

from persona_rag.freet.corpus import CorpusBuildConfig, build_corpus
from persona_rag.freet.model import (
    BinaryMapper,
    FreeTransformer,
    FreeTransformerConfig,
    free_transformer_loss,
)


@pytest.fixture
def small_cfg() -> FreeTransformerConfig:
    return FreeTransformerConfig(
        vocab_size=257,
        dim=64,
        n_layers=4,
        n_q_heads=4,
        n_kv_heads=2,
        mlp_ratio=2,
        max_seq_len=32,
        latent_bits=4,  # 16 categories — small enough to enumerate in tests
        pad_token_id=0,
    )


def test_inject_after_layer_default(small_cfg: FreeTransformerConfig) -> None:
    """Default inject_after_layer is n_layers // 2."""
    assert small_cfg.inject_after_layer == small_cfg.n_layers // 2


def test_forward_shape(small_cfg: FreeTransformerConfig) -> None:
    """Forward pass returns (B, T, V) logits and per-token KL."""
    model = FreeTransformer(small_cfg)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    out = model(tokens)
    assert out.logits.shape == (2, 16, small_cfg.vocab_size)
    assert out.kl_per_token is not None
    assert out.kl_per_token.shape == (2, 16)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.kl_per_token).all()


def test_inference_z_from_prior(small_cfg: FreeTransformerConfig) -> None:
    """sample_z_from_prior=True produces logits with kl_per_token=None."""
    model = FreeTransformer(small_cfg).eval()
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 8))
    out = model(tokens, sample_z_from_prior=True)
    assert out.kl_per_token is None
    assert out.logits.shape == (1, 8, small_cfg.vocab_size)


def test_binary_mapper_one_hot(small_cfg: FreeTransformerConfig) -> None:
    """BinaryMapper produces hard one-hot Z."""
    bm = BinaryMapper(small_cfg.latent_bits)
    logits = torch.randn(2, 5, small_cfg.latent_bits)
    z, kl = bm(logits)
    assert z.shape == (2, 5, 1 << small_cfg.latent_bits)
    # Hard one-hot: each (B, T) slot sums to ~1 (gradient pass-through term has
    # zero forward contribution because Y + G - G.detach() == Y in value).
    assert torch.allclose(z.sum(dim=-1), torch.ones(2, 5), atol=1e-4)
    assert kl.shape == (2, 5)


def test_binary_mapper_kl_against_uniform_prior() -> None:
    """If sigmoid logits are 0 (uniform Bernoulli per bit), Q ≡ uniform => KL = 0.

    KL bitwise: p log(2p) + (1-p) log(2(1-p)). At p=0.5 each bit contributes
    0.5*log(1) + 0.5*log(1) = 0; sum over H bits = 0.
    """
    bm = BinaryMapper(latent_bits=8)
    logits = torch.zeros(1, 4, 8)
    _, kl = bm(logits)
    assert torch.allclose(kl, torch.zeros(1, 4), atol=1e-6)


def test_binary_mapper_kl_sigmoid_high_logits() -> None:
    """If sigmoid logits are very large positive, Q is near a single index.
    KL approaches H*log(2) (the maximum) per token.
    """
    bm = BinaryMapper(latent_bits=4)
    logits = torch.full((1, 1, 4), 50.0)
    _, kl = bm(logits)
    expected = 4 * math.log(2.0)
    assert kl.item() == pytest.approx(expected, rel=1e-3)


def test_binary_mapper_gradient_passthrough() -> None:
    """Gradient of Z·v should flow back to the logits via the joint Bernoulli prob.

    With Y + G - G.detach(), forward returns the hard Y but backward propagates
    through G (paper Eq. 8). We check that .backward() through z * v produces
    a non-zero gradient on the input logits.
    """
    torch.manual_seed(0)
    bm = BinaryMapper(latent_bits=3)
    logits = torch.randn(1, 2, 3, requires_grad=True)
    z, _ = bm(logits)
    v = torch.randn(1 << 3)
    (z @ v).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss_excess_only_above_kappa(small_cfg: FreeTransformerConfig) -> None:
    """Free-bits term is 0 when KL < κ for every token."""
    logits = torch.randn(2, 4, small_cfg.vocab_size)
    targets = torch.randint(0, small_cfg.vocab_size, (2, 4))
    kl = torch.full((2, 4), 0.05)
    total, metrics = free_transformer_loss(
        logits, targets, kl, pad_token_id=0, free_bits_kappa=0.5
    )
    assert metrics["kl_excess"] == pytest.approx(0.0)
    assert total.item() == pytest.approx(metrics["ce"], rel=1e-5)


def test_loss_excess_when_kl_exceeds_kappa() -> None:
    """When all KL_t = 1.0 and κ = 0.5, the excess term contributes 0.5."""
    logits = torch.randn(2, 4, 17)
    targets = torch.randint(0, 17, (2, 4))
    kl = torch.full((2, 4), 1.0)
    _total, metrics = free_transformer_loss(
        logits, targets, kl, pad_token_id=0, free_bits_kappa=0.5
    )
    assert metrics["kl_excess"] == pytest.approx(0.5, rel=1e-5)


def test_train_loss_is_finite(small_cfg: FreeTransformerConfig) -> None:
    """End-to-end forward + loss is finite + .backward() doesn't NaN."""
    torch.manual_seed(0)
    model = FreeTransformer(small_cfg)
    tokens = torch.randint(1, small_cfg.vocab_size, (2, 8))
    out = model(tokens)
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:]
    targets[:, -1] = small_cfg.pad_token_id
    assert out.kl_per_token is not None
    loss, _metrics = free_transformer_loss(
        out.logits, targets, out.kl_per_token,
        pad_token_id=small_cfg.pad_token_id, free_bits_kappa=0.5 * math.log(2.0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_corpus_builds_under_a_temp_dir(tmp_path) -> None:
    """The corpus builder reads the in-repo personas + drift YAMLs and writes JSONL."""
    out = tmp_path / "freet_corpus"
    cfg = CorpusBuildConfig(
        seed=0,
        facts_repeats=2,
        worldview_repeats=2,
        constraints_repeats=2,
        episodic_repeats=2,
        intro_repeats=4,
        drift_repeats=2,
        probe_repeats=1,
        chunk_repeats=1,
    )
    stats = build_corpus(out, cfg)
    assert stats["n_total"] > 0
    assert (out / "train.jsonl").exists()
    assert (out / "test.jsonl").exists()
    # All three personas should be represented.
    assert set(stats["by_persona"].keys()) == {"climate_scientist", "cs_tutor", "historian"}
    for pid, n in stats["by_persona"].items():
        assert n > 0, f"persona {pid} produced 0 examples"


def test_hydra_train_config_loads() -> None:
    """The training entry config resolves on Darwin without touching CUDA / weights.

    Reproduces the lesson from the persona-vector V100 run: a Hydra dry-run that
    fails on Darwin would fail identically on the server.
    """
    pytest.importorskip("hydra")
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    cfg_dir = (Path(__file__).resolve().parents[1] / "src" / "persona_rag" / "config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="train_freet")
        assert "freet" in cfg
        assert cfg.freet.dim > 0
        assert cfg.freet.free_bits_kappa_bits > 0
        # report_dir uses ${now:...} so the value will be non-trivially expanded.
        assert "results/freet/" in str(cfg.report_dir)


def test_hydra_validate_config_loads() -> None:
    pytest.importorskip("hydra")
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    cfg_dir = (Path(__file__).resolve().parents[1] / "src" / "persona_rag" / "config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="validate_freet")
        assert cfg.ckpt == "???"  # required override at runtime
        assert "freet" in cfg
