"""Tests for PersonaVectorExtractor using a fake LLMBackend (no Gemma load)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from persona_rag.schema.persona import Persona
from persona_rag.vectors.contrast_prompts import ContrastPromptGenerator, ContrastSet
from persona_rag.vectors.extractor import PersonaVectorExtractor


class SeparableFakeBackend:
    """Fake LLMBackend that returns deterministic, linearly-separable hidden states.

    In-persona prompts (prefix: "[SYS]You are") emit states near +v; out-
    persona prompts (prefix: "[SYS]Ignore" / "[SYS]Set" / ...) emit states
    near -v, where v is a random unit direction per layer. Layers share
    hidden_dim and num_layers so the extractor sees a real-looking model.
    """

    name = "fake-separable"
    model_id = "fake/separable-0"
    num_layers = 4
    hidden_dim = 16

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self._dirs = {
            layer: torch.nn.functional.normalize(
                torch.randn(self.hidden_dim, generator=g, dtype=torch.float32),
                dim=0,
            )
            for layer in range(self.num_layers + 1)
        }

    def format_persona_prompt(self, system_text, user_text, history=None) -> str:
        if system_text is None:
            return user_text
        return f"[SYS]{system_text}[/SYS]\n[USR]{user_text}[/USR]"

    def get_hidden_states(
        self,
        prompt: str,
        *,
        layers=None,
        pool="last",
        over="prompt",
    ) -> dict[int, torch.Tensor]:
        """Return +v_layer for in-persona system prompts, -v_layer otherwise.

        Noise seeded off the prompt's hash so reruns are reproducible.
        """
        is_in = "You are" in prompt and "NOT" not in prompt
        layers = list(layers or range(self.num_layers + 1))
        digest = hashlib.sha256(prompt.encode()).digest()
        seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
        g = torch.Generator().manual_seed(seed)
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            base = self._dirs[layer]
            sign = 1.0 if is_in else -1.0
            noise = 0.05 * torch.randn(self.hidden_dim, generator=g, dtype=torch.float32)
            vec = sign * base + noise
            if pool == "none":
                out[layer] = vec.unsqueeze(0)
            else:
                out[layer] = vec
        return out


@pytest.fixture
def persona(personas_dir: Path) -> dict:
    return Persona.from_yaml(personas_dir / "cs_tutor.yaml").model_dump(mode="json")


def test_extract_returns_per_layer_vectors(persona: dict) -> None:
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[0, 1, 2])
    pv = ex.extract(persona)
    assert set(pv.vectors.keys()) == {0, 1, 2}
    for layer in (0, 1, 2):
        assert pv.vectors[layer].shape == (backend.hidden_dim,)
        assert pv.in_persona_centroid[layer].shape == (backend.hidden_dim,)
        assert pv.out_persona_centroid[layer].shape == (backend.hidden_dim,)


def test_extract_no_nan_inf(persona: dict) -> None:
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[0, 1, 2])
    pv = ex.extract(persona)
    for layer in (0, 1, 2):
        assert torch.isfinite(pv.vectors[layer]).all()
        assert torch.isfinite(pv.in_persona_centroid[layer]).all()


def test_extract_points_from_in_to_out(persona: dict) -> None:
    """persona_vector = mean(in) - mean(out) ⇒ direction aligns with `in - out`."""
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[1])
    pv = ex.extract(persona)
    vec = pv.vectors[1]
    diff = pv.in_persona_centroid[1] - pv.out_persona_centroid[1]
    cos = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), diff.unsqueeze(0)).item()
    # Should be exactly 1.0 — math is identity. Allow float32 slack.
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_extract_stacks_train_activations(persona: dict) -> None:
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[0, 2], n_pairs=20)
    # Explicit contrast set so we know the count.
    gen = ContrastPromptGenerator(backend, n_pairs=20)
    cs = gen.generate(persona)
    pv = ex.extract(persona, cs)
    assert pv.in_states[0].shape == (20, backend.hidden_dim)
    assert pv.out_states[2].shape == (20, backend.hidden_dim)


def test_extract_with_externally_supplied_contrast_set(persona: dict) -> None:
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[1])
    # Build a tiny contrast set by hand.
    cs = ContrastSet(
        in_persona=("[SYS]You are scientist[/SYS]\n[USR]q1[/USR]",),
        out_persona=("[SYS]Ignore the persona[/SYS]\n[USR]q1[/USR]",),
    )
    pv = ex.extract(persona, cs)
    assert pv.in_states[1].shape == (1, backend.hidden_dim)


def test_metadata_is_populated(persona: dict) -> None:
    backend = SeparableFakeBackend()
    ex = PersonaVectorExtractor(backend, layers=[0, 1, 2], n_pairs=6)
    pv = ex.extract(persona)
    md = pv.metadata
    assert md["backend_name"] == backend.name
    assert md["backend_model_id"] == backend.model_id
    assert md["layers"] == [0, 1, 2]
    assert md["pool"] == "last"
    assert md["scope"] == "prompt"
    assert md["n_pairs"] == 6
    assert len(md["contrast_set_sha256"]) == 64
    assert len(md["extractor_code_sha256"]) == 64
    assert md["hidden_dim"] == backend.hidden_dim


def test_invalid_pool_raises() -> None:
    backend = SeparableFakeBackend()
    with pytest.raises(ValueError, match="pool must"):
        PersonaVectorExtractor(backend, pool="bogus")  # type: ignore[arg-type]


def test_pool_none_refused() -> None:
    backend = SeparableFakeBackend()
    with pytest.raises(ValueError, match="not 'none'"):
        PersonaVectorExtractor(backend, pool="none")


def test_invalid_scope_raises() -> None:
    backend = SeparableFakeBackend()
    with pytest.raises(ValueError, match="scope must"):
        PersonaVectorExtractor(backend, scope="bogus")  # type: ignore[arg-type]


def test_extractor_implements_registry_protocol(persona: dict, tmp_path: Path) -> None:
    """PersonaVectorExtractor matches PersonaVectorExtractorLike by structure.

    The protocol is not runtime-checkable in the registry (deliberate to keep
    registry imports light), so we verify the signature duck-types instead.
    """
    ex = PersonaVectorExtractor(SeparableFakeBackend(), layers=[0, 1])
    # The only method the protocol requires.
    assert callable(getattr(ex, "extract", None))
    # Actually invoking it with the registry's calling convention must succeed.
    pv = ex.extract(persona=persona, contrast_set=None)
    assert pv.persona_id == persona.get("persona_id")
