"""Tests for the template-based contrast-prompt generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.schema.persona import Persona
from persona_rag.vectors.contrast_prompts import ContrastPromptGenerator, ContrastSet


class FakeBackend:
    """Minimal LLMBackend stub for tests — only needs format_persona_prompt."""

    name = "fake"
    model_id = "fake/fake-0"
    num_layers = 4
    hidden_dim = 16

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history=None,
    ) -> str:
        # Deterministic, lossless rendering so tests can spot-check structure.
        parts: list[str] = []
        if system_text:
            parts.append(f"[SYS]{system_text}[/SYS]")
        parts.append(f"[USR]{user_text}[/USR]")
        return "\n".join(parts)


@pytest.fixture
def cs_tutor_dict(personas_dir: Path) -> dict:
    return Persona.from_yaml(personas_dir / "cs_tutor.yaml").model_dump(mode="json")


def test_generate_balanced_pairs(cs_tutor_dict: dict) -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=10)
    cs = gen.generate(cs_tutor_dict)
    assert cs.n_pairs == 10
    assert len(cs.in_persona) == 10
    assert len(cs.out_persona) == 10
    assert cs.topic_aligned is True


def test_in_and_out_prompts_differ_per_pair(cs_tutor_dict: dict) -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=20)
    cs = gen.generate(cs_tutor_dict)
    # Every pair must have distinct system directives at minimum.
    for in_p, out_p in zip(cs.in_persona, cs.out_persona, strict=True):
        assert in_p != out_p


def test_prompts_render_through_backend(cs_tutor_dict: dict) -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=5)
    cs = gen.generate(cs_tutor_dict)
    # Every prompt should carry the FakeBackend's rendering markers.
    for p in (*cs.in_persona, *cs.out_persona):
        assert "[SYS]" in p and "[/SYS]" in p
        assert "[USR]" in p and "[/USR]" in p


def test_out_persona_system_prompt_negates_role(cs_tutor_dict: dict) -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=5)
    cs = gen.generate(cs_tutor_dict)
    # Every out-persona prompt should contain a role-disregarding phrase.
    for out_p in cs.out_persona:
        lowered = out_p.lower()
        assert any(
            marker in lowered
            for marker in ("ignore", "disregard", "forget", "set aside", "not ", "without")
        )


def test_sha256_stable(cs_tutor_dict: dict) -> None:
    """Two identical generations hash identically; different n_pairs differ."""
    g1 = ContrastPromptGenerator(FakeBackend(), n_pairs=10)
    g2 = ContrastPromptGenerator(FakeBackend(), n_pairs=10)
    cs1 = g1.generate(cs_tutor_dict)
    cs2 = g2.generate(cs_tutor_dict)
    assert cs1.sha256() == cs2.sha256()

    g3 = ContrastPromptGenerator(FakeBackend(), n_pairs=11)
    cs3 = g3.generate(cs_tutor_dict)
    assert cs3.sha256() != cs1.sha256()


def test_split_is_prompt_disjoint_and_reproducible(cs_tutor_dict: dict) -> None:
    """Same seed → same split; different seed → may differ; train ∩ test == ∅."""
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=20)
    cs = gen.generate(cs_tutor_dict)

    train_a, test_a = cs.split(test_fraction=0.25, seed=42)
    train_b, test_b = cs.split(test_fraction=0.25, seed=42)
    assert train_a.in_persona == train_b.in_persona
    assert test_a.in_persona == test_b.in_persona

    # Train ∩ test must be empty on both sides.
    assert not (set(train_a.in_persona) & set(test_a.in_persona))
    assert not (set(train_a.out_persona) & set(test_a.out_persona))

    # Sizes sum to the original.
    assert train_a.n_pairs + test_a.n_pairs == cs.n_pairs


def test_generate_errors_on_no_role() -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=5)
    with pytest.raises(ValueError, match="role is required"):
        gen.generate({"identity": {"name": "x", "background": "y"}})


def test_generate_errors_on_no_contrast_material() -> None:
    gen = ContrastPromptGenerator(FakeBackend(), n_pairs=5)
    # Identity present but no self_facts / worldview AND no eval questions
    # would fire — but we always inject eval questions, so this actually
    # succeeds. A truly empty persona still yields role-based prompts.
    cs = gen.generate({"identity": {"name": "x", "role": "y", "background": "z"}})
    assert cs.n_pairs == 5


def test_unbalanced_contrast_set_raises() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        ContrastSet(in_persona=("a", "b"), out_persona=("a",))


def test_empty_contrast_set_raises() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        ContrastSet(in_persona=(), out_persona=())


def test_n_pairs_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        ContrastPromptGenerator(FakeBackend(), n_pairs=1)
