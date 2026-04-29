"""Tests for the PersonaGym free-form-string → typed-Persona mapper.

Focus: invariants over the lossy mapping. The mapper preserves the source
text in ``identity.background`` (so the report can show reviewers what
PersonaGym actually shipped), splits worldview-claim verbs out as belief
entries, and produces non-empty self-fact lists for typical persona
strings. We do not assert exact natural-language matches — those are
brittle — but we assert structural invariants.

The full sampler test exercises ``load_personagym`` against a small
synthetic on-disk fixture (no network, no real PersonaGym dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_rag.benchmarks.personagym import (
    PERSONAGYM_TASKS,
    load_personagym,
    map_persona_string_to_typed,
)


def test_mapper_preserves_source_text_in_background() -> None:
    text = (
        "A 71-year-old retired nurse from Italy, volunteering in hospice care "
        "and advocating for compassionate end-of-life support"
    )
    persona = map_persona_string_to_typed(text, sequence_number=0)
    assert persona.identity.background.startswith(text[:50])
    assert persona.identity.role.startswith("A 71-year-old retired nurse from Italy")


def test_mapper_extracts_worldview_from_advocacy_verb() -> None:
    text = (
        "A 36-year-old environmental lawyer from Australia, fighting against "
        "illegal deforestation and protecting indigenous lands"
    )
    persona = map_persona_string_to_typed(text)
    domains = {w.domain for w in persona.worldview}
    assert "activism" in domains
    # Worldview claim text should reference the verb phrase.
    claim_texts = " ".join(w.claim for w in persona.worldview)
    assert "fighting against illegal deforestation" in claim_texts.lower()


def test_mapper_handles_persona_without_advocacy_verbs() -> None:
    text = "A 32-year-old writer from Sydney who loves dancing"
    persona = map_persona_string_to_typed(text)
    assert persona.worldview == []
    # Self-facts are non-empty.
    assert len(persona.self_facts) >= 1


def test_mapper_constraints_always_empty() -> None:
    """PersonaGym does not supply constraints; the typed mapping must reflect that."""
    text = (
        "A 19-year-old college student from California, majoring in environmental "
        "science and passionate about combating climate change"
    )
    persona = map_persona_string_to_typed(text)
    assert persona.identity.constraints == []


def test_mapper_persona_id_slugify() -> None:
    text = "A 19-year-old college student from California, majoring in env science"
    persona = map_persona_string_to_typed(text)
    assert persona.persona_id is not None
    # Slug-only chars: alnum + underscore + hyphen.
    assert all(c.isalnum() or c in {"_", "-"} for c in persona.persona_id)


def test_mapper_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        map_persona_string_to_typed("")


def test_load_personagym_against_synthetic_fixture(tmp_path: Path) -> None:
    """End-to-end: tiny fixture root, deterministic sample, balanced task draw."""
    root = tmp_path / "pg"
    root.mkdir()
    persona_strings = [
        "A baker from Lyon who hates marathons.",
        "A 30-year-old climber from Nepal, advocating for fair guide pay.",
        "A 22-year-old DJ from Berlin who collects vinyl.",
    ]
    (root / "personas.json").write_text(json.dumps(persona_strings), encoding="utf-8")
    qdir = root / "questions"
    qdir.mkdir()
    for s in persona_strings:
        # Two questions per task — enough to exercise round-robin sampling.
        payload = {task: [f"{task} q1 for {s[:10]}", f"{task} q2"] for task in PERSONAGYM_TASKS}
        (qdir / f"{s}.json").write_text(json.dumps(payload), encoding="utf-8")

    personas, conversations = load_personagym(
        root,
        n_personas=2,
        n_questions_per_persona=5,
        seed=0,
    )
    assert len(personas) == 2
    assert len(conversations) == 10  # 2 personas x 5 questions each
    # Every conversation is single-turn, no probe.
    for conv in conversations:
        assert conv.probe is None
        assert len(conv.user_turns) == 1
    # Tasks should be balanced (round-robin); first 5 conversations cover all
    # five tasks before the second persona begins.
    tasks_first_persona = [c.notes for c in conversations[:5]]
    assert {n.split("=")[1] for n in tasks_first_persona} == set(PERSONAGYM_TASKS)


def test_load_personagym_seed_determinism(tmp_path: Path) -> None:
    """Two calls with the same seed must select the same personas."""
    root = tmp_path / "pg2"
    root.mkdir()
    strings = [f"A {i}-year-old test persona who likes hiking." for i in range(20, 30)]
    (root / "personas.json").write_text(json.dumps(strings), encoding="utf-8")
    qdir = root / "questions"
    qdir.mkdir()
    for s in strings:
        (qdir / f"{s}.json").write_text(
            json.dumps({task: ["q"] for task in PERSONAGYM_TASKS}), encoding="utf-8"
        )

    personas_a, conv_a = load_personagym(root, n_personas=3, seed=99)
    personas_b, conv_b = load_personagym(root, n_personas=3, seed=99)
    assert [p.persona_id for p in personas_a] == [p.persona_id for p in personas_b]
    assert [c.conversation_id for c in conv_a] == [c.conversation_id for c in conv_b]
