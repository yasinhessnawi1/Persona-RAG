"""PersonaChat loader: helper-level tests + end-to-end with a fake dataset.

The full ``load_personachat`` path imports ``datasets``; we stub it out
with a fake module to avoid the ~100MB download in unit tests. The
``_build_persona_from_traits`` and ``_select_final_user_turn`` helpers
are tested directly.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from persona_rag.benchmarks import personachat as pc


def test_select_final_user_turn_returns_last_history_item() -> None:
    history = ["hi there", "how are you", "fine thanks"]
    assert pc._select_final_user_turn(history) == "fine thanks"


def test_select_final_user_turn_empty_history_returns_none() -> None:
    assert pc._select_final_user_turn([]) is None


def test_select_final_user_turn_strips_whitespace() -> None:
    assert pc._select_final_user_turn(["a", "  b  "]) == "b"


def test_select_final_user_turn_blank_string_returns_none() -> None:
    assert pc._select_final_user_turn(["a", "   "]) is None


def test_build_persona_from_traits_maps_to_self_facts() -> None:
    traits = ["i love dogs", "i am a baker", "i live in Lyon"]
    persona = pc._build_persona_from_traits(traits, conv_id=42)
    assert persona.persona_id == "personachat_000042"
    assert len(persona.self_facts) == 3
    assert persona.worldview == []
    assert persona.identity.constraints == []
    # Traits are preserved verbatim in self_facts.
    fact_texts = {f.fact for f in persona.self_facts}
    assert fact_texts == set(traits)


def test_build_persona_from_traits_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty trait list"):
        pc._build_persona_from_traits([], conv_id=1)


def test_load_personachat_uses_fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the ``datasets`` import so we exercise the loader without a real download."""

    class _FakeDataset(list):
        pass

    rows: list[dict[str, Any]] = [
        # conv 1 has two utterances; loader picks the later (utterance_idx=1).
        {
            "conv_id": 1,
            "utterance_idx": 0,
            "personality": ["i like cats"],
            "history": ["hi"],
            "candidates": ["x"],
        },
        {
            "conv_id": 1,
            "utterance_idx": 1,
            "personality": ["i like cats"],
            "history": ["hi", "how was your day"],
            "candidates": ["x"],
        },
        {
            "conv_id": 2,
            "utterance_idx": 0,
            "personality": ["i bake bread", "i live in Paris"],
            "history": ["bonjour", "what brings you here"],
            "candidates": ["x"],
        },
        # conv 3 has empty history → must be skipped.
        {
            "conv_id": 3,
            "utterance_idx": 0,
            "personality": ["i collect stamps"],
            "history": [],
            "candidates": ["x"],
        },
    ]

    fake_dataset = _FakeDataset(rows)

    class _FakeDatasetsModule:
        @staticmethod
        def load_dataset(*_a: Any, **_kw: Any) -> _FakeDataset:
            return fake_dataset

    monkeypatch.setitem(sys.modules, "datasets", _FakeDatasetsModule())

    personas, conversations = pc.load_personachat(n_conversations=10, seed=0)
    # 2 valid conversations (conv 1 picks utterance_idx=1; conv 2; conv 3 is skipped).
    assert len(personas) == 2
    assert len(conversations) == 2
    persona_ids = {p.persona_id for p in personas}
    assert persona_ids == {"personachat_000001", "personachat_000002"}
    # conv 1's chosen user turn is from the latest utterance_idx row.
    conv_1 = next(c for c in conversations if "000001" in c.conversation_id)
    assert conv_1.user_turns == ["how was your day"]
    # All conversations are single-turn with no probe.
    for c in conversations:
        assert len(c.user_turns) == 1
        assert c.probe is None
        assert c.benchmark == "personachat"
