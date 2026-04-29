"""Integration test for the on-disk pilot probe suite.

The pilot ships 5 conversations x 3 personas + 6 counter-evidence
chunks. This test asserts the suite loads end-to-end and that every
Type B probe references a chunk that's actually present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.benchmarks.counterfactual_probes import (
    load_counterfactual_probe_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT / "benchmarks_data" / "counterfactual_probes"


@pytest.fixture(scope="module")
def loaded_suite():
    if not SUITE_ROOT.exists():
        pytest.skip(f"counterfactual_probes suite not present under {SUITE_ROOT}")
    return load_counterfactual_probe_suite(SUITE_ROOT)


def test_pilot_loads_15_conversations(loaded_suite) -> None:
    conversations, _ = loaded_suite
    assert len(conversations) == 15, (
        f"expected 5/persona x 3 personas = 15 pilot conversations, got {len(conversations)}"
    )


def test_pilot_loads_six_chunks(loaded_suite) -> None:
    _, chunks = loaded_suite
    assert len(chunks) == 6, (
        f"expected 2 counter-evidence chunks per persona x 3 personas = 6, got {len(chunks)}"
    )


def test_pilot_probe_type_balance(loaded_suite) -> None:
    """Per-persona pilot is 2A / 2B / 1C as per the calibration design."""
    conversations, _ = loaded_suite
    by_persona: dict[str, dict[str, int]] = {}
    for conv in conversations:
        assert conv.probe is not None
        bucket = by_persona.setdefault(conv.persona_id, {})
        bucket[conv.probe.probe_type] = bucket.get(conv.probe.probe_type, 0) + 1
    assert set(by_persona) == {"cs_tutor", "historian", "climate_scientist"}
    for persona_id, counts in by_persona.items():
        assert counts.get("self_fact_challenge", 0) == 2, persona_id
        assert counts.get("counterfactual", 0) == 2, persona_id
        assert counts.get("constraint_bait", 0) == 1, persona_id


def test_pilot_type_b_probes_have_present_chunks(loaded_suite) -> None:
    """Loader raises ``KeyError`` if a Type-B probe references a missing chunk."""
    conversations, chunks = loaded_suite
    type_b = [
        c for c in conversations if c.probe is not None and c.probe.probe_type == "counterfactual"
    ]
    assert len(type_b) == 6
    for conv in type_b:
        assert conv.probe is not None
        assert conv.probe.injected_chunk_id in chunks


def test_pilot_user_turn_counts_in_range(loaded_suite) -> None:
    """Author guidance: 7-10 user turns per probe."""
    conversations, _ = loaded_suite
    for conv in conversations:
        assert 7 <= len(conv.user_turns) <= 10, (
            f"{conv.conversation_id}: {len(conv.user_turns)} user turns is out of [7,10]"
        )


def test_pilot_probe_turn_in_middle(loaded_suite) -> None:
    """Probes fire in the middle of the conversation, not at the boundaries."""
    conversations, _ = loaded_suite
    for conv in conversations:
        assert conv.probe is not None
        ix = conv.probe.probe_turn_index
        n = len(conv.user_turns)
        assert 1 <= ix < n - 1, (
            f"{conv.conversation_id}: probe at turn {ix} of {n} is too close to the edge"
        )
