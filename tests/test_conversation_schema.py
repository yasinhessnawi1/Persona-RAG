"""Tests for the drift-trajectory conversation schema."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from persona_rag.schema.conversation import (
    DriftTrajectoryConversation,
    assert_user_turns_match,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(dedent(body), encoding="utf-8")
    return path


_VALID_IN_PERSONA = """\
    persona_id: cs_tutor
    condition: in_persona
    n_pairs: 2
    turns:
      - role: user
        text: "Q1"
      - role: assistant
        text: "A1"
      - role: user
        text: "Q2"
      - role: assistant
        text: "A2"
"""

_VALID_DRIFTING = """\
    persona_id: cs_tutor
    condition: drifting
    n_pairs: 2
    turns:
      - role: user
        text: "Q1"
      - role: assistant
        text: "A1-drift"
        drift_level: in
      - role: user
        text: "Q2"
      - role: assistant
        text: "A2-drift"
        drift_level: clear
"""


def test_valid_in_persona_loads(tmp_path: Path) -> None:
    p = _write(tmp_path / "in.yaml", _VALID_IN_PERSONA)
    conv = DriftTrajectoryConversation.from_yaml(p)
    assert conv.condition == "in_persona"
    assert conv.n_pairs == 2
    assert len(conv.turns) == 4


def test_valid_drifting_loads_with_annotations(tmp_path: Path) -> None:
    p = _write(tmp_path / "drift.yaml", _VALID_DRIFTING)
    conv = DriftTrajectoryConversation.from_yaml(p)
    assert conv.condition == "drifting"
    assert [t.drift_level for t in conv.turns if t.role == "assistant"] == ["in", "clear"]


def test_wrong_turn_count_raises(tmp_path: Path) -> None:
    body = """\
        persona_id: x
        condition: in_persona
        n_pairs: 2
        turns:
          - role: user
            text: Q1
          - role: assistant
            text: A1
    """
    p = _write(tmp_path / "wrong.yaml", body)
    with pytest.raises(ValueError, match="must have 4 turns"):
        DriftTrajectoryConversation.from_yaml(p)


def test_role_alternation_enforced(tmp_path: Path) -> None:
    body = """\
        persona_id: x
        condition: in_persona
        n_pairs: 1
        turns:
          - role: assistant
            text: starts wrong
          - role: user
            text: this is wrong
    """
    p = _write(tmp_path / "bad_roles.yaml", body)
    with pytest.raises(ValueError, match="expected role 'user'"):
        DriftTrajectoryConversation.from_yaml(p)


def test_drifting_requires_drift_level_on_every_assistant_turn(tmp_path: Path) -> None:
    body = """\
        persona_id: x
        condition: drifting
        n_pairs: 2
        turns:
          - role: user
            text: Q1
          - role: assistant
            text: A1
            drift_level: in
          - role: user
            text: Q2
          - role: assistant
            text: A2
    """
    p = _write(tmp_path / "missing_drift.yaml", body)
    with pytest.raises(ValueError, match="every assistant turn must carry"):
        DriftTrajectoryConversation.from_yaml(p)


def test_in_persona_forbids_drift_level(tmp_path: Path) -> None:
    body = """\
        persona_id: x
        condition: in_persona
        n_pairs: 1
        turns:
          - role: user
            text: Q
          - role: assistant
            text: A
            drift_level: clear
    """
    p = _write(tmp_path / "wrong_anno.yaml", body)
    with pytest.raises(ValueError, match="must not carry"):
        DriftTrajectoryConversation.from_yaml(p)


def test_assert_user_turns_match_passes_for_aligned_pair(tmp_path: Path) -> None:
    a = _write(tmp_path / "in.yaml", _VALID_IN_PERSONA)
    b = _write(tmp_path / "drift.yaml", _VALID_DRIFTING)
    convs = [
        DriftTrajectoryConversation.from_yaml(a),
        DriftTrajectoryConversation.from_yaml(b),
    ]
    # Both share Q1/Q2 user turns — must pass.
    assert_user_turns_match(convs)


def test_assert_user_turns_match_raises_on_divergence(tmp_path: Path) -> None:
    drift_with_different_user = """\
        persona_id: cs_tutor
        condition: drifting
        n_pairs: 2
        turns:
          - role: user
            text: "Q1-changed"
          - role: assistant
            text: "A1"
            drift_level: in
          - role: user
            text: "Q2"
          - role: assistant
            text: "A2"
            drift_level: clear
    """
    a = _write(tmp_path / "in.yaml", _VALID_IN_PERSONA)
    b = _write(tmp_path / "drift_bad.yaml", drift_with_different_user)
    convs = [
        DriftTrajectoryConversation.from_yaml(a),
        DriftTrajectoryConversation.from_yaml(b),
    ]
    with pytest.raises(ValueError, match="user-turn mismatch"):
        assert_user_turns_match(convs)


def test_user_turn_texts_extracts_in_order(tmp_path: Path) -> None:
    p = _write(tmp_path / "in.yaml", _VALID_IN_PERSONA)
    conv = DriftTrajectoryConversation.from_yaml(p)
    assert conv.user_turn_texts() == ["Q1", "Q2"]
    assert conv.assistant_turn_texts() == ["A1", "A2"]
