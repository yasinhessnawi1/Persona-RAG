"""Tests for the generator's prompt-style registry."""

from __future__ import annotations

import pytest

from option8_rag.synthesize.generator import (
    SYSTEM_PROMPT_EXTRACTIVE,
    SYSTEM_PROMPT_VERBOSE,
    get_system_prompt,
)


def test_verbose_and_extractive_are_distinct() -> None:
    assert get_system_prompt("verbose") == SYSTEM_PROMPT_VERBOSE
    assert get_system_prompt("extractive") == SYSTEM_PROMPT_EXTRACTIVE
    assert SYSTEM_PROMPT_VERBOSE != SYSTEM_PROMPT_EXTRACTIVE


def test_extractive_prompt_demands_short_answers() -> None:
    body = SYSTEM_PROMPT_EXTRACTIVE.lower()
    assert "few words" in body
    assert "do not write a sentence" in body
    assert "i don't know" in body


def test_unknown_style_raises() -> None:
    with pytest.raises(ValueError):
        get_system_prompt("nonsense")
