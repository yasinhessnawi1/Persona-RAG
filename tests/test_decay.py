"""Tests for the Ebbinghaus decay primitives."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from persona_rag.stores.decay import (
    DEFAULT_TAU,
    combine_similarity_and_decay,
    decay_score,
)

UTC_NOW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def test_default_tau_is_24h():
    assert timedelta(hours=24) == DEFAULT_TAU


def test_score_at_t0_is_one():
    assert decay_score(UTC_NOW, now=UTC_NOW) == pytest.approx(1.0)


def test_score_at_one_tau_is_one_over_e():
    t0 = UTC_NOW - DEFAULT_TAU
    assert decay_score(t0, now=UTC_NOW) == pytest.approx(math.exp(-1.0))


def test_score_in_future_clamped_to_one():
    future = UTC_NOW + timedelta(hours=1)
    assert decay_score(future, now=UTC_NOW) == 1.0


def test_score_decays_monotonically_with_age():
    ages_hours = [0, 1, 6, 12, 24, 48, 168]
    scores = [decay_score(UTC_NOW - timedelta(hours=h), now=UTC_NOW) for h in ages_hours]
    assert scores == sorted(scores, reverse=True)


def test_score_is_always_in_unit_interval():
    for h in [0, 1, 10, 100, 10000]:
        s = decay_score(UTC_NOW - timedelta(hours=h), now=UTC_NOW)
        assert 0.0 <= s <= 1.0


def test_custom_tau_changes_decay_speed():
    t0 = UTC_NOW - timedelta(hours=1)
    fast = decay_score(t0, now=UTC_NOW, tau=timedelta(minutes=10))
    slow = decay_score(t0, now=UTC_NOW, tau=timedelta(hours=24))
    assert fast < slow


def test_naive_datetime_rejected():
    naive = datetime(2026, 4, 24, 12, 0, 0)
    with pytest.raises(ValueError):
        decay_score(naive, now=UTC_NOW)


def test_zero_or_negative_tau_rejected():
    with pytest.raises(ValueError):
        decay_score(UTC_NOW, now=UTC_NOW, tau=timedelta(0))
    with pytest.raises(ValueError):
        decay_score(UTC_NOW, now=UTC_NOW, tau=timedelta(seconds=-1))


def test_combine_multiplicative():
    assert combine_similarity_and_decay(0.5, 0.5) == pytest.approx(0.25)
    assert combine_similarity_and_decay(1.0, 1.0) == pytest.approx(1.0)
    assert combine_similarity_and_decay(0.0, 1.0) == 0.0
    assert combine_similarity_and_decay(1.0, 0.0) == 0.0


def test_combine_rejects_out_of_range():
    with pytest.raises(ValueError):
        combine_similarity_and_decay(1.1, 0.5)
    with pytest.raises(ValueError):
        combine_similarity_and_decay(0.5, -0.1)
