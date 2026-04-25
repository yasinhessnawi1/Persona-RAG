"""Ebbinghaus forgetting-curve decay for episodic memory ranking."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

DEFAULT_TAU = timedelta(hours=24)


def decay_score(
    decay_t0: datetime,
    now: datetime | None = None,
    tau: timedelta = DEFAULT_TAU,
) -> float:
    """Ebbinghaus decay: exp(-(now - decay_t0) / tau).

    Returned value is in (0.0, 1.0]. If `decay_t0` is in the future (clock skew
    or explicit scheduling), delta is clamped at 0 and the score is 1.0.
    `now` defaults to the current UTC time; callers can pin it for
    reproducibility in tests.
    """
    if tau.total_seconds() <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if now is None:
        now = datetime.now(UTC)
    _check_tz_aware("decay_t0", decay_t0)
    _check_tz_aware("now", now)

    delta = (now - decay_t0).total_seconds()
    if delta <= 0.0:
        return 1.0
    return math.exp(-delta / tau.total_seconds())


def combine_similarity_and_decay(similarity: float, decay: float) -> float:
    """Multiplicative combination: both inputs assumed to live in [0, 1]."""
    if not (0.0 <= similarity <= 1.0):
        raise ValueError(f"similarity must be in [0, 1], got {similarity}")
    if not (0.0 <= decay <= 1.0 + 1e-9):
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    return similarity * decay


def _check_tz_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
