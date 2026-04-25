"""Worldview store: epistemic-tagged, bi-temporal-filterable claims."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from persona_rag.schema.chunker import PersonaChunk
from persona_rag.stores.base import TypedMemoryStore

_VALID_TIME_POINT_RE = re.compile(r"^\d{4}$")
_VALID_TIME_RANGE_RE = re.compile(r"^(\d{4})-(\d{4})$")
_VALID_TIME_OPEN_RE = re.compile(r"^(\d{4})-$")


class WorldviewStore(TypedMemoryStore):
    """Stores `worldview` chunks. Supports epistemic-tag and bi-temporal filtering."""

    COLLECTION_NAME = "persona_worldview"
    ALLOW_RUNTIME_WRITE = False

    def _accepts_kind(self, kind: str) -> bool:
        return kind == "worldview"

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        persona_id: str | None = None,
        epistemic: str | Iterable[str] | None = None,
        as_of: str | None = None,
        extra_where: dict[str, Any] | None = None,
    ) -> list[PersonaChunk]:
        """Top-k query with optional epistemic-tag and as_of-year filters.

        `epistemic`: a single tag (e.g. `"fact"`) or an iterable of tags — only
        chunks with matching epistemic metadata are considered.
        `as_of`: a four-digit year string. Chunks whose `valid_time` range does
        not cover that year are excluded.
        """
        # --- epistemic filter ---
        clauses: list[dict[str, Any]] = []
        if isinstance(epistemic, str):
            clauses.append({"epistemic": epistemic})
        elif epistemic is not None:
            epi_list = list(epistemic)
            if epi_list:
                clauses.append({"epistemic": {"$in": epi_list}})
        if extra_where:
            clauses.append(extra_where)

        combined_extra: dict[str, Any] | None
        if not clauses:
            combined_extra = None
        elif len(clauses) == 1:
            combined_extra = clauses[0]
        else:
            combined_extra = {"$and": clauses}

        # Fetch without valid_time filter (ChromaDB can't parse ranges itself),
        # then post-filter in Python.
        candidates = super().query(
            text,
            top_k=top_k if as_of is None else max(top_k * 3, top_k),
            persona_id=persona_id,
            extra_where=combined_extra,
        )
        if as_of is None:
            return candidates

        as_of_year = _require_year(as_of)
        kept = [
            c
            for c in candidates
            if _matches_as_of(c.metadata.get("valid_time", "always"), as_of_year)
        ]
        return kept[:top_k]


def _require_year(value: str) -> int:
    m = _VALID_TIME_POINT_RE.match(value)
    if not m:
        raise ValueError(f"as_of must be a four-digit year string (e.g. '1880'); got {value!r}")
    return int(value)


def _matches_as_of(valid_time: str, as_of_year: int) -> bool:
    """True iff `valid_time` covers `as_of_year`.

    Grammar: 'always' | 'YYYY' | 'YYYY-YYYY' | 'YYYY-'. Anything else returns
    False to fail closed.
    """
    if valid_time == "always":
        return True
    if _VALID_TIME_POINT_RE.match(valid_time):
        return int(valid_time) == as_of_year

    m = _VALID_TIME_RANGE_RE.match(valid_time)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return start <= as_of_year <= end

    m = _VALID_TIME_OPEN_RE.match(valid_time)
    if m:
        return int(m.group(1)) <= as_of_year

    return False
