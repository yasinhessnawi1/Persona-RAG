"""Transcript loaders: turn run-output JSONs into ``EvalConversation`` objects.

The harness consumes conversations as ``EvalConversation`` (one per
unit of mechanism + persona + seed). Two on-disk shapes are supported:

- **Single-turn baseline output** -- the ``run_baseline.py`` per-query
  ``response_NN.json`` files. Each file becomes a 1-turn
  conversation; every file in the directory is one conversation.
- **Multi-turn conversation YAML** -- the
  ``DriftTrajectoryConversation`` schema. Each file becomes one
  conversation containing N (user, assistant) turns.

Both loaders attach ``mechanism`` + ``persona_id`` to each
``EvalConversation`` so the metric output can be joined back to the
source pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation, ScoredTurn
from persona_rag.schema.conversation import DriftTrajectoryConversation


def load_baseline_response_dir(
    dir_path: Path,
    *,
    mechanism: str,
    persona_id: str,
    glob: str = "response_*.json",
) -> list[EvalConversation]:
    """Load every per-query response JSON under ``dir_path`` as a 1-turn conversation.

    ``conversation_id`` is the JSON file stem. The user turn comes from
    the JSON's ``query`` field; the assistant turn from ``text``. Per-turn
    metadata captures ``bucket`` and ``seed`` if present so downstream
    slicing on those is possible.
    """
    dir_path = Path(dir_path)
    paths = sorted(dir_path.glob(glob))
    if not paths:
        logger.warning("transcripts: no files matched {} under {}", glob, dir_path)
        return []
    convs: list[EvalConversation] = []
    for p in paths:
        payload: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        user_text = str(payload.get("query", ""))
        assistant_text = str(payload.get("text", ""))
        if not user_text or not assistant_text:
            logger.warning("transcripts: skipping {} (missing query or text)", p)
            continue
        turn = ScoredTurn(turn_index=0, user_text=user_text, assistant_text=assistant_text)
        per_turn_meta: dict[str, Any] = {}
        for key in ("bucket", "seed", "skip_in_matrix", "multi_seed", "drift_signal", "metadata"):
            if key in payload:
                per_turn_meta[key] = payload[key]
        convs.append(
            EvalConversation(
                conversation_id=f"{persona_id}::{mechanism}::{p.stem}",
                mechanism=mechanism,
                persona_id=persona_id,
                turns=(turn,),
                per_turn_metadata=(per_turn_meta,),
            )
        )
    logger.info("transcripts: loaded {} conversations from {}", len(convs), dir_path)
    return convs


def conversation_yaml_to_eval(
    yaml_path: Path,
    *,
    mechanism: str,
    persona_id: str | None = None,
    conversation_id: str | None = None,
) -> EvalConversation:
    """Load a hand-authored conversation YAML as an ``EvalConversation``.

    ``persona_id`` defaults to the YAML's own ``persona_id`` field;
    ``conversation_id`` defaults to ``"<persona_id>::<mechanism>::<file_stem>"``.
    Per-turn metadata records the YAML's ``drift_level`` annotation when
    present (drifting conversations only).
    """
    conv = DriftTrajectoryConversation.from_yaml(yaml_path)
    persona = persona_id or conv.persona_id
    cid = conversation_id or f"{persona}::{mechanism}::{Path(yaml_path).stem}"

    pairs: list[ScoredTurn] = []
    per_turn_meta: list[dict[str, Any]] = []
    user_buf: str | None = None
    user_meta: dict[str, Any] = {}
    asst_idx = 0
    for turn in conv.turns:
        if turn.role == "user":
            user_buf = turn.text
            user_meta = {"condition": conv.condition}
        else:
            assert user_buf is not None  # schema guarantees alternation
            pairs.append(
                ScoredTurn(
                    turn_index=asst_idx,
                    user_text=user_buf,
                    assistant_text=turn.text,
                )
            )
            meta = {**user_meta}
            if turn.drift_level is not None:
                meta["drift_level"] = turn.drift_level
            per_turn_meta.append(meta)
            asst_idx += 1
            user_buf = None

    return EvalConversation(
        conversation_id=cid,
        mechanism=mechanism,
        persona_id=persona,
        turns=tuple(pairs),
        per_turn_metadata=tuple(per_turn_meta),
    )


def load_conversation_yamls(
    paths: Iterable[Path],
    *,
    mechanism: str,
) -> list[EvalConversation]:
    """Load multiple conversation YAMLs at once."""
    return [conversation_yaml_to_eval(p, mechanism=mechanism) for p in paths]


def load_m3_records_json(
    records_path: Path,
    *,
    mechanism: str,
    persona_id: str,
) -> list[EvalConversation]:
    """Load the records-bundle JSON shape produced by ``run_m3_vs_baselines_pilot.py``.

    The file is a list of records, one per query. Each record carries:

    - ``query_id``: stable id for the query.
    - ``query``: the user turn text.
    - ``label``: bucket label (e.g. ``drift_triggering``, ``in_persona_extra``).
    - ``by_pipeline``: ``{pipeline_name: {text, metadata}, ...}`` covering every
      mechanism that was run on this query (B1, B2, M1, M3, ...).

    This loader pulls out the assistant turn for *one* mechanism (matched
    against the ``by_pipeline`` keys case-insensitively, so ``"m3"`` finds
    ``"M3"``). Each record becomes one 1-turn ``EvalConversation``.
    """
    records_path = Path(records_path)
    raw = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{records_path}: expected a list at top-level, got {type(raw).__name__}")
    target = mechanism.lower()
    convs: list[EvalConversation] = []
    skipped = 0
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        by_pipeline = rec.get("by_pipeline") or {}
        match_key = next(
            (k for k in by_pipeline if k.lower() == target),
            None,
        )
        if match_key is None:
            skipped += 1
            continue
        pipeline_payload = by_pipeline[match_key] or {}
        user_text = str(rec.get("query", "")).strip()
        assistant_text = str(pipeline_payload.get("text", "")).strip()
        if not user_text or not assistant_text:
            skipped += 1
            continue
        per_turn_meta: dict[str, Any] = {
            "label": rec.get("label"),
            "query_id": rec.get("query_id"),
        }
        pipeline_meta = pipeline_payload.get("metadata")
        if isinstance(pipeline_meta, dict):
            per_turn_meta["metadata"] = pipeline_meta
        turn = ScoredTurn(
            turn_index=0,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        convs.append(
            EvalConversation(
                conversation_id=f"{persona_id}::{mechanism}::{rec.get('query_id', f'r{len(convs)}')}",
                mechanism=mechanism,
                persona_id=persona_id,
                turns=(turn,),
                per_turn_metadata=(per_turn_meta,),
            )
        )
    if skipped:
        logger.warning(
            "transcripts: skipped {} records (no '{}' pipeline or empty text) in {}",
            skipped,
            mechanism,
            records_path,
        )
    logger.info(
        "transcripts: loaded {} {} conversations from {}",
        len(convs),
        mechanism,
        records_path,
    )
    return convs


__all__ = [
    "conversation_yaml_to_eval",
    "load_baseline_response_dir",
    "load_conversation_yamls",
    "load_m3_records_json",
]
