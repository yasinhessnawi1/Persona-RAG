"""Human-validation pipeline: stratified sampling + CSV export + alpha computation.

The pipeline has three stages:

1. **Sample** -- pick N conversations stratified across mechanisms
   (target ~38 per mechanism for the full 150-item sweep). The 20-item
   pilot uses a 5-per-mechanism split.

2. **Export** -- write a blinded CSV: each row carries the conversation
   text + empty rubric columns the human fills in. Mechanism is
   replaced by an opaque token in the export (mapped back at scoring
   time) so the human can't anchor to mechanism identity.

3. **Score** -- read the human-filled CSV, join with the panel
   checkpoints, compute Krippendorff's alpha (per-judge and panel
   aggregate).

The 1-5 ordinal scale matches the PoLL panel rubric so alpha has a clean
common scale.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation
from persona_rag.evaluation.poll_panel import (
    JudgeCheckpoint,
    load_checkpoints_from_dir,
)
from persona_rag.evaluation.rubrics import PERSONA_ADHERENCE_DIMENSIONS

HUMAN_RUBRIC_DIMENSIONS: tuple[str, ...] = PERSONA_ADHERENCE_DIMENSIONS

EXPORT_HEADER: tuple[str, ...] = (
    "row_id",
    "blinded_token",
    "persona_id",
    "conversation_text",
    *(f"score_{dim}" for dim in HUMAN_RUBRIC_DIMENSIONS),
    "notes",
)


@dataclass(frozen=True, slots=True)
class SampledConversation:
    """One sampled item: opaque token + true mechanism (kept private)."""

    row_id: int
    blinded_token: str
    mechanism: str
    persona_id: str
    conversation_id: str
    conversation_text: str


def _conversation_to_text(conv: EvalConversation) -> str:
    """Render conversation as plain text for human reading."""
    lines: list[str] = []
    for i, turn in enumerate(conv.turns):
        lines.append(f"[Turn {i + 1}] USER: {turn.user_text}")
        lines.append(f"[Turn {i + 1}] ASSISTANT: {turn.assistant_text}")
        lines.append("")
    return "\n".join(lines)


def _make_blind_token(seed_text: str) -> str:
    """Short opaque token derived from a SHA hash, used as the human-facing label."""
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return f"item_{h[:8]}"


def stratified_sample(
    conversations_by_mechanism: dict[str, list[EvalConversation]],
    *,
    per_mechanism: int,
    seed: int,
) -> list[SampledConversation]:
    """Sample ``per_mechanism`` conversations from each mechanism bucket.

    Conversations within each bucket are randomly permuted under the
    given ``seed`` and the first ``per_mechanism`` are kept. Output
    items are interleaved across mechanisms (round-robin) so the human
    annotator does not see all of one mechanism's items consecutively
    (helps avoid order effects).
    """
    rng = random.Random(seed)
    picked: dict[str, list[EvalConversation]] = {}
    for mechanism, convs in conversations_by_mechanism.items():
        if len(convs) < per_mechanism:
            logger.warning(
                "human_validation: only {} conversations for mechanism {!r} (asked for {})",
                len(convs),
                mechanism,
                per_mechanism,
            )
        order = list(range(len(convs)))
        rng.shuffle(order)
        picked[mechanism] = [convs[i] for i in order[:per_mechanism]]

    # Round-robin interleave.
    interleaved: list[SampledConversation] = []
    max_n = max(len(v) for v in picked.values()) if picked else 0
    row_id = 0
    for i in range(max_n):
        for mechanism in sorted(picked.keys()):
            if i >= len(picked[mechanism]):
                continue
            conv = picked[mechanism][i]
            blinded = _make_blind_token(f"{seed}::{conv.conversation_id}::{i}")
            interleaved.append(
                SampledConversation(
                    row_id=row_id,
                    blinded_token=blinded,
                    mechanism=conv.mechanism,
                    persona_id=conv.persona_id,
                    conversation_id=conv.conversation_id,
                    conversation_text=_conversation_to_text(conv),
                )
            )
            row_id += 1
    return interleaved


def export_csv(
    items: list[SampledConversation],
    csv_path: Path,
    mapping_path: Path,
) -> None:
    """Write the blinded human-rating CSV plus the private blinded->mechanism map.

    The CSV is what the human sees: blinded tokens, conversation text,
    empty score columns. The mapping file is private and lets us rejoin
    the human scores to the mechanism.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EXPORT_HEADER))
        writer.writeheader()
        for item in items:
            row: dict[str, Any] = {
                "row_id": item.row_id,
                "blinded_token": item.blinded_token,
                "persona_id": item.persona_id,
                "conversation_text": item.conversation_text,
                "notes": "",
            }
            for dim in HUMAN_RUBRIC_DIMENSIONS:
                row[f"score_{dim}"] = ""
            writer.writerow(row)

    mapping = {
        item.blinded_token: {
            "row_id": item.row_id,
            "mechanism": item.mechanism,
            "persona_id": item.persona_id,
            "conversation_id": item.conversation_id,
        }
        for item in items
    }
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")


@dataclass
class HumanScoreRow:
    """One human-filled rating row, post-deblinding."""

    row_id: int
    blinded_token: str
    mechanism: str
    persona_id: str
    conversation_id: str
    scores: dict[str, int] = field(default_factory=dict)
    notes: str = ""


def load_human_csv(
    csv_path: Path,
    mapping_path: Path,
) -> list[HumanScoreRow]:
    """Read the human-filled CSV + rejoin against the blinded->mechanism map.

    Rows where every score column is empty are skipped (annotator did
    not finish them). Rows with malformed scores raise.
    """
    mapping: dict[str, dict[str, Any]] = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows: list[HumanScoreRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            scores: dict[str, int] = {}
            for dim in HUMAN_RUBRIC_DIMENSIONS:
                v = (raw.get(f"score_{dim}") or "").strip()
                if v == "":
                    continue
                try:
                    iv = int(float(v))
                except ValueError as exc:
                    raise ValueError(
                        f"row {raw.get('row_id')}: non-integer score for {dim} ({v!r})"
                    ) from exc
                if not 1 <= iv <= 5:
                    raise ValueError(
                        f"row {raw.get('row_id')}: score for {dim} out of [1,5] ({iv})"
                    )
                scores[dim] = iv
            if not scores:
                continue
            blinded = (raw.get("blinded_token") or "").strip()
            meta = mapping.get(blinded)
            if meta is None:
                raise ValueError(f"blinded_token {blinded!r} not in mapping file")
            rows.append(
                HumanScoreRow(
                    row_id=int(raw["row_id"]),
                    blinded_token=blinded,
                    mechanism=meta["mechanism"],
                    persona_id=meta["persona_id"],
                    conversation_id=meta["conversation_id"],
                    scores=scores,
                    notes=str(raw.get("notes", "")),
                )
            )
    return rows


def alpha_against_panel(
    human_rows: list[HumanScoreRow],
    panel_checkpoints: dict[str, JudgeCheckpoint],
    *,
    rubric: Literal["persona_adherence"] = "persona_adherence",
) -> dict[str, float]:
    """Compute Krippendorff's alpha: human vs each panel judge + vs panel aggregate.

    Returns ``{"vs_<judge>": float, ..., "vs_panel": float}``. Requires
    ``krippendorff`` to be installed.
    """
    try:
        import krippendorff
    except ImportError as exc:
        raise RuntimeError(
            "krippendorff not installed -- install with: uv pip install krippendorff"
        ) from exc

    # Build lookup: conversation_id -> human overall_mean.
    human_overall: dict[str, float] = {}
    for row in human_rows:
        if not row.scores:
            continue
        # Use the mean of the four declared dimensions (matches the
        # PoLL panel's overall_mean computation).
        present = [v for k, v in row.scores.items() if k in HUMAN_RUBRIC_DIMENSIONS]
        if present:
            human_overall[row.conversation_id] = statistics.fmean(present)

    item_ids = sorted(human_overall.keys())
    if not item_ids:
        return {}

    out: dict[str, float] = {}

    # Per-judge alpha.
    panel_per_item: dict[str, list[float]] = defaultdict(list)
    for judge_name, ckpt in panel_checkpoints.items():
        judge_lookup: dict[str, float] = {}
        for s in ckpt.scores:
            if rubric == "persona_adherence":
                judge_lookup[s.conversation_id] = s.persona_adherence.overall_mean
            else:
                judge_lookup[s.conversation_id] = float(s.task_quality.score)
        matrix = [
            [human_overall.get(item, float("nan")) for item in item_ids],
            [judge_lookup.get(item, float("nan")) for item in item_ids],
        ]
        try:
            alpha = float(
                krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
            )
        except Exception as exc:
            logger.warning("alpha vs {} failed: {}", judge_name, exc)
            alpha = float("nan")
        out[f"vs_{judge_name}"] = alpha

        for item in item_ids:
            v = judge_lookup.get(item)
            if v is not None:
                panel_per_item[item].append(v)

    # Panel aggregate vs human.
    panel_aggregate = [
        statistics.fmean(panel_per_item[item]) if panel_per_item[item] else float("nan")
        for item in item_ids
    ]
    matrix = [
        [human_overall[item] for item in item_ids],
        panel_aggregate,
    ]
    try:
        alpha_panel = float(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
        )
    except Exception as exc:
        logger.warning("alpha vs panel aggregate failed: {}", exc)
        alpha_panel = float("nan")
    out["vs_panel"] = alpha_panel
    return out


def write_alpha_report(
    out_path: Path,
    alphas: dict[str, float],
    *,
    n_human_items: int,
    panel_dir: Path,
) -> None:
    """Render a small markdown report next to the panel checkpoint dir."""
    lines: list[str] = [
        "# Human-validation alpha report",
        f"n_human_items: {n_human_items}",
        f"panel_dir: {panel_dir}",
        "",
        "## Krippendorff's alpha (ordinal)",
    ]
    for k, v in alphas.items():
        if math.isnan(v):
            lines.append(f"- {k}: nan")
        else:
            lines.append(f"- {k}: alpha = {v:.3f}")
    target = 0.5
    panel_alpha = alphas.get("vs_panel", float("nan"))
    if not math.isnan(panel_alpha):
        if panel_alpha >= target:
            lines.append(f"\nverdict: GREEN -- panel alpha {panel_alpha:.3f} ≥ target {target}")
        elif panel_alpha >= 0.3:
            lines.append(
                f"\nverdict: YELLOW -- panel alpha {panel_alpha:.3f} below target {target}; report as limitation"
            )
        else:
            lines.append(
                f"\nverdict: RED -- panel alpha {panel_alpha:.3f} below 0.3; rubric needs revision"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_panel(panel_dir: Path) -> dict[str, JudgeCheckpoint]:
    """Convenience re-export so the human-validation script does not need to know the path layout."""
    return load_checkpoints_from_dir(panel_dir)


__all__ = [
    "EXPORT_HEADER",
    "HUMAN_RUBRIC_DIMENSIONS",
    "HumanScoreRow",
    "SampledConversation",
    "alpha_against_panel",
    "export_csv",
    "load_human_csv",
    "load_panel",
    "stratified_sample",
    "write_alpha_report",
]
