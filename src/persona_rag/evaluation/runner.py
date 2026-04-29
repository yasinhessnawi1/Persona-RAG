"""Evaluation runner: drive metrics across (mechanism, persona, seed) cells.

The runner takes:

- A bundle of conversations grouped by (mechanism, persona). Each
  conversation already carries a ``seed`` (in metadata) if relevant.
- A list of metrics to score.
- An output directory.

It emits:

- ``results.csv`` -- long-form: one row per
  (mechanism, persona, metric, conversation, seed).
- ``results_aggregate.csv`` -- wide: one row per (mechanism, persona,
  metric) with the headline aggregate.
- ``results.json`` -- machine-readable bundle.
- ``run_config.json`` -- the inputs (config, mechanisms, metrics) for
  reproducibility audits.
- Optional wandb logging of headline aggregates.

Reproducibility: the runner does not generate; it only scores already-
generated transcripts. Determinism therefore reduces to "do the metrics
themselves return the same values on the same inputs across runs".
That property is enforced by metric-level tests (greedy decoding, fixed
random seeds where stochasticity is unavoidable).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from persona_rag.evaluation.metrics import EvalConversation, Metric, MetricResult
from persona_rag.schema.persona import Persona

LONG_FORM_HEADER: tuple[str, ...] = (
    "run_id",
    "mechanism",
    "model",
    "benchmark",
    "persona_id",
    "metric_name",
    "conversation_id",
    "value",
    "seed",
)

AGGREGATE_HEADER: tuple[str, ...] = (
    "run_id",
    "mechanism",
    "model",
    "benchmark",
    "persona_id",
    "metric_name",
    "value",
    "n_conversations",
)


@dataclass(frozen=True, slots=True)
class MechanismCell:
    """One (mechanism, persona) bundle of conversations to score."""

    mechanism: str
    model: str  # backbone identifier, e.g. "gemma2-9b-it"
    benchmark: str  # benchmark id (e.g. "drift_trajectory")
    persona: Persona
    conversations: list[EvalConversation]
    seed: int = 0


@dataclass
class EvaluationRunner:
    """Drive metrics across (mechanism, persona) cells.

    ``run_id`` is generated from a timestamp at construction; pass an
    explicit one for tests.
    """

    output_dir: Path
    metrics: list[Metric]
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42
    wandb_enabled: bool = False
    wandb_project: str = "persona-rag"
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("EvaluationRunner needs at least one metric.")
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass

    def run(self, cells: list[MechanismCell]) -> dict[str, list[MetricResult]]:
        """Score every metric over every cell. Write CSV + JSON outputs."""
        if not cells:
            raise ValueError("EvaluationRunner.run: no cells supplied.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config(cells)

        # Cell-keyed results: {(mechanism, persona_id): [MetricResult, ...]}
        cell_results: dict[tuple[str, str], list[MetricResult]] = {}
        long_rows: list[dict[str, Any]] = []
        agg_rows: list[dict[str, Any]] = []

        for cell in cells:
            key = (cell.mechanism, cell.persona.persona_id or "<unknown>")
            cell_results.setdefault(key, [])
            for metric in self.metrics:
                # Per-mechanism metrics (e.g. ``CostTracker(mechanism=...)``)
                # carry a ``mechanism`` attribute; only run them against
                # cells whose mechanism matches. Without this, every cell
                # gets scored by every per-mechanism tracker, producing
                # bogus duplicate cost rows in the CSV.
                metric_mechanism = getattr(metric, "mechanism", None)
                if metric_mechanism is not None and metric_mechanism != cell.mechanism:
                    continue
                logger.info(
                    "runner: cell {} x metric {} ({} conversations)",
                    key,
                    metric.name,
                    len(cell.conversations),
                )
                result = metric.score(cell.conversations, cell.persona)
                cell_results[key].append(result)
                # Long-form rows: one per conversation x metric.
                if result.per_conversation:
                    for conv_id, value in zip(
                        result.per_conversation_ids,
                        result.per_conversation,
                        strict=True,
                    ):
                        long_rows.append(
                            {
                                "run_id": self.run_id,
                                "mechanism": cell.mechanism,
                                "model": cell.model,
                                "benchmark": cell.benchmark,
                                "persona_id": key[1],
                                "metric_name": result.name,
                                "conversation_id": conv_id,
                                "value": value,
                                "seed": cell.seed,
                            }
                        )
                agg_rows.append(
                    {
                        "run_id": self.run_id,
                        "mechanism": cell.mechanism,
                        "model": cell.model,
                        "benchmark": cell.benchmark,
                        "persona_id": key[1],
                        "metric_name": result.name,
                        "value": result.value,
                        "n_conversations": len(cell.conversations),
                    }
                )

        self._write_csv(self.output_dir / "results.csv", LONG_FORM_HEADER, long_rows)
        self._write_csv(self.output_dir / "results_aggregate.csv", AGGREGATE_HEADER, agg_rows)
        self._write_json_bundle(cell_results)

        if self.wandb_enabled:
            self._log_wandb(cells, cell_results)
        return {f"{k[0]}::{k[1]}": v for k, v in cell_results.items()}

    # ---------------------------------------------------------------- IO

    def _write_run_config(self, cells: list[MechanismCell]) -> None:
        config_path = self.output_dir / "run_config.json"
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "wandb": {
                "enabled": self.wandb_enabled,
                "project": self.wandb_project,
                "mode": self.wandb_mode,
            },
            "metrics": [m.name for m in self.metrics],
            "cells": [
                {
                    "mechanism": c.mechanism,
                    "model": c.model,
                    "benchmark": c.benchmark,
                    "persona_id": c.persona.persona_id,
                    "n_conversations": len(c.conversations),
                    "seed": c.seed,
                }
                for c in cells
            ],
        }
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_csv(
        path: Path,
        header: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(header))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in header})

    def _write_json_bundle(
        self,
        cell_results: dict[tuple[str, str], list[MetricResult]],
    ) -> None:
        path = self.output_dir / "results.json"
        bundle: dict[str, Any] = {"run_id": self.run_id, "cells": []}
        for (mechanism, persona_id), results in cell_results.items():
            bundle["cells"].append(
                {
                    "mechanism": mechanism,
                    "persona_id": persona_id,
                    "metrics": [r.model_dump() for r in results],
                }
            )
        path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    def _log_wandb(
        self,
        cells: list[MechanismCell],
        cell_results: dict[tuple[str, str], list[MetricResult]],
    ) -> None:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb requested but not installed -- skipping logging")
            return
        mode = os.environ.get("WANDB_MODE") or self.wandb_mode
        run = wandb.init(
            project=self.wandb_project,
            mode=mode,
            name=f"eval_{self.run_id}",
            dir=str(self.output_dir),
            config={
                "run_id": self.run_id,
                "n_cells": len(cells),
                "metrics": [m.name for m in self.metrics],
            },
        )
        for (mechanism, persona_id), results in cell_results.items():
            for result in results:
                run.log(
                    {
                        f"{mechanism}/{persona_id}/{result.name}": result.value,
                    }
                )
        run.finish()


__all__ = [
    "AGGREGATE_HEADER",
    "LONG_FORM_HEADER",
    "EvaluationRunner",
    "MechanismCell",
]
