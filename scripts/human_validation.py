"""Human-validation pipeline CLI: export -> annotate -> score.

Two subcommands:

- ``export`` -- sample N conversations stratified across mechanisms,
  write a blinded CSV the human fills in.
- ``score`` -- read the human-filled CSV, join with PoLL panel
  checkpoints, compute Krippendorff's alpha (per-judge + panel).

Defaults targeted at the 20-item pilot. Pass ``--per-mechanism 38``
for the full 150-item sweep.

Usage:

    # Export 20-item pilot from existing baseline runs.
    uv run python scripts/human_validation.py export \\
        --b1-dir results/run_b1/  --b2-dir results/run_b2/ \\
        --m1-dir results/run_m1/  --m3-dir results/run_m3/ \\
        --persona cs_tutor --per-mechanism 5 \\
        --out-dir results/human_validation/pilot

    # After the human fills the CSV:
    uv run python scripts/human_validation.py score \\
        --human-csv results/human_validation/pilot/export.csv \\
        --mapping   results/human_validation/pilot/mapping.json \\
        --panel-dir results/poll_pilot/<timestamp> \\
        --out-dir   results/human_validation/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from persona_rag.evaluation.human_validation import (
    alpha_against_panel,
    export_csv,
    load_human_csv,
    load_panel,
    stratified_sample,
    write_alpha_report,
)
from persona_rag.evaluation.transcripts import load_baseline_response_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cmd_export(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_mech: dict[str, list] = {}
    for mech, src in [
        ("b1", args.b1_dir),
        ("b2", args.b2_dir),
        ("m1", args.m1_dir),
        ("m3", args.m3_dir),
    ]:
        if not src:
            continue
        convs = load_baseline_response_dir(
            Path(src).resolve(),
            mechanism=mech,
            persona_id=args.persona,
        )
        by_mech[mech] = convs
        logger.info("loaded {} conversations for mechanism {}", len(convs), mech)

    if not by_mech:
        raise SystemExit("Provide at least one of --b1-dir / --b2-dir / --m1-dir / --m3-dir")

    items = stratified_sample(
        by_mech,
        per_mechanism=args.per_mechanism,
        seed=args.seed,
    )
    csv_path = out_dir / "export.csv"
    mapping_path = out_dir / "mapping.json"
    export_csv(items, csv_path, mapping_path)
    logger.info(
        "wrote {} blinded items to {} (mapping at {})",
        len(items),
        csv_path,
        mapping_path,
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    human_rows = load_human_csv(Path(args.human_csv), Path(args.mapping))
    if not human_rows:
        raise SystemExit("No human-scored rows found -- did the annotator fill in the CSV?")
    logger.info("loaded {} human-scored rows", len(human_rows))

    panel = load_panel(Path(args.panel_dir))
    if not panel:
        raise SystemExit(f"No panel checkpoints found under {args.panel_dir}")
    logger.info("loaded {} panel judges: {}", len(panel), sorted(panel.keys()))

    alphas = alpha_against_panel(human_rows, panel)
    (out_dir / "alpha.json").write_text(json.dumps(alphas, indent=2) + "\n", encoding="utf-8")
    write_alpha_report(
        out_dir / "alpha_report.md",
        alphas,
        n_human_items=len(human_rows),
        panel_dir=Path(args.panel_dir),
    )
    logger.info("alphas: {}", alphas)
    return 0


def main() -> int:
    """CLI entry. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="export blinded CSV for human rating")
    p_export.add_argument("--b1-dir")
    p_export.add_argument("--b2-dir")
    p_export.add_argument("--m1-dir")
    p_export.add_argument("--m3-dir")
    p_export.add_argument("--persona", required=True)
    p_export.add_argument("--per-mechanism", type=int, default=5)
    p_export.add_argument("--seed", type=int, default=42)
    p_export.add_argument("--out-dir", required=True)
    p_export.set_defaults(func=_cmd_export)

    p_score = sub.add_parser("score", help="join human CSV with panel; compute alpha")
    p_score.add_argument("--human-csv", required=True)
    p_score.add_argument("--mapping", required=True)
    p_score.add_argument("--panel-dir", required=True)
    p_score.add_argument("--out-dir", required=True)
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
