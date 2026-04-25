"""Drift-trajectory sanity experiment.

For every persona under ``conversations_dir`` and each of two conditions
(``in_persona``, ``drifting``), runs one forward pass per turn — projecting
the prompt-time hidden state at the configured layer onto the persona's
cached persona vector — and records per-turn drift trajectories. Produces:

    results/drift_trajectory/<run_id>/
        report.md           — verdict, per-persona deltas
        raw.json            — per-turn drift values + metadata
        trajectories.png    — drift x turn, both conditions, per persona
        verify.log          — full debug log

Uses the cached persona vectors written by the validation script
(``cache_dir`` config) — those files are bit-exact via safetensors.

Usage:

    uv run python scripts/drift_trajectory_sanity.py model=gemma
    uv run python scripts/drift_trajectory_sanity.py model=gemma layer=8 \\
        cache_dir=results/a11_validation/<run_id>/.chroma/persona_vectors

Notes:

- The script imports from ``persona_rag.models``; on macOS this fails because
  ``bitsandbytes`` is unavailable. Run on a CUDA host.
- The persona-vector cache must already exist (run the validation script
  first).
- Each persona is rendered into the system prompt as identity + constraints
  + self_facts + worldview, assembled into a single block.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from persona_rag.schema.conversation import (
    DriftTrajectoryConversation,
    assert_user_turns_match,
)
from persona_rag.schema.persona import Persona
from persona_rag.vectors.cache import load_persona_vectors
from persona_rag.vectors.drift import DriftSignal

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"


# --------------------------------------------------------------------- helpers


def _load_backend(cfg: DictConfig) -> Any:
    """Load the LLM backend. Mirrors scripts/validate_persona_vectors.py."""
    from persona_rag.models import HFBackendConfig
    from persona_rag.models.gemma import GemmaBackend
    from persona_rag.models.llama import LlamaBackend

    model_cfg = cfg.model
    backend_cfg = HFBackendConfig(
        model_id=model_cfg.model_id,
        name=model_cfg.name,
        revision=model_cfg.get("revision"),
        compute_dtype=model_cfg.compute_dtype,
        attn_implementation=model_cfg.attn_implementation,
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        max_input_tokens=model_cfg.max_input_tokens,
        trust_remote_code=model_cfg.trust_remote_code,
        warmup_nan_guard=bool(model_cfg.get("warmup_nan_guard", True)),
    )
    name = str(model_cfg.name)
    if name.startswith("gemma"):
        return GemmaBackend(backend_cfg)
    if name.startswith("llama"):
        return LlamaBackend(backend_cfg)
    raise ValueError(f"unsupported model name: {name!r}")


def _persona_system_text(persona: Persona) -> str:
    """Render the persona as a system-style block.

    Composition: identity + constraints + self_facts + worldview, assembled
    into a single string. This is the canonical assembly used by both this
    experiment and any downstream retrieval mechanism that needs a stable
    persona prompt prefix.
    """
    parts: list[str] = []
    parts.append(f"You are {persona.identity.name}, {persona.identity.role}.")
    parts.append(f"Background: {persona.identity.background}".strip())
    if persona.identity.constraints:
        parts.append("Constraints:")
        parts.extend(f"- {c}" for c in persona.identity.constraints)
    if persona.self_facts:
        parts.append("Self-facts:")
        parts.extend(f"- {f.fact}" for f in persona.self_facts)
    if persona.worldview:
        parts.append("Worldview claims (with epistemic tags):")
        parts.extend(f"- ({w.epistemic}) {w.claim}" for w in persona.worldview)
    parts.append(
        "Respond as this persona would. Use the self-facts and worldview as your "
        "own. Respect every constraint."
    )
    return "\n".join(parts)


@dataclass
class TurnReading:
    persona_id: str
    condition: str
    turn_idx: int  # zero-based pair index (0..n_pairs-1)
    user_text: str
    assistant_text: str
    drift_level: str | None  # only on drifting condition
    drift: float


def _measure_turn_drift(
    *,
    backend: Any,
    persona_system: str,
    conv: DriftTrajectoryConversation,
    drift_signal: DriftSignal,
    layer: int,
) -> list[TurnReading]:
    """Per-turn pre-generation drift measurement.

    DESIGN CHOICE — history threading:
        At turn t, the prompt's history contains turns 0..t-1 of THIS
        conversation's assistant content. So the drifting condition's
        prompt at turn 5 contains the drifting assistant turns 1-4 in
        history; the in-persona condition's prompt at turn 5 contains
        the in-persona assistant turns 1-4.

        This matches the runtime semantics of a drift-gated retrieval
        mechanism: the drift detector at turn t sees the conversation as
        actually generated, not as it would have been in an in-persona run.
        An alternative design (always thread in-persona history regardless
        of condition) would isolate single-turn drift detection — but that's
        not what runtime gating does, so it's not what we test here.

        Consequence: drift measurement at turn t consumes assistant turns
        0..t-1 as history. With the shipped YAMLs, drifting assistant
        content first diverges at turn 2 (the first ``subtle`` annotation),
        so drift *measurements* first diverge at turn 3 — i.e. measurement
        at turns 0, 1, 2 is guaranteed identical between conditions; the
        drift-turns delta averages turns 3..n_pairs-1.

    For each pair index t (0..n_pairs-1):
      - Build the prompt: system + persona block + history through turn t-1
        + the user message for turn t. (We measure BEFORE the assistant turn
        t, so the hidden state captures what the model believes the next
        response should look like, not the response itself.)
      - Run forward through the model; capture the layer-l last-token hidden
        state of the prompt; project onto the persona vector → drift.
    """
    from persona_rag.models.base import ChatMessage

    user_texts = conv.user_turn_texts()
    assistant_texts = conv.assistant_turn_texts()

    readings: list[TurnReading] = []
    for t in range(conv.n_pairs):
        history: list[ChatMessage] = []
        for prior in range(t):
            history.append(ChatMessage(role="user", content=user_texts[prior]))
            history.append(ChatMessage(role="assistant", content=assistant_texts[prior]))

        prompt = backend.format_persona_prompt(
            system_text=persona_system,
            user_text=user_texts[t],
            history=history if history else None,
        )

        captured = backend.get_hidden_states(
            prompt,
            layers=[layer],
            pool="last",
            over="prompt",
        )
        # captured is dict[layer_idx, Tensor(1, hidden_dim)] or similar — the
        # extractor's get_hidden_states returns CPU float32 by spec; we squeeze
        # any singleton dims to land on a 1-D tensor for cosine math.
        h = captured[layer].detach().to("cpu").float().squeeze()
        if h.dim() != 1:
            raise RuntimeError(
                f"unexpected hidden-state shape {tuple(h.shape)} at layer {layer}; "
                "expected 1-D after pool='last' over='prompt'"
            )

        d = drift_signal.compute(h)
        drift_level = None
        if conv.condition == "drifting":
            # The conversation schema has drift_level on assistant turns only;
            # the i-th assistant turn corresponds to the i-th pair index.
            asst_turns = [tt for tt in conv.turns if tt.role == "assistant"]
            drift_level = asst_turns[t].drift_level

        readings.append(
            TurnReading(
                persona_id=conv.persona_id,
                condition=conv.condition,
                turn_idx=t,
                user_text=user_texts[t][:120],
                assistant_text=assistant_texts[t][:120],
                drift_level=drift_level,
                drift=float(d),
            )
        )
        logger.info(
            "{}/{} turn={} drift={:+.3f} authored={}",
            conv.persona_id,
            conv.condition,
            t,
            d,
            drift_level or "-",
        )
    return readings


# --------------------------------------------------------------------- analysis


def _decide_verdict(
    per_persona_delta: dict[str, float],
    *,
    delta_h1: float,
    delta_h2: float,
) -> tuple[str, str]:
    """Return (verdict, rationale) for one of three outcomes.

    - ``proceed``: delta >= delta_h1 on >= 2 of 3 personas (signal is real).
    - ``refuted``: delta < delta_h2 on ALL personas (signal is approximately
      constant across conditions).
    - ``inconclusive``: anything in between — needs human review.
    """
    n = len(per_persona_delta)
    n_strong = sum(1 for d in per_persona_delta.values() if d >= delta_h1)
    n_weak = sum(1 for d in per_persona_delta.values() if d < delta_h2)
    if n_strong >= max(2, n - 1):
        return "proceed", (
            f"{n_strong}/{n} personas show delta >= {delta_h1:.2f} (in_persona - drifting)."
        )
    if n_weak == n:
        return "refuted", (
            f"all {n} personas show delta < {delta_h2:.2f} — drift signal is approximately "
            "constant across in_persona vs drifting conditions."
        )
    return "inconclusive", (
        f"intermediate: {n_strong}/{n} personas above the proceed threshold; "
        f"{n - n_weak}/{n} above the refuted floor — needs review."
    )


def _trajectory_figure(
    readings: list[TurnReading],
    out_path: Path,
) -> None:
    """One subplot per persona: drift x turn for both conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_persona: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for r in readings:
        by_persona.setdefault(r.persona_id, {}).setdefault(r.condition, []).append(
            (r.turn_idx, r.drift)
        )

    persona_ids = sorted(by_persona.keys())
    fig, axes = plt.subplots(
        1, len(persona_ids), figsize=(5 * len(persona_ids), 4), sharey=True, squeeze=False
    )
    for ax, pid in zip(axes[0], persona_ids, strict=True):
        for cond, marker, color in (
            ("in_persona", "o", "C0"),
            ("drifting", "x", "C3"),
        ):
            pts = sorted(by_persona[pid].get(cond, []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=marker, color=color, label=cond, linewidth=1.5)
        ax.axhline(0.0, linestyle=":", color="grey", linewidth=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("turn index")
        ax.set_title(pid)
        ax.legend(loc="lower left", fontsize=8)
    axes[0][0].set_ylabel("drift signal (cosine, [-1, +1])")
    fig.suptitle("Drift trajectory — per turn, by persona and condition")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------- main


@hydra.main(
    config_path="../src/persona_rag/config", config_name="drift_trajectory", version_base=None
)
def main(cfg: DictConfig) -> int:
    """Run the drift-trajectory experiment end-to-end."""
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.add(report_dir / "verify.log", level="DEBUG")
    logger.info("drift-trajectory: config\n{}", OmegaConf.to_yaml(cfg))
    (report_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    layer = int(cfg.layer)
    cache_dir = Path(cfg.cache_dir)
    convs_dir = Path(cfg.conversations_dir)
    if not cache_dir.exists():
        logger.error(
            "persona-vector cache_dir does not exist: {} — run scripts/validate_persona_vectors.py first",
            cache_dir,
        )
        return 2

    # Discover personas via the conversation directory layout.
    persona_dirs = sorted(p for p in convs_dir.iterdir() if p.is_dir())
    if not persona_dirs:
        logger.error("no persona conversations under {}", convs_dir)
        return 2
    logger.info("personas to evaluate: {}", [p.name for p in persona_dirs])

    backend = _load_backend(cfg)

    all_readings: list[TurnReading] = []
    per_persona_delta: dict[str, float] = {}
    per_persona_summary: dict[str, dict[str, Any]] = {}

    for persona_dir in persona_dirs:
        pid = persona_dir.name

        # Persona YAML for the system prompt.
        persona_yaml = PERSONAS_DIR / f"{pid}.yaml"
        if not persona_yaml.exists():
            logger.error("persona {!r}: yaml not found at {}", pid, persona_yaml)
            return 2
        persona = Persona.from_yaml(persona_yaml)

        # Cached persona vectors → drift signal at the requested layer.
        try:
            pv = load_persona_vectors(cache_dir, pid)
        except FileNotFoundError as err:
            logger.error("persona {!r}: cache miss — {}", pid, err)
            return 2
        if layer not in pv.in_persona_centroid:
            logger.error(
                "persona {!r}: layer {} not in cached vectors (have {})",
                pid,
                layer,
                sorted(pv.in_persona_centroid),
            )
            return 2
        drift_signal = DriftSignal.from_persona_vectors(pv, layer)

        # Conversations.
        in_conv = DriftTrajectoryConversation.from_yaml(persona_dir / "in_persona.yaml")
        drift_conv = DriftTrajectoryConversation.from_yaml(persona_dir / "drifting.yaml")
        assert_user_turns_match([in_conv, drift_conv])

        persona_system = _persona_system_text(persona)

        in_readings = _measure_turn_drift(
            backend=backend,
            persona_system=persona_system,
            conv=in_conv,
            drift_signal=drift_signal,
            layer=layer,
        )
        drift_readings = _measure_turn_drift(
            backend=backend,
            persona_system=persona_system,
            conv=drift_conv,
            drift_signal=drift_signal,
            layer=layer,
        )
        all_readings.extend(in_readings)
        all_readings.extend(drift_readings)

        # Measurement at turn t consumes assistant history 0..t-1. With the
        # shipped YAMLs, drifting assistant content first diverges at turn 2
        # (`subtle`), so drift *measurements* first differ at turn 3.
        # Averaging across all six turns dilutes the signal we're after; the
        # authoritative delta averages turns 3..n_pairs-1, where measurements
        # actually diverge between conditions.
        DRIFT_TURNS = tuple(range(3, in_conv.n_pairs))  # (3, 4, 5) for n_pairs=6

        in_drifts_all = np.array([r.drift for r in in_readings])
        drift_drifts_all = np.array([r.drift for r in drift_readings])
        mu_in_all = float(in_drifts_all.mean())
        mu_drift_all = float(drift_drifts_all.mean())
        delta_all = mu_in_all - mu_drift_all

        in_drifts_dt = np.array(
            [r.drift for r in in_readings if r.turn_idx in DRIFT_TURNS]
        )
        drift_drifts_dt = np.array(
            [r.drift for r in drift_readings if r.turn_idx in DRIFT_TURNS]
        )
        mu_in_dt = float(in_drifts_dt.mean())
        mu_drift_dt = float(drift_drifts_dt.mean())
        delta_dt = mu_in_dt - mu_drift_dt

        # Verdict uses the drift-turns-only delta — that's where the signal is
        # expected; turns 0-1 are control, turn 2 is in-persona-by-design.
        per_persona_delta[pid] = delta_dt

        per_persona_summary[pid] = {
            # All-turns (legacy / disclosure):
            "mu_in_all_turns": mu_in_all,
            "mu_drift_all_turns": mu_drift_all,
            "delta_all_turns": delta_all,
            # Drift-turns-only (authoritative for verdict):
            "mu_in_drift_turns": mu_in_dt,
            "mu_drift_drift_turns": mu_drift_dt,
            "delta_drift_turns": delta_dt,
            "drift_turns": list(DRIFT_TURNS),
            # Per-turn (full disclosure):
            "in_persona_per_turn": [r.drift for r in in_readings],
            "drifting_per_turn": [r.drift for r in drift_readings],
            "drift_levels_drifting": [r.drift_level for r in drift_readings],
        }
        logger.info(
            "persona {!r}: delta_all_turns={:+.3f} | delta_drift_turns={:+.3f} (turns {})",
            pid,
            delta_all,
            delta_dt,
            list(DRIFT_TURNS),
        )

    # Trajectory figure.
    _trajectory_figure(all_readings, report_dir / "trajectories.png")

    verdict, rationale = _decide_verdict(
        per_persona_delta,
        delta_h1=float(cfg.delta_h1),
        delta_h2=float(cfg.delta_h2),
    )

    raw = {
        "script": "scripts/drift_trajectory_sanity.py",
        "backend": {
            "name": getattr(backend, "name", "?"),
            "model_id": getattr(backend, "model_id", "?"),
            "num_layers": getattr(backend, "num_layers", None),
            "hidden_dim": getattr(backend, "hidden_dim", None),
        },
        "config": {
            "layer": layer,
            "cache_dir": str(cache_dir),
            "delta_h1": float(cfg.delta_h1),
            "delta_h2": float(cfg.delta_h2),
        },
        "per_persona": per_persona_summary,
        # per_persona_delta carries the drift-turns-only delta (turns 3..n-1),
        # which is what the verdict is decided from. The all-turns delta is
        # in per_persona[<pid>].delta_all_turns for disclosure.
        "per_persona_delta": per_persona_delta,
        "per_persona_delta_kind": "delta_drift_turns",
        "verdict": verdict,
        "rationale": rationale,
        "readings": [
            {
                "persona_id": r.persona_id,
                "condition": r.condition,
                "turn_idx": r.turn_idx,
                "user_text": r.user_text,
                "assistant_text": r.assistant_text,
                "drift_level": r.drift_level,
                "drift": r.drift,
            }
            for r in all_readings
        ],
    }
    (report_dir / "raw.json").write_text(
        json.dumps(raw, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    # report.md
    lines: list[str] = []
    lines.append(f"# Drift Trajectory — Verdict: **{verdict}**")
    lines.append("")
    lines.append(f"- Backend: `{raw['backend']['name']}` ({raw['backend']['model_id']})")
    lines.append(f"- Layer: {layer}")
    lines.append(f"- Cache dir: `{cache_dir}`")
    lines.append(f"- Thresholds: proceed >= {cfg.delta_h1}; refuted < {cfg.delta_h2}")
    lines.append("")
    lines.append(f"**Rationale:** {rationale}")
    lines.append("")
    lines.append(
        "**Design note:** Per-turn measurements thread each condition's own "
        "assistant content into history. The drifting condition's prompt at "
        "turn t contains the drifting assistant turns 0..t-1; the in-persona "
        "condition's prompt at turn t contains the in-persona assistant turns "
        "0..t-1. This matches the runtime semantics of a drift-gated retrieval "
        "mechanism — the drift detector sees the conversation as actually "
        "generated. Consequence: drift measurement at turn t depends on "
        "history 0..t-1; with the shipped YAMLs the first authored drift sits "
        "in assistant turn 2 (`subtle`), so drift *measurements* first diverge "
        "at turn 3. Turns 0-2 produce identical readings between conditions "
        "by design."
    )
    lines.append("")
    lines.append("## Per-persona deltas")
    lines.append("")
    lines.append(
        "| Persona | mu_in (all) | mu_drift (all) | delta (all) | "
        "delta (drift turns) | per-persona verdict |"
    )
    lines.append("|---|---|---|---|---|---|")
    for pid, summary in per_persona_summary.items():
        d_dt = summary["delta_drift_turns"]
        if d_dt >= float(cfg.delta_h1):
            pp_verdict = "proceed"
        elif d_dt < float(cfg.delta_h2):
            pp_verdict = "refuted"
        else:
            pp_verdict = "inconclusive"
        lines.append(
            "| `{pid}` | {mi:+.3f} | {md:+.3f} | {dla:+.3f} | **{dldt:+.3f}** | {v} |".format(
                pid=pid,
                mi=summary["mu_in_all_turns"],
                md=summary["mu_drift_all_turns"],
                dla=summary["delta_all_turns"],
                dldt=d_dt,
                v=pp_verdict,
            )
        )
    lines.append("")
    lines.append(
        "**Note:** the verdict uses `delta (drift turns)` — averaged over "
        "turns 3..n_pairs-1, where drift measurements actually diverge "
        "between conditions. The `delta (all)` column is shown for "
        "disclosure; it dilutes the signal because turns 0-2 are guaranteed "
        "identical across conditions by experimental design."
    )
    lines.append("")
    lines.append("## Per-turn drift trajectories")
    lines.append("")
    for pid, summary in per_persona_summary.items():
        lines.append(f"### {pid}")
        lines.append("")
        lines.append("| turn | in_persona drift | drifting drift | authored drift_level |")
        lines.append("|---|---|---|---|")
        for t, (di, dd) in enumerate(
            zip(summary["in_persona_per_turn"], summary["drifting_per_turn"], strict=True)
        ):
            lvl = summary["drift_levels_drifting"][t] or "-"
            lines.append(f"| {t} | {di:+.3f} | {dd:+.3f} | {lvl} |")
        lines.append("")
    lines.append("![drift trajectories](trajectories.png)")
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append(f"- **proceed**: delta >= {cfg.delta_h1} on >= 2 of {{n}} personas.")
    lines.append(f"- **refuted**: delta < {cfg.delta_h2} on ALL personas.")
    lines.append("- **inconclusive**: anything in between — needs human review.")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("drift-trajectory complete: verdict={}", verdict)
    # Exit codes: 0 on proceed, 1 on inconclusive, 2 on refuted.
    if verdict == "proceed":
        return 0
    if verdict == "inconclusive":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
