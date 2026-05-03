"""Analyse Z separability for the off-spec Free-Transformer experiment.

Mirrors the persona-vector A11 protocol so the user reads both verdicts in the
same units:

  * Run the trained Free Transformer's encoder on held-out persona-labelled
    sequences, collect the per-token H-dim sigmoid-logits, mean-pool over
    valid (non-pad) tokens to get a per-sequence H-dim feature vector.
  * For each persona vs rest, fit a logistic regression on the H-dim feature
    and report the held-out AUROC.
  * Spurious-correlation control: shuffled-label probe (mean of N=10 shuffles).
  * Random-feature control: 1-D probe on a random scalar feature drawn iid.
  * UMAP figure of the H-dim features coloured by persona.
  * Drift-trajectory probe: encoder logits on `drifting.yaml` turn-by-turn vs
    `in_persona.yaml` turn-by-turn.

Verdict bands (copied from `vectors/probe.py`):
  AUROC ≥ 0.80 → Z separates personas → confirmed.
  0.70 ≤ AUROC < 0.80 → weak.
  AUROC < 0.70 → refuted at this scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from persona_rag.freet.corpus import DRIFT_DIR, CorpusExample
from persona_rag.freet.model import FreeTransformer, FreeTransformerConfig
from persona_rag.freet.tokenizer import TokenizerBundle
from persona_rag.vectors.probe import AUROC_CONFIRMED_FLOOR, AUROC_WEAK_FLOOR

# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


def load_checkpoint(path: Path, device: torch.device) -> tuple[FreeTransformer, dict]:
    raw = torch.load(path, map_location=device)
    cfg = FreeTransformerConfig(**raw["model_cfg"])
    model = FreeTransformer(cfg).to(device)
    state = raw["state_dict"]
    model.load_state_dict(state)
    if device.type == "cuda":
        model = model.to(torch.float16)
    model.eval()
    return model, raw


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class EncodedFeature:
    persona_id: str
    source: str
    feature: np.ndarray  # (H,) mean-pooled sigmoid logits


def _encode_one(
    model: FreeTransformer, ids: torch.Tensor, attn: torch.Tensor, device: torch.device
) -> np.ndarray:
    ids = ids.unsqueeze(0).to(device)
    attn = attn.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model.encode(ids, attn_mask=attn)  # (1, T, H)
    logits = logits.float().squeeze(0)  # (T, H)
    keep = attn.squeeze(0).bool()
    if keep.sum() == 0:
        return logits.mean(dim=0).cpu().numpy()
    return logits[keep].mean(dim=0).cpu().numpy()


def encode_dataset(
    model: FreeTransformer,
    examples: list[CorpusExample],
    tokenizer: TokenizerBundle,
    seq_len: int,
    device: torch.device,
) -> list[EncodedFeature]:
    out: list[EncodedFeature] = []
    for ex in examples:
        ids, attn = tokenizer.encode(ex.text, seq_len)
        feat = _encode_one(model, ids, attn, device)
        out.append(EncodedFeature(ex.persona_id, ex.source, feat))
    return out


# --------------------------------------------------------------------------- #
# Probes                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class PerPersonaResult:
    persona_id: str
    auroc: float
    train_auroc: float
    n_train_pos: int
    n_train_neg: int
    n_test_pos: int
    n_test_neg: int
    shuffled_label_auroc: float
    random_feature_auroc: float
    verdict: str


@dataclass
class SeparationResult:
    per_persona: list[PerPersonaResult]
    macro_auroc: float
    overall_verdict: str
    z_index_entropy_bits: float
    feature_dim: int
    n_train: int
    n_test: int


def _split(
    features: list[EncodedFeature], train_frac: float, seed: int
) -> tuple[list[EncodedFeature], list[EncodedFeature]]:
    """Deterministic prompt-disjoint split via seeded shuffle. The corpus is
    already shuffled at build time but we re-split here so analysis is
    independent of trainer details."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(features))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_frac)
    train_idx = idx[:n_train].tolist()
    test_idx = idx[n_train:].tolist()
    return [features[i] for i in train_idx], [features[i] for i in test_idx]


def _persona_vs_rest_auroc(
    train: list[EncodedFeature], test: list[EncodedFeature], target: str, seed: int
) -> tuple[float, float, int, int, int, int]:
    x_train = np.stack([f.feature for f in train])
    x_test = np.stack([f.feature for f in test])
    y_train = np.array([1 if f.persona_id == target else 0 for f in train])
    y_test = np.array([1 if f.persona_id == target else 0 for f in test])
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        raise ValueError(f"degenerate train labels for persona {target!r}")
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        raise ValueError(f"degenerate test labels for persona {target!r}")
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    train_scores = clf.decision_function(x_train)
    test_scores = clf.decision_function(x_test)
    return (
        roc_auc_score(y_test, test_scores),
        roc_auc_score(y_train, train_scores),
        int((y_train == 1).sum()),
        int((y_train == 0).sum()),
        int((y_test == 1).sum()),
        int((y_test == 0).sum()),
    )


def _shuffled_label_auroc(
    train: list[EncodedFeature], test: list[EncodedFeature], target: str, n_shuffles: int, seed: int
) -> float:
    """Mean test AUROC over `n_shuffles` random label permutations (Dubanowska control)."""
    x_train = np.stack([f.feature for f in train])
    x_test = np.stack([f.feature for f in test])
    y_test = np.array([1 if f.persona_id == target else 0 for f in test])
    rng = np.random.default_rng(seed + hash(target) % (2**32))
    aurocs: list[float] = []
    for _ in range(n_shuffles):
        y_train_shuffled = np.array([1 if f.persona_id == target else 0 for f in train])
        rng.shuffle(y_train_shuffled)
        if y_train_shuffled.sum() in (0, len(y_train_shuffled)):
            continue
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        clf.fit(x_train, y_train_shuffled)
        scores = clf.decision_function(x_test)
        aurocs.append(roc_auc_score(y_test, scores))
    return float(np.mean(aurocs)) if aurocs else 0.5


def _random_feature_auroc(
    train: list[EncodedFeature], test: list[EncodedFeature], target: str, n_repeats: int, seed: int
) -> float:
    """1-D probe on a random scalar feature, mean over `n_repeats`."""
    rng = np.random.default_rng(seed + 1 + hash(target) % (2**32))
    aurocs: list[float] = []
    for _ in range(n_repeats):
        rand_train = rng.standard_normal(len(train)).reshape(-1, 1)
        rand_test = rng.standard_normal(len(test)).reshape(-1, 1)
        y_train = np.array([1 if f.persona_id == target else 0 for f in train])
        y_test = np.array([1 if f.persona_id == target else 0 for f in test])
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        clf.fit(rand_train, y_train)
        scores = clf.decision_function(rand_test)
        aurocs.append(roc_auc_score(y_test, scores))
    return float(np.mean(aurocs))


def _verdict(auroc: float) -> str:
    if auroc >= AUROC_CONFIRMED_FLOOR:
        return "confirmed"
    if auroc >= AUROC_WEAK_FLOOR:
        return "weak"
    return "refuted"


def run_separability(
    features: list[EncodedFeature],
    *,
    train_frac: float = 0.7,
    seed: int = 42,
    n_shuffles: int = 10,
    n_random_repeats: int = 10,
) -> SeparationResult:
    train, test = _split(features, train_frac, seed)
    personas = sorted({f.persona_id for f in features})
    logger.info("personas in feature set: {}", personas)
    per_persona: list[PerPersonaResult] = []
    for pid in personas:
        try:
            auroc, train_auroc, n_tr_pos, n_tr_neg, n_te_pos, n_te_neg = _persona_vs_rest_auroc(
                train, test, pid, seed=seed
            )
        except ValueError as exc:
            logger.warning("skipping persona {!r}: {}", pid, exc)
            continue
        shuf = _shuffled_label_auroc(train, test, pid, n_shuffles, seed)
        rand = _random_feature_auroc(train, test, pid, n_random_repeats, seed)
        per_persona.append(
            PerPersonaResult(
                persona_id=pid,
                auroc=auroc,
                train_auroc=train_auroc,
                n_train_pos=n_tr_pos,
                n_train_neg=n_tr_neg,
                n_test_pos=n_te_pos,
                n_test_neg=n_te_neg,
                shuffled_label_auroc=shuf,
                random_feature_auroc=rand,
                verdict=_verdict(auroc),
            )
        )
    macro = float(np.mean([r.auroc for r in per_persona])) if per_persona else 0.0
    worst = "confirmed"
    for r in per_persona:
        if r.verdict == "refuted":
            worst = "refuted"
        elif r.verdict == "weak" and worst == "confirmed":
            worst = "weak"

    # Z-index entropy: histogram of (most-likely binary code) on the test set.
    # Mirrors the paper's "Z is using only k of 2^H values" diagnostic.
    feats = np.stack([f.feature for f in features])
    bits = (feats > 0).astype(np.int64)
    powers = (1 << np.arange(bits.shape[1])).astype(np.int64)
    indices = (bits * powers).sum(axis=1)
    counts = np.bincount(indices, minlength=int(2 ** bits.shape[1]))
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())

    return SeparationResult(
        per_persona=per_persona,
        macro_auroc=macro,
        overall_verdict=worst,
        z_index_entropy_bits=entropy_bits,
        feature_dim=feats.shape[1],
        n_train=len(train),
        n_test=len(test),
    )


# --------------------------------------------------------------------------- #
# Drift-trajectory probe                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class DriftProbeRow:
    persona_id: str
    condition: str  # "in_persona" | "drifting"
    turn_idx: int
    drift_level: str | None
    feature: list[float]


def run_drift_trajectory(
    model: FreeTransformer,
    tokenizer: TokenizerBundle,
    seq_len: int,
    device: torch.device,
) -> list[DriftProbeRow]:
    """Encode each assistant turn of the drift YAMLs separately and return per-turn features."""
    out: list[DriftProbeRow] = []
    for persona_dir in sorted(DRIFT_DIR.iterdir()):
        if not persona_dir.is_dir():
            continue
        pid = persona_dir.name
        for cond in ("in_persona", "drifting"):
            ypath = persona_dir / f"{cond}.yaml"
            if not ypath.exists():
                continue
            raw = yaml.safe_load(ypath.read_text(encoding="utf-8"))
            for i, turn in enumerate(raw.get("turns", [])):
                if turn.get("role") != "assistant":
                    continue
                ids, attn = tokenizer.encode(turn.get("text", ""), seq_len)
                feat = _encode_one(model, ids, attn, device)
                out.append(
                    DriftProbeRow(
                        persona_id=pid,
                        condition=cond,
                        turn_idx=i,
                        drift_level=turn.get("drift_level"),
                        feature=feat.tolist(),
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# UMAP figure (optional — same convention as validate_persona_vectors)        #
# --------------------------------------------------------------------------- #


def umap_figure(features: list[EncodedFeature], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    feats = np.stack([f.feature for f in features])
    pids = [f.persona_id for f in features]
    unique = sorted(set(pids))
    n_neighbors = min(15, max(2, feats.shape[0] - 1))
    proj = UMAP(random_state=42, n_neighbors=n_neighbors, metric="cosine").fit_transform(feats)
    fig, ax = plt.subplots(figsize=(6, 5))
    for pid in unique:
        mask = np.array([p == pid for p in pids])
        ax.scatter(proj[mask, 0], proj[mask, 1], label=pid, alpha=0.7, s=14)
    ax.set_title("UMAP of encoder features (Q(Z|S) sigmoid logits, mean over T)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Top-level entry                                                             #
# --------------------------------------------------------------------------- #


def write_verdict(report_dir: Path, sep: SeparationResult, ckpt_meta: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": "scripts/validate_freet_z_separation.py",
        "checkpoint_step": ckpt_meta.get("step"),
        "model_cfg": ckpt_meta.get("model_cfg"),
        "train_cfg": ckpt_meta.get("train_cfg"),
        "thresholds": {
            "confirmed_floor": AUROC_CONFIRMED_FLOOR,
            "weak_floor": AUROC_WEAK_FLOOR,
        },
        "separation": asdict(sep),
    }
    (report_dir / "z_separability.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Free-Transformer Z-Separability — {sep.overall_verdict.upper()}",
        "",
        f"- Macro AUROC across personas: **{sep.macro_auroc:.3f}**",
        f"- Z-index entropy on held-out features: {sep.z_index_entropy_bits:.2f} bits "
        f"(max possible: {sep.feature_dim} bits = {2**sep.feature_dim} indices)",
        f"- Train / test sizes: {sep.n_train} / {sep.n_test}",
        "",
        "## Per-persona (one-vs-rest)",
        "",
        "| Persona | Test AUROC | Train AUROC | Shuffled-label | Random-feature | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in sep.per_persona:
        lines.append(
            f"| `{r.persona_id}` | {r.auroc:.3f} | {r.train_auroc:.3f} | "
            f"{r.shuffled_label_auroc:.3f} | {r.random_feature_auroc:.3f} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## Verdict rule")
    lines.append("")
    lines.append(f"- AUROC ≥ {AUROC_CONFIRMED_FLOOR:.2f} → Z separates personas (confirmed)")
    lines.append(f"- {AUROC_WEAK_FLOOR:.2f} ≤ AUROC < {AUROC_CONFIRMED_FLOOR:.2f} → weak")
    lines.append(f"- AUROC < {AUROC_WEAK_FLOOR:.2f} → refuted at this scale")
    (report_dir / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
