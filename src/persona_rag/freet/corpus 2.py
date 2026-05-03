"""Build a persona-labeled training corpus for the off-spec Free-Transformer experiment.

Sources we draw from (all already in the repo):

  * ``personas/<pid>.yaml``                 — identity, self_facts, worldview, episodic.
  * ``benchmarks_data/drift_trajectory/<pid>/{in_persona,drifting}.yaml``
                                            — multi-turn conversations, in-persona side
                                              labelled with the persona, drifting side
                                              labelled with the persona too (it is *that*
                                              persona drifting, not a different one).
  * ``benchmarks_data/counterfactual_probes/<pid>/probe_*.yaml`` and
    ``benchmarks_data/counterfactual_probes/chunks/<pid>/*.md``
                                            — probe inputs and grounding chunks.

The 6 trajectory YAMLs alone are too small to train a Transformer from scratch,
so we generate ~10 MB of templated paraphrases of the self-facts, worldview
claims, episodic snippets, identity, and constraints. Each rendered example
gets a ``persona_id`` label that is **not** shown to the model — we only use
it during analysis to ask whether Q(Z|S) recovers it.

The corpus is **deterministic** (`seed` controls every random choice), so
re-running this module produces identical training data — required for the
A11-style verdict to be re-runnable.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from persona_rag.schema.persona import Persona

# ---------- paths ----------

REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONAS_DIR = REPO_ROOT / "personas"
DRIFT_DIR = REPO_ROOT / "benchmarks_data" / "drift_trajectory"
PROBES_DIR = REPO_ROOT / "benchmarks_data" / "counterfactual_probes"
PROBE_CHUNKS_DIR = PROBES_DIR / "chunks"

# ---------- schema ----------


@dataclass(frozen=True)
class CorpusExample:
    """A single labelled training/eval example.

    The label is **not** prepended to the text. It is carried alongside so the
    analysis script knows which sequence belongs to which persona.
    """

    persona_id: str
    source: str  # e.g. "self_facts" | "worldview" | "drift_in" | "drift_drift" | "probe" | "synth"
    text: str

    def sha(self) -> str:
        return hashlib.sha256(f"{self.persona_id}|{self.source}|{self.text}".encode()).hexdigest()[:16]


# ---------- templated paraphrases ----------


# First-person rephrasing templates over self_facts. The intent is to give the
# model hundreds of varied first-person utterances that share underlying facts,
# so Q(Z|S) has something to compress.
SELF_FACT_TEMPLATES: tuple[str, ...] = (
    "{fact}",
    "Quick note about me: {fact_lower}",
    "If it helps to know: {fact_lower}",
    "One thing I should mention — {fact_lower}",
    "For context, {fact_lower}",
    "I'll add a small bio note: {fact_lower}",
    "Background fact: {fact_lower}",
    "Let me put a stake in the ground: {fact_lower}",
    "It's worth saying up front that {fact_lower}",
    "I think it's relevant that {fact_lower}",
)

WORLDVIEW_TEMPLATES: tuple[str, ...] = (
    "I think {claim_lower}",
    "My view is that {claim_lower}",
    "I'd argue {claim_lower}",
    "It's been my experience that {claim_lower}",
    "The position I hold is that {claim_lower}",
    "I tend to believe {claim_lower}",
    "Speaking from my own work, {claim_lower}",
    "If pressed, I'd say {claim_lower}",
)

CONSTRAINT_TEMPLATES: tuple[str, ...] = (
    "I should note: {constraint_lower}",
    "A small caveat — {constraint_lower}",
    "I keep a rule for myself that I {constraint_lower}",
    "It's worth flagging that I {constraint_lower}",
)

EPISODIC_TEMPLATES: tuple[str, ...] = (
    "I remember {gist_lower}",
    "There was a time when {gist_lower}",
    "From a session a while back: {gist_lower}",
    "One conversation that stays with me: {gist_lower}",
    "I keep going back to this exchange: {gist_lower}",
)

QUESTION_PROMPTS: tuple[str, ...] = (
    "Could you say a bit about who you are and how you work?",
    "What is your background, and how does it shape your advice?",
    "What are some things you find yourself saying often?",
    "Tell me how you'd introduce yourself to someone who's new to your field.",
    "What are the constraints you keep on yourself?",
    "What's a position you hold strongly?",
    "Describe a recent conversation you've had with someone you've helped.",
    "Walk me through how you'd start with a new person.",
)


# ---------- helpers ----------


def _lower(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def _render_self_facts(persona: Persona, rng: random.Random, repeats: int) -> Iterable[str]:
    for fact_obj in persona.self_facts:
        for _ in range(repeats):
            tmpl = rng.choice(SELF_FACT_TEMPLATES)
            yield tmpl.format(fact=fact_obj.fact, fact_lower=_lower(fact_obj.fact))


def _render_worldview(persona: Persona, rng: random.Random, repeats: int) -> Iterable[str]:
    for w in persona.worldview:
        for _ in range(repeats):
            tmpl = rng.choice(WORLDVIEW_TEMPLATES)
            yield tmpl.format(claim=w.claim, claim_lower=_lower(w.claim))


def _render_constraints(persona: Persona, rng: random.Random, repeats: int) -> Iterable[str]:
    for c in persona.identity.constraints:
        for _ in range(repeats):
            tmpl = rng.choice(CONSTRAINT_TEMPLATES)
            yield tmpl.format(constraint=c, constraint_lower=_lower(c))


def _render_episodic(persona: Persona, rng: random.Random, repeats: int) -> Iterable[str]:
    for e in persona.episodic:
        gist = getattr(e, "gist", None) or getattr(e, "summary", None) or ""
        if not gist:
            continue
        for _ in range(repeats):
            tmpl = rng.choice(EPISODIC_TEMPLATES)
            yield tmpl.format(gist=gist, gist_lower=_lower(gist))


def _render_introductions(persona: Persona, rng: random.Random, repeats: int) -> Iterable[str]:
    """Templated first-person introductions, joined from the persona's structured fields.

    The pattern: question + answer with random subset of the persona's content.
    Generates Q&A-shaped sequences so the encoder sees varied conversational
    framing on top of the bare facts.
    """
    for _ in range(repeats):
        question = rng.choice(QUESTION_PROMPTS)
        chunks: list[str] = [f"I'm {persona.identity.name}, {persona.identity.role}."]
        if persona.self_facts:
            chunks.append(rng.choice(persona.self_facts).fact)
        if persona.worldview:
            chunks.append("I think " + _lower(rng.choice(persona.worldview).claim))
        if persona.identity.constraints:
            chunks.append("I'm careful to " + _lower(rng.choice(persona.identity.constraints)))
        answer = " ".join(chunks)
        yield f"User: {question}\nAssistant: {answer}"


# ---------- multi-turn YAMLs (drift trajectories) ----------


def _load_multiturn_yaml(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(raw.get("turns", []))


def _render_multiturn(turns: list[dict]) -> str:
    """Render a list of role-tagged turns into a flat User:/Assistant: string."""
    lines: list[str] = []
    for turn in turns:
        role = turn.get("role", "")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


# ---------- counterfactual probes ----------


def _load_probe_yaml(path: Path) -> list[str]:
    """Pull human-readable text fields from a probe YAML.

    The probe schema across types: the canonical fields are ``user_turns`` (a
    list of strings) and ``notes`` (free text). Older / draft probes also use
    ``prompt`` / ``assistant`` / ``expected`` strings or a ``turns`` list of
    role-tagged dicts. We tolerate all shapes.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for key in ("prompt", "user", "assistant", "expected", "context", "notes"):
        if key in raw and isinstance(raw[key], str):
            out.append(raw[key])
    if "user_turns" in raw and isinstance(raw["user_turns"], list):
        for t in raw["user_turns"]:
            if isinstance(t, str):
                out.append(t)
    if "turns" in raw and isinstance(raw["turns"], list):
        for turn in raw["turns"]:
            if isinstance(turn, dict) and isinstance(turn.get("text"), str):
                out.append(turn["text"])
    return out


def _load_probe_chunk_text(persona_id: str) -> Iterable[str]:
    chunks_dir = PROBE_CHUNKS_DIR / persona_id
    if not chunks_dir.exists():
        return []
    paragraphs: list[str] = []
    for path in sorted(chunks_dir.glob("*.md")):
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph and not paragraph.startswith("#"):
                paragraphs.append(paragraph)
    return paragraphs


# ---------- top-level builder ----------


@dataclass
class CorpusBuildConfig:
    seed: int = 42
    facts_repeats: int = 80
    worldview_repeats: int = 60
    constraints_repeats: int = 30
    episodic_repeats: int = 40
    intro_repeats: int = 200
    drift_repeats: int = 60
    probe_repeats: int = 6
    chunk_repeats: int = 4
    train_fraction: float = 0.9


def build_corpus(
    out_dir: Path,
    cfg: CorpusBuildConfig | None = None,
    persona_dir: Path = PERSONAS_DIR,
) -> dict[str, int]:
    """Build the persona-labeled corpus, write JSONL train/test, return stats."""
    cfg = cfg or CorpusBuildConfig()
    rng = random.Random(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    persona_paths = sorted(persona_dir.glob("*.yaml"))
    if not persona_paths:
        raise RuntimeError(f"no personas at {persona_dir}")

    examples: list[CorpusExample] = []
    for path in persona_paths:
        persona = Persona.from_yaml(path)
        pid = persona.persona_id or path.stem

        for text in _render_self_facts(persona, rng, cfg.facts_repeats):
            examples.append(CorpusExample(pid, "self_facts", text))
        for text in _render_worldview(persona, rng, cfg.worldview_repeats):
            examples.append(CorpusExample(pid, "worldview", text))
        for text in _render_constraints(persona, rng, cfg.constraints_repeats):
            examples.append(CorpusExample(pid, "constraints", text))
        for text in _render_episodic(persona, rng, cfg.episodic_repeats):
            examples.append(CorpusExample(pid, "episodic", text))
        for text in _render_introductions(persona, rng, cfg.intro_repeats):
            examples.append(CorpusExample(pid, "intro", text))

        # Drift trajectories.
        for cond in ("in_persona", "drifting"):
            ypath = DRIFT_DIR / pid / f"{cond}.yaml"
            if ypath.exists():
                turns = _load_multiturn_yaml(ypath)
                rendered = _render_multiturn(turns)
                if rendered:
                    for _ in range(cfg.drift_repeats):
                        examples.append(
                            CorpusExample(pid, f"drift_{cond}", rendered)
                        )
            else:
                logger.warning("missing drift trajectory: {}", ypath)

        # Counterfactual probes (text fields only).
        probes_dir = PROBES_DIR / pid
        if probes_dir.exists():
            for ppath in sorted(probes_dir.glob("probe_*.yaml")):
                for text in _load_probe_yaml(ppath):
                    for _ in range(cfg.probe_repeats):
                        examples.append(CorpusExample(pid, "probe", text))

        # Chunk paragraphs (grounding for the probes).
        for paragraph in _load_probe_chunk_text(pid):
            for _ in range(cfg.chunk_repeats):
                examples.append(CorpusExample(pid, "chunk", paragraph))

    rng.shuffle(examples)
    n_train = int(len(examples) * cfg.train_fraction)
    train, test = examples[:n_train], examples[n_train:]

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(test_path, test)

    stats = {
        "n_total": len(examples),
        "n_train": len(train),
        "n_test": len(test),
        "n_personas": len(persona_paths),
    }
    by_persona = {p.stem: 0 for p in persona_paths}
    for ex in examples:
        if ex.persona_id in by_persona:
            by_persona[ex.persona_id] += 1
    stats["by_persona"] = by_persona  # type: ignore[assignment]
    by_source: dict[str, int] = {}
    for ex in examples:
        by_source[ex.source] = by_source.get(ex.source, 0) + 1
    stats["by_source"] = by_source  # type: ignore[assignment]

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "corpus built: {} examples ({} train / {} test); per-persona: {}",
        stats["n_total"],
        stats["n_train"],
        stats["n_test"],
        by_persona,
    )
    return stats


def _write_jsonl(path: Path, examples: list[CorpusExample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {"persona_id": ex.persona_id, "source": ex.source, "text": ex.text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def iter_jsonl(path: Path) -> Iterable[CorpusExample]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield CorpusExample(row["persona_id"], row["source"], row["text"])
