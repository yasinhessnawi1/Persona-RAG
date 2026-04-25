"""Contrastive-prompt generation for persona-vector extraction.

Takes a validated persona dict (the output of
``Persona.model_dump(mode="json")``) and deterministically builds two lists
of prompts:

- **in-persona** — prompts rendered with a system directive instructing the
  model to act *as* the persona.
- **out-of-persona** — prompts rendered with a system directive instructing
  the model to respond *ignoring* the persona, or (for worldview claims) to
  argue the *opposite* stance on the same topic.

Topics are drawn from the persona's ``identity.role``, ``self_facts``, and
``worldview`` fields so the two sets are topic-aligned (the contrast sits in
the stance, not the subject matter). Template-based generation is fully
deterministic and cheap; no LLM round-trip, no sampling, no API.

Why template-based and not LLM-generated? Reproducibility (seed-free).
The validation gate must replicate across sessions and hardware; a contrast
set produced by an LLM generating prompts would drift run-to-run. The
persona-vectors paper's pipeline is also deterministic — it pairs two system
prompts with a shared set of evaluation questions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Shared evaluation prompts — used across all personas. Mirrors the paper's
# "set of evaluation questions" that accompany the two opposing system
# prompts. Deliberately a mix of open and directive forms so the rendered
# text varies enough to expose real persona-specific activations (and not
# just "did the system prompt change").
_EVAL_QUESTIONS: tuple[str, ...] = (
    "Describe your approach to this subject.",
    "What do you think is most important in your work?",
    "How would you respond to a student asking for help?",
    "Which aspect of your field do you find most compelling?",
    "Walk me through how you would tackle a hard problem in your area.",
    "What kind of advice would you give a beginner?",
    "How do you evaluate competing claims in your field?",
    "What is a common misconception people have about your field?",
    "Describe a typical day in your professional life.",
    "What tools or methods do you rely on most?",
)

# Template strings for system directives. {role} gets the persona role;
# {topic} gets a single self-fact claim or worldview claim text.
_IN_PERSONA_SYSTEM_TEMPLATES: tuple[str, ...] = (
    "You are {role}. Respond in character, consistent with that role.",
    "You are {role}. Stay true to the persona in every response.",
    "You are {role}. Answer as you genuinely would in that role.",
    "You are {role}, answering faithfully to that identity.",
    "You are {role}. Maintain that perspective throughout.",
)

_OUT_PERSONA_SYSTEM_TEMPLATES: tuple[str, ...] = (
    "Ignore any prior persona. Respond as a generic assistant without any role.",
    "Set aside the role of {role}. Respond as a neutral, persona-less assistant.",
    "You are NOT {role}. Respond without adopting any particular persona.",
    "Forget the {role} framing. Respond as a general-purpose assistant.",
    "Disregard any persona instruction. Respond as a default chatbot.",
)

# Topic-aligned templates for self-facts and worldview claims. Every claim
# yields two prompts: one in-persona ("describe why you hold X") and one
# out-persona ("describe why someone would reject X").
_CLAIM_IN_TEMPLATES: tuple[str, ...] = (
    "Explain why the following is central to your thinking: {topic}",
    "In your own words, describe: {topic}",
    "How does this shape your approach? {topic}",
)

_CLAIM_OUT_TEMPLATES: tuple[str, ...] = (
    "Argue against the following claim as if you disagreed: {topic}",
    "Describe a perspective that rejects this view: {topic}",
    "Explain why someone might hold the opposite of: {topic}",
)


@dataclass(frozen=True, slots=True)
class ContrastSet:
    """A balanced pair of prompt lists for persona-vector extraction.

    Attributes
    ----------
    in_persona:
        List of fully-rendered prompts that elicit on-persona activations.
    out_persona:
        List of fully-rendered prompts that elicit off-persona activations.
        Same length as ``in_persona`` by construction.
    topic_aligned:
        ``True`` if the two lists draw from the same pool of topics in the
        same order — i.e. ``in_persona[i]`` and ``out_persona[i]`` target the
        same subject with opposite stance. Set by the builder, not by
        callers.
    seed:
        Seed used for any deterministic shuffling. ``0`` means "no shuffle."
    """

    in_persona: tuple[str, ...]
    out_persona: tuple[str, ...]
    topic_aligned: bool = True
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.in_persona) != len(self.out_persona):
            raise ValueError(
                f"ContrastSet unbalanced: in={len(self.in_persona)}, out={len(self.out_persona)}"
            )
        if len(self.in_persona) == 0:
            raise ValueError("ContrastSet must contain at least one pair")

    @property
    def n_pairs(self) -> int:
        """Number of contrast pairs on each side."""
        return len(self.in_persona)

    def sha256(self) -> str:
        """Stable content hash over the prompt content + seed.

        Used as the cache-invalidation key in the persona-vector cache
        (``meta.json.contrast_set_sha256``). Excludes metadata dict because
        metadata may change without altering the extraction inputs.
        """
        payload = json.dumps(
            {
                "in_persona": list(self.in_persona),
                "out_persona": list(self.out_persona),
                "topic_aligned": self.topic_aligned,
                "seed": self.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def split(self, test_fraction: float, seed: int = 42) -> tuple[ContrastSet, ContrastSet]:
        """Prompt-disjoint train/test split.

        Deterministic given ``seed``: indices are hash-partitioned so the same
        contrast set produces the same split on every invocation (addresses
        author's Q2 reproducibility note). Same-index pairs go to the same
        side of the split, preserving ``topic_aligned``.
        """
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction!r}")
        n = self.n_pairs
        # Hash-based deterministic partition — stable across Python processes
        # because int.from_bytes is deterministic. A simple shuffle with
        # random.Random(seed) would also be deterministic in practice, but
        # hash-based is future-proof against any stdlib shuffle algorithm
        # change.
        test_indices: set[int] = set()
        for i in range(n):
            key = hashlib.sha256(f"{seed}:{i}".encode()).digest()
            score = int.from_bytes(key[:8], "big") / (2**64)
            if score < test_fraction:
                test_indices.add(i)
        if not test_indices or len(test_indices) == n:
            # Degenerate edge: force at least one test and one train sample.
            test_indices = {n - 1} if not test_indices else set(range(n - 1))

        train_idx = sorted(set(range(n)) - test_indices)
        test_idx = sorted(test_indices)
        train = ContrastSet(
            in_persona=tuple(self.in_persona[i] for i in train_idx),
            out_persona=tuple(self.out_persona[i] for i in train_idx),
            topic_aligned=self.topic_aligned,
            seed=self.seed,
            metadata={**self.metadata, "split": "train", "split_seed": seed},
        )
        test = ContrastSet(
            in_persona=tuple(self.in_persona[i] for i in test_idx),
            out_persona=tuple(self.out_persona[i] for i in test_idx),
            topic_aligned=self.topic_aligned,
            seed=self.seed,
            metadata={**self.metadata, "split": "test", "split_seed": seed},
        )
        return train, test


class ContrastPromptGenerator:
    """Template-based contrastive-prompt generator.

    Uses the backend's
    :meth:`~persona_rag.models.base.LLMBackend.format_persona_prompt` so
    rendered prompts exercise the same chat-template path that downstream
    retrieval mechanisms exercise at inference. Without that, validation-
    time activations would be captured in a different regime from how drift
    is measured at inference.

    Parameters
    ----------
    backend:
        The backend whose chat template the prompts render through. For
        tests, a fake backend implementing ``format_persona_prompt`` is
        sufficient — no model load.
    n_pairs:
        Number of (in-persona, out-persona) pairs to produce per persona.
        Defaults to 50 per the spec. Raises if the persona has insufficient
        content to generate that many unique pairs.
    seed:
        Currently unused (generator is deterministic by template-order); kept
        for interface symmetry with the LLM-generated approach B (not
        implemented for semester).
    """

    def __init__(
        self,
        backend: Any,
        n_pairs: int = 50,
        *,
        seed: int = 0,
    ) -> None:
        if n_pairs < 2:
            raise ValueError(f"n_pairs must be >= 2, got {n_pairs}")
        self._backend = backend
        self._n_pairs = n_pairs
        self._seed = seed

    def generate(self, persona: dict[str, Any]) -> ContrastSet:
        """Build a :class:`ContrastSet` of ``n_pairs`` x 2 prompts.

        Pulls topics from ``persona["identity"]["role"]``,
        ``persona["self_facts"]`` (each ``{"fact": ...}``), and
        ``persona["worldview"]`` (each ``{"claim": ...}``). Cycles through
        system-directive templates and claim-framing templates to generate
        enough unique prompts.

        Raises
        ------
        ValueError
            If the persona has no role, no self-facts, and no worldview
            claims — there is literally nothing to contrast on.
        """
        identity = persona.get("identity") or {}
        role = identity.get("role")
        if not role:
            raise ValueError("persona.identity.role is required for contrast generation")

        # Each "topic" becomes one contrast pair. Pool of topics:
        # 1. The role itself (used with _EVAL_QUESTIONS as the content).
        # 2. Each self_fact claim (used with _CLAIM_*_TEMPLATES).
        # 3. Each worldview claim (used with _CLAIM_*_TEMPLATES).
        topics: list[tuple[str, str]] = []
        # Kind (1): role x eval-question — each eval question generates one topic.
        for q in _EVAL_QUESTIONS:
            topics.append(("role", q))
        # Kind (2): self-facts.
        for sf in persona.get("self_facts") or []:
            fact = sf.get("fact") if isinstance(sf, dict) else None
            if fact:
                topics.append(("self_fact", fact))
        # Kind (3): worldview claims.
        for wv in persona.get("worldview") or []:
            claim = wv.get("claim") if isinstance(wv, dict) else None
            if claim:
                topics.append(("worldview", claim))

        if not topics:
            raise ValueError(
                "persona has no contrast material (need role + eval questions, "
                "self_facts, or worldview claims)"
            )

        # Template cycling generates multiple prompts per topic until we
        # reach n_pairs. This keeps the contrast set topic-aligned and
        # balanced by construction.
        in_prompts: list[str] = []
        out_prompts: list[str] = []
        i = 0
        while len(in_prompts) < self._n_pairs:
            kind, topic_text = topics[i % len(topics)]
            sys_in = _IN_PERSONA_SYSTEM_TEMPLATES[
                (i // len(topics)) % len(_IN_PERSONA_SYSTEM_TEMPLATES)
            ].format(role=role)
            sys_out = _OUT_PERSONA_SYSTEM_TEMPLATES[
                (i // len(topics)) % len(_OUT_PERSONA_SYSTEM_TEMPLATES)
            ].format(role=role)
            if kind == "role":
                # Eval-question case — same user text for both sides.
                user_in = topic_text
                user_out = topic_text
            else:
                in_tmpl = _CLAIM_IN_TEMPLATES[i % len(_CLAIM_IN_TEMPLATES)]
                out_tmpl = _CLAIM_OUT_TEMPLATES[i % len(_CLAIM_OUT_TEMPLATES)]
                user_in = in_tmpl.format(topic=topic_text)
                user_out = out_tmpl.format(topic=topic_text)
            in_prompts.append(
                self._backend.format_persona_prompt(
                    system_text=sys_in,
                    user_text=user_in,
                    history=None,
                )
            )
            out_prompts.append(
                self._backend.format_persona_prompt(
                    system_text=sys_out,
                    user_text=user_out,
                    history=None,
                )
            )
            i += 1
            # Safety rail: if we've cycled through every template
            # combination and still haven't reached n_pairs, the persona
            # is under-sized for this n_pairs. Unique-prompt count is
            # bounded by topics x system_templates x claim_templates.
            max_unique = (
                len(topics) * len(_IN_PERSONA_SYSTEM_TEMPLATES) * max(len(_CLAIM_IN_TEMPLATES), 1)
            )
            if i > max_unique and len(in_prompts) < self._n_pairs:
                logger.warning(
                    "persona {!r}: only {} unique contrast prompts available (requested {}); "
                    "consider fewer n_pairs or adding self_facts / worldview claims",
                    persona.get("persona_id", "?"),
                    len(in_prompts),
                    self._n_pairs,
                )
                break

        return ContrastSet(
            in_persona=tuple(in_prompts[: self._n_pairs]),
            out_persona=tuple(out_prompts[: self._n_pairs]),
            topic_aligned=True,
            seed=self._seed,
            metadata={
                "method": "template",
                "role": role,
                "n_topics": len(topics),
                "persona_id": persona.get("persona_id"),
            },
        )
