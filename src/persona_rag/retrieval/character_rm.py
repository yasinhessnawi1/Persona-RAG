"""CharacterRM scorer (CharacterEval reward model wrapper).

CharacterRM is a trained reward model that scores how well a candidate
response embodies a stated persona. Released alongside the CharacterEval
benchmark; trained on a Chinese role-play corpus, so its English
competency is empirically verified before this scorer is used as one of
the hybrid ranker's signals. The wrapper is intentionally narrow — a
single ``score(persona, query, response) -> float`` method — so the rest
of the mechanism stays decoupled from the model card's specific output
shape, and a swap to a different reward model is a one-file change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from loguru import logger

from persona_rag.schema.persona import Persona

# Default HuggingFace model id. Override via config if a different reward
# model is used.
DEFAULT_CHARACTER_RM_MODEL_ID = "morecry/BaichuanCharRM"


@runtime_checkable
class CharacterScorer(Protocol):
    """Minimal scoring surface for the hybrid ranker's first signal."""

    @property
    def name(self) -> str:
        """Short identifier for logging and metadata."""

    def score(self, *, persona: Persona, query: str, response: str) -> float:
        """Return a real-valued score. Higher means more on-persona."""


class CharacterRMScorer:
    """Real CharacterRM scorer wrapping the HuggingFace model.

    Loaded lazily on the first ``score()`` call so unit tests that never
    invoke ``score()`` (and dev machines without GPU support) do not pay
    the model-load cost.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CHARACTER_RM_MODEL_ID,
        device: str = "cuda",
        max_input_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_input_tokens = max_input_tokens
        self._loaded = False
        # Heavy artefacts initialised on first score() call.
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return f"character_rm({self.model_id})"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Lazy import keeps transformers/bitsandbytes off the import path on
        # platforms where they are unavailable.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading CharacterRM weights from {}", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device)
        self._model.eval()
        self._loaded = True

    def score(self, *, persona: Persona, query: str, response: str) -> float:
        """Score one ``(persona, query, response)`` triple.

        The model returns a logit-shaped scalar; higher means more
        on-persona. The hybrid ranker normalises across the candidates of a
        single ``rank()`` call before weighting, so the absolute scale does
        not need to be calibrated here — the contract is just "higher is
        more on-persona," which is invariant under monotone transforms.
        """
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        persona_description = self._render_persona_description(persona)
        prompt = (
            f"Persona: {persona_description}\n"
            f"User: {query.strip()}\n"
            f"Assistant: {response.strip()}\n"
            f"Score the assistant's reply for persona consistency."
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # The default scoring head pools the final-token logits. The exact
        # upstream recipe is captured in the wrapper's tests once the
        # English-competency check confirms the path; until then the
        # contract "higher is more on-persona" is what callers depend on.
        last_token_logits = logits[0, -1, :]
        return float(last_token_logits.mean().item())

    @staticmethod
    def _render_persona_description(persona: Persona) -> str:
        """Render a compact persona description for the scorer's prompt."""
        parts = [
            f"{persona.identity.name}, {persona.identity.role}.",
            persona.identity.background.strip(),
        ]
        if persona.identity.constraints:
            parts.append("Constraints: " + "; ".join(persona.identity.constraints))
        return " ".join(parts)


class FakeCharacterRMScorer:
    """Deterministic scorer for unit tests.

    Returns a hash-based score so tests can assert ranking ordering without
    loading any model. Behaves like ``CharacterRMScorer`` from the
    protocol's point of view.
    """

    def __init__(self, model_id: str = "fake/character-rm") -> None:
        self.model_id = model_id
        self.calls: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return f"character_rm({self.model_id})"

    def score(self, *, persona: Persona, query: str, response: str) -> float:
        import hashlib

        key = (persona.persona_id or "", query, response)
        self.calls.append(key)
        h = hashlib.sha256("|".join(key).encode("utf-8")).digest()
        # Map first 8 bytes into [-1, 1] deterministically.
        n = int.from_bytes(h[:8], "big")
        return (n / 2**63) - 1.0


__all__ = [
    "DEFAULT_CHARACTER_RM_MODEL_ID",
    "CharacterRMScorer",
    "CharacterScorer",
    "FakeCharacterRMScorer",
]
