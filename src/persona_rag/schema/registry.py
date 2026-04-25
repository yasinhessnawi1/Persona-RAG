"""Persona registration pipeline: load YAML, index four typed stores, hook persona vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from persona_rag.schema.chunker import chunk_persona
from persona_rag.schema.persona import Persona

if TYPE_CHECKING:  # pragma: no cover
    from persona_rag.stores.episodic_store import EpisodicStore
    from persona_rag.stores.identity_store import IdentityStore
    from persona_rag.stores.self_facts_store import SelfFactsStore
    from persona_rag.stores.worldview_store import WorldviewStore


class PersonaVectorExtractorLike(Protocol):
    """Duck-typed interface for the persona-vector extractor.

    This file intentionally does NOT import from `persona_rag.vectors` to keep
    the import graph one-way (vectors may import schema, not the reverse).
    Callers wanting vector extraction pass their own extractor instance.
    """

    def extract(self, persona: dict, contrast_set: Any) -> Any: ...


@dataclass
class RegisteredPersona:
    """Handle returned by `PersonaRegistry.register`.

    `vectors` is populated only if a `PersonaVectorExtractor` was supplied.
    When present it carries the concrete `PersonaVectors` object that
    downstream code (drift-signal computation, geometry analyses) consumes
    directly.
    """

    persona: Persona
    identity_store: IdentityStore
    self_facts_store: SelfFactsStore
    worldview_store: WorldviewStore
    episodic_store: EpisodicStore
    vectors: Any | None = None
    vectors_cache_path: Path | None = None


class PersonaRegistry:
    """One-shot registration pipeline.

    Steps:
      1. Load + validate YAML via `Persona.from_yaml`.
      2. Chunk + upsert into the four typed stores (idempotent by chunk id).
      3. If a `PersonaVectorExtractor` is supplied, extract and cache per-layer
         persona vectors; otherwise skip with a warning.
      4. Return a `RegisteredPersona` bundling the persona + store handles +
         (optional) vectors.
    """

    def __init__(
        self,
        identity_store: IdentityStore,
        self_facts_store: SelfFactsStore,
        worldview_store: WorldviewStore,
        episodic_store: EpisodicStore,
        *,
        vector_extractor: PersonaVectorExtractorLike | None = None,
        vectors_cache_dir: Path | str | None = None,
    ) -> None:
        self._identity = identity_store
        self._self_facts = self_facts_store
        self._worldview = worldview_store
        self._episodic = episodic_store
        self._extractor = vector_extractor
        self._vectors_cache_dir = Path(vectors_cache_dir) if vectors_cache_dir else None

    def register(self, yaml_path: Path | str) -> RegisteredPersona:
        """Load, validate, index, and (optionally) extract vectors for one persona."""
        yaml_path = Path(yaml_path)
        persona = Persona.from_yaml(yaml_path)
        logger.info("registering persona {!r} from {}", persona.persona_id, yaml_path)

        chunks = chunk_persona(persona)
        counts = {
            "identity": self._identity.index(chunks),
            "self_facts": self._self_facts.index(chunks),
            "worldview": self._worldview.index(chunks),
            "episodic": self._episodic.index(chunks),
        }
        logger.info("persona {!r} indexed: {}", persona.persona_id, counts)

        vectors = None
        cache_path = None
        if self._extractor is not None:
            vectors, cache_path = self._extract_and_cache_vectors(persona)
        else:
            logger.warning(
                "persona {!r}: persona-vector extraction SKIPPED — no extractor injected.",
                persona.persona_id,
            )

        return RegisteredPersona(
            persona=persona,
            identity_store=self._identity,
            self_facts_store=self._self_facts,
            worldview_store=self._worldview,
            episodic_store=self._episodic,
            vectors=vectors,
            vectors_cache_path=cache_path,
        )

    def _extract_and_cache_vectors(self, persona: Persona) -> tuple[Any | None, Path | None]:
        """Call the persona-vector extractor and cache the result to disk.

        The cache format (safetensors + meta.json) lives in
        :mod:`persona_rag.vectors.cache`; this method's sole responsibility
        is to call the extractor and hand back what it returns — the extractor
        itself owns serialisation.
        """
        assert self._extractor is not None
        # ContrastSet generation is the extractor's responsibility — we don't
        # construct it here. Callers that pass an extractor are expected to
        # configure it with its own contrast-set generator.
        vectors = self._extractor.extract(
            persona=persona.model_dump(mode="json"),
            contrast_set=None,  # extractor may build its own internally
        )
        cache_path: Path | None = None
        if self._vectors_cache_dir is not None:
            self._vectors_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._vectors_cache_dir / f"{persona.persona_id}.safetensors"
            logger.info(
                "persona {!r}: persona vectors extracted; cache path reserved at {}",
                persona.persona_id,
                cache_path,
            )
        return vectors, cache_path

    def delete(self, persona_id: str) -> dict[str, int]:
        """Remove a persona from every store. Returns per-store deletion counts."""
        return {
            "identity": self._identity.delete_persona(persona_id),
            "self_facts": self._self_facts.delete_persona(persona_id),
            "worldview": self._worldview.delete_persona(persona_id),
            "episodic": self._episodic.delete_persona(persona_id),
        }
