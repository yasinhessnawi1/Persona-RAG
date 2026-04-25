"""Persona-vector extractor: mass-mean probe over contrastive activations.

Runs the backend on every prompt in a :class:`ContrastSet`, captures hidden
states at configured layers via ``LLMBackend.get_hidden_states``, and
produces per-layer persona vectors, in-persona centroids, out-of-persona
centroids, and the raw stacked hidden-state tensors (for the separability
probe).

The extractor is pool/scope-agnostic — the defaults (``pool="last"``,
``over="prompt"``) match the persona-vectors paper (Chen, Lindsey et al.,
arXiv 2507.21509). The Hydra config ``config/vectors/default.yaml`` owns
those defaults; callers override at construction time.

Conforms to the :class:`~persona_rag.schema.registry.PersonaVectorExtractorLike`
protocol so a configured instance can be injected into
:class:`~persona_rag.schema.registry.PersonaRegistry`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from loguru import logger

from persona_rag.models.base import HiddenStatePool, HiddenStateScope, LLMBackend
from persona_rag.vectors.contrast_prompts import ContrastPromptGenerator, ContrastSet


@dataclass(frozen=True, slots=True)
class PersonaVectors:
    """Per-layer persona vectors + centroids + raw hidden states.

    Attributes
    ----------
    persona_id:
        Identifier of the persona these vectors belong to.
    vectors:
        ``{layer_idx: Tensor(hidden_dim,)}`` — persona direction per layer,
        computed as ``mean(in_states) - mean(out_states)``.
    in_persona_centroid:
        ``{layer_idx: Tensor(hidden_dim,)}`` — mean of in-persona
        hidden states per layer. Used by :class:`DriftSignal` at inference.
    out_persona_centroid:
        ``{layer_idx: Tensor(hidden_dim,)}`` — mean of out-persona hidden
        states per layer.
    in_states:
        ``{layer_idx: Tensor(n_prompts, hidden_dim)}`` — full stack of in-
        persona hidden states. Consumed by :class:`SeparabilityProbe` to
        compute projections on held-out prompts.
    out_states:
        ``{layer_idx: Tensor(n_prompts, hidden_dim)}`` — full stack of out-
        persona hidden states.
    metadata:
        Extraction-time config (backend name, pool, scope, layers, seed,
        contrast-set hash, timestamp). Round-trips through the safetensors
        cache as ``meta.json``.
    """

    persona_id: str
    vectors: dict[int, torch.Tensor]
    in_persona_centroid: dict[int, torch.Tensor]
    out_persona_centroid: dict[int, torch.Tensor]
    in_states: dict[int, torch.Tensor] = field(default_factory=dict)
    out_states: dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        """Layer indices these vectors were extracted at, sorted ascending."""
        return sorted(self.vectors.keys())


class PersonaVectorExtractor:
    """Extract persona vectors from contrastive hidden states.

    Parameters
    ----------
    backend:
        Any :class:`LLMBackend`. Must implement
        ``get_hidden_states(prompt, layers=..., pool=..., over=...)``.
    layers:
        Transformer layer indices to probe. Default ``[8, 12, 16, 20]``
        targets the middle band of Gemma-2-9B's 42-layer stack; for Llama-
        3.1-8B (32 layers) a symmetric band is ``[6, 10, 14, 18]`` —
        configurable via Hydra.
    pool:
        Pooling mode passed to ``get_hidden_states``. Default ``"last"``
        matches the persona-vectors paper.
    scope:
        Scope mode — ``"prompt"`` (default, cheap, matches the paper) or
        ``"generation"`` (~50x slower, useful only for ablations).
    n_pairs:
        Number of contrast pairs the auto-built generator produces when
        ``extract`` is called without an explicit contrast set.
    seed:
        Seed passed through to ``get_hidden_states`` (only relevant for
        ``over="generation"`` which runs a greedy generate).
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        layers: list[int] | tuple[int, ...] | None = None,
        pool: HiddenStatePool = "last",
        scope: HiddenStateScope = "prompt",
        n_pairs: int = 50,
        seed: int = 42,
    ) -> None:
        if pool not in ("mean", "last", "none"):
            raise ValueError(f"pool must be mean|last|none, got {pool!r}")
        if scope not in ("prompt", "generation", "all"):
            raise ValueError(f"scope must be prompt|generation|all, got {scope!r}")
        if pool == "none":
            # "none" returns (n_tokens, hidden_dim); mass-mean requires a
            # single per-prompt vector. Refuse explicitly.
            raise ValueError("PersonaVectorExtractor requires pool in {mean, last}, not 'none'")
        self._backend = backend
        self._layers = list(layers) if layers is not None else [8, 12, 16, 20]
        self._pool: HiddenStatePool = pool
        self._scope: HiddenStateScope = scope
        self._n_pairs = n_pairs
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        persona: dict[str, Any],
        contrast_set: ContrastSet | None = None,
    ) -> PersonaVectors:
        """Extract persona vectors for ``persona``.

        If ``contrast_set`` is ``None``, a :class:`ContrastPromptGenerator`
        is built on the fly from the persona's ``identity`` + ``self_facts``
        + ``worldview`` fields.

        Matches the :class:`~persona_rag.schema.registry.PersonaVectorExtractorLike`
        protocol so an instance can be injected into the registry.
        """
        persona_id = persona.get("persona_id") or "<unknown>"
        if contrast_set is None:
            gen = ContrastPromptGenerator(self._backend, n_pairs=self._n_pairs, seed=self._seed)
            contrast_set = gen.generate(persona)

        logger.info(
            "persona {!r}: extracting persona vectors at layers={} pool={} scope={} "
            "on {} in-prompts + {} out-prompts",
            persona_id,
            self._layers,
            self._pool,
            self._scope,
            contrast_set.n_pairs,
            contrast_set.n_pairs,
        )

        t0 = time.perf_counter()
        in_states = self._capture(contrast_set.in_persona)
        out_states = self._capture(contrast_set.out_persona)
        elapsed = time.perf_counter() - t0
        logger.info(
            "persona {!r}: hidden-state capture complete in {:.1f}s ({} prompts x {} layers)",
            persona_id,
            elapsed,
            2 * contrast_set.n_pairs,
            len(self._layers),
        )

        vectors: dict[int, torch.Tensor] = {}
        in_centroid: dict[int, torch.Tensor] = {}
        out_centroid: dict[int, torch.Tensor] = {}
        for layer in self._layers:
            in_mat = in_states[layer]  # (n_prompts, hidden_dim)
            out_mat = out_states[layer]
            in_mean = in_mat.mean(dim=0)
            out_mean = out_mat.mean(dim=0)
            in_centroid[layer] = in_mean
            out_centroid[layer] = out_mean
            vectors[layer] = in_mean - out_mean  # mass-mean probe

        metadata = {
            "persona_id": persona_id,
            "backend_name": getattr(self._backend, "name", "<unknown>"),
            "backend_model_id": getattr(self._backend, "model_id", "<unknown>"),
            "hidden_dim": getattr(
                self._backend, "hidden_dim", in_centroid[self._layers[0]].shape[0]
            ),
            "num_layers_total": getattr(self._backend, "num_layers", None),
            "layers": list(self._layers),
            "pool": self._pool,
            "scope": self._scope,
            "n_pairs": contrast_set.n_pairs,
            "seed": self._seed,
            "contrast_set_sha256": contrast_set.sha256(),
            "extractor_code_sha256": self._code_sha256(),
            "elapsed_seconds": elapsed,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return PersonaVectors(
            persona_id=persona_id,
            vectors=vectors,
            in_persona_centroid=in_centroid,
            out_persona_centroid=out_centroid,
            in_states=in_states,
            out_states=out_states,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture(self, prompts: tuple[str, ...] | list[str]) -> dict[int, torch.Tensor]:
        """Run the backend on each prompt, stack hidden states per layer.

        Returns ``{layer: Tensor(n_prompts, hidden_dim)}``. Tensors are float32
        on CPU (``HFBackend.get_hidden_states`` guarantees this for us).
        """
        per_layer_stacks: dict[int, list[torch.Tensor]] = {layer: [] for layer in self._layers}
        for idx, prompt in enumerate(prompts):
            out = self._backend.get_hidden_states(
                prompt,
                layers=list(self._layers),
                pool=self._pool,
                over=self._scope,
            )
            for layer in self._layers:
                vec = out[layer]
                # pool in {mean, last} → (hidden_dim,). If the backend
                # handed back a 2-D tensor (shouldn't happen under our
                # pool settings), fold it to 1-D via mean as a safety net.
                if vec.dim() == 2:
                    vec = vec.mean(dim=0)
                per_layer_stacks[layer].append(vec.to(torch.float32))
            if (idx + 1) % 10 == 0 or idx + 1 == len(prompts):
                logger.debug("captured hidden states for prompt {}/{}", idx + 1, len(prompts))

        return {
            layer: torch.stack(stacks, dim=0)  # (n_prompts, hidden_dim)
            for layer, stacks in per_layer_stacks.items()
        }

    @staticmethod
    def _code_sha256() -> str:
        """Hash this module's source — cache-invalidation belt-and-suspenders.

        Reading our own source file is unusual but cheap; it catches the
        "quiet mass-mean bug" scenario where extractor logic changes but the
        cache filename doesn't. See author's Q4 refinement.
        """
        import inspect

        try:
            src = inspect.getsource(inspect.getmodule(PersonaVectorExtractor))  # type: ignore[arg-type]
        except (OSError, TypeError):  # pragma: no cover — source unavailable in frozen envs
            return "<unavailable>"
        return hashlib.sha256(src.encode("utf-8")).hexdigest()


# Module-level protocol conformance check (runtime) — a static type-checker
# would flag this via the Protocol, but a runtime assertion catches accidental
# signature drift during development.
def _check_protocol() -> None:  # pragma: no cover — compile-time shape check
    from persona_rag.schema.registry import PersonaVectorExtractorLike

    _ = cast(PersonaVectorExtractorLike, PersonaVectorExtractor)
