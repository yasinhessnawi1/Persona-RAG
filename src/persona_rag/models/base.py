"""Abstract interface shared by every LLM backend in the project.

Backends wrap a HuggingFace causal-LM, load it in 4-bit (NF4, double-quant), and
expose four capabilities:

1. Text generation from a raw prompt (``generate``) or a chat message list
   (``chat``).
2. Batched generation for evaluation throughput.
3. Hidden-state capture from arbitrary transformer layers, with pooling and
   scope selection — used by persona-vector extraction and downstream geometry
   work.
4. Backend-agnostic persona-prompt formatting via ``format_persona_prompt`` —
   abstracts the Gemma-vs-Llama ``system``-role asymmetry.

The protocol is intentionally small. Backend-specific quirks (chat-template
rendering, stop tokens, BOS/EOS handling) live in the concrete subclass. See
:mod:`persona_rag.models.gemma` and :mod:`persona_rag.models.llama`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

Role = str  # "system" | "user" | "assistant". Plain str to keep call sites lightweight.


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single chat turn. Backend decides how (or whether) to render each role."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation parameters the LLMBackend honors.

    Kept as a dataclass for ergonomic passing into ``chat()`` and ``generate_batch()``;
    the spec's top-level ``generate()`` signature accepts the fields directly as kwargs
    and internally constructs one of these.

    Greedy decoding is triggered by ``do_sample=False`` (the default, used for
    reproducibility checks). ``seed`` is set before every call regardless of
    sampling mode so stochastic runs are at least reproducible within a single
    process.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int | None = None


HiddenStatePool = Literal["mean", "last", "none"]
HiddenStateScope = Literal["generation", "prompt", "all"]


@runtime_checkable
class LLMBackend(Protocol):
    """The contract every concrete backend implements.

    Concrete backends MUST:

    - Load the model in 4-bit (NF4 + double-quant) with compute_dtype picked up from
      the Hydra config (fp16 on V100, bf16 on Ampere+).
    - Use ``attn_implementation="eager"`` — required by Gemma 2 for softcap
      correctness, mirrored on Llama 3.1 for kernel-path symmetry.
    - Set random seeds at the start of every generation call (CPU + CUDA).
    - Return plain Python strings from generation methods (no tokenizer artifacts, no
      role markers, no special tokens).
    - Perform a NaN/inf warm-up guard on load — fail loudly if the quantized + fp16
      stack is producing invalid logits before any user-facing generation happens.
    - Return detached CPU tensors from ``get_hidden_states`` to free GPU memory.
    """

    @property
    def name(self) -> str:
        """Short human-readable identifier (e.g. ``"gemma2-9b-it"``)."""

    @property
    def model_id(self) -> str:
        """Hugging Face model repo id, e.g. ``"google/gemma-2-9b-it"``."""

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""

    @property
    def hidden_dim(self) -> int:
        """Hidden state dimensionality (per-token embedding size)."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Generate text from a raw prompt string (no chat template applied).

        ``temperature=0.0`` plus ``top_p=1.0`` selects greedy decoding.
        """

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        cfg: GenerationConfig | None = None,
    ) -> str:
        """Generate a reply given a list of chat messages.

        Backend renders the messages with the model's chat template (or an equivalent
        substitute — Gemma 2 folds ``system`` into the first user turn per decision
        #004b).
        """

    def generate_batch(
        self,
        prompts: list[str],
        *,
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate one reply per prompt. Order of returned list matches ``prompts``."""

    def get_hidden_states(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
        pool: HiddenStatePool = "mean",
        over: HiddenStateScope = "generation",
    ) -> dict[int, torch.Tensor]:
        """Capture hidden states from selected transformer layers.

        Returns ``{layer_index: tensor}``. Tensor shape depends on ``pool``:

        - ``"mean"``: ``(hidden_dim,)`` — mean across the selected token scope.
        - ``"last"``: ``(hidden_dim,)`` — last token in scope.
        - ``"none"``: ``(seq_len, hidden_dim)`` — one row per token in scope.

        ``over`` selects which tokens contribute:

        - ``"generation"``: the model-output tokens only. Requires running generation.
        - ``"prompt"``: the input tokens only. No generation performed.
        - ``"all"``: prompt + generation tokens concatenated.

        If ``layers`` is ``None``, every transformer layer is returned (layer 0 =
        embedding output, layers 1..L = transformer blocks).

        Tensors are returned on CPU with gradients detached, so callers don't have to
        worry about leaking the GPU.
        """

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Render a persona-conditioned prompt for this backend.

        Both backends accept a ``system_text`` string and fold it in according to
        their own template rules:

        - Llama 3.1: native ``system`` role.
        - Gemma 2: no ``system`` role; prepended to the first user turn.

        ``history`` is an optional list of past chat turns (user/assistant) threaded
        before the final ``user_text`` turn. Returns a fully-rendered string ready to
        pass to ``generate()``.
        """
