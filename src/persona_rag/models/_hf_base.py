"""Shared HuggingFace causal-LM implementation used by the Gemma 2 and Llama 3.1 backends.

Kept internal (underscore prefix) because the public contract is :class:`LLMBackend` —
subclasses may freely share this plumbing, but external callers should program against
the protocol, not this class.

Why a shared base instead of copy-pasting into each backend?
    Loading, seeding, hidden-state extraction, and batched generation are identical
    across the two backends we care about. Only the chat-template handling differs
    (Gemma 2 has no ``system`` role; Llama 3.1 has two EOS tokens). Subclasses override
    ``_render_chat``, ``_stop_token_ids``, and ``format_persona_prompt``.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from persona_rag.models.base import (
    ChatMessage,
    GenerationConfig,
    HiddenStatePool,
    HiddenStateScope,
    LLMBackend,
)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HFBackendConfig:
    """Load-time config for a HuggingFace causal-LM backend.

    Attributes
    ----------
    model_id:
        HuggingFace repo id, e.g. ``"google/gemma-2-9b-it"``.
    name:
        Short stable identifier used in logs, run_ids, and wandb.
    revision:
        Optional git revision / commit sha for full reproducibility. If ``None``, HF
        resolves the current ``main`` at load time (recorded into the load log so we
        at least know which sha we got).
    compute_dtype:
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``. On V100 (CC 7.0)
        this MUST be ``"float16"`` (no hardware bf16). The Hydra config owns
        this knob so we can flip to ``bfloat16`` unchanged on Ampere+.
    attn_implementation:
        Must be ``"eager"`` for Gemma 2 (softcap correctness). Mirrored on
        Llama 3.1 for fair comparison.
    load_in_4bit / bnb_4bit_quant_type / bnb_4bit_use_double_quant:
        Quantization knobs. Defaults match the project standard (NF4 +
        double-quant).
    max_input_tokens:
        Hard cap on tokenized prompt length passed to the model. Protects
        against OOM on a 32 GB V100.
    trust_remote_code:
        Both Gemma 2 and Llama 3.1 are stock transformers architectures, so
        ``False``.
    warmup_nan_guard:
        If True (default), run a short warm-up after load: a few short prompts,
        check logits for NaN/inf, raise ``RuntimeError`` on trigger. A broader
        stability suite runs in the smoke test; this on-load guard is the
        cheap "did we just load something broken?" check that fires before
        any user-facing generation.
    """

    model_id: str
    name: str
    revision: str | None = None
    compute_dtype: str = "float16"
    attn_implementation: str = "eager"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    max_input_tokens: int = 4096
    trust_remote_code: bool = False
    warmup_nan_guard: bool = True


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name]
    except KeyError as err:
        raise ValueError(
            f"Unknown compute_dtype {name!r}; must be one of {sorted(_DTYPE_MAP)}"
        ) from err


def _seed_everything(seed: int) -> None:
    """Set every seed we can, for greedy-decoding reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers' own seeder covers numpy + tf if present


def _peak_gpu_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


class NaNGuardError(RuntimeError):
    """Raised when the NaN/inf guard detects invalid logits from the model.

    On Gemma 2 + fp16 + V100, this usually means the transformers pin slipped below
    4.49 (PR #35398 not present) or a softcap path is being skipped. Fail loudly.
    """


# ------------------------------------------------------------------------------
# Backend
# ------------------------------------------------------------------------------


class HFBackend(LLMBackend):
    """Concrete HuggingFace causal-LM backend. Subclasses override chat rendering only.

    Construction loads the model eagerly. This is deliberate: a 9B-parameter 4-bit
    model takes ~20s to load on a V100, and we want that cost paid at startup, not on
    the first generation.
    """

    def __init__(self, cfg: HFBackendConfig) -> None:
        self._cfg = cfg
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading {name} from {repo}@{rev} (dtype={dt}, attn={attn}, 4bit={q})",
            name=cfg.name,
            repo=cfg.model_id,
            rev=cfg.revision or "main",
            dt=cfg.compute_dtype,
            attn=cfg.attn_implementation,
            q=cfg.load_in_4bit,
        )

        quant_cfg = None
        if cfg.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=_resolve_dtype(cfg.compute_dtype),
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
        )
        # Llama 3.1 ships without a pad token; use eos for right-padding during batches.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only models require LEFT padding for batched generate(): otherwise
        # the model attends to pad tokens at positions before the real prompt tail and
        # produces garbage continuations.
        self._tokenizer.padding_side = "left"

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            quantization_config=quant_cfg,
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
            trust_remote_code=cfg.trust_remote_code,
            # torch_dtype is ignored when quantization_config is set, but set for the
            # (unquantized) fallback path.
            torch_dtype=_resolve_dtype(cfg.compute_dtype),
        )
        self._model.eval()

        # Disable transformers' automatic torch.compile on generate() for Gemma 2
        # (and, defensively, for Llama). On transformers 4.49 + bitsandbytes 0.43,
        # Gemma 2's generation path tries to compile the forward, and TorchDynamo
        # cannot trace bitsandbytes' ``Params4bit.t()``:
        #   torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable(Params4bit) t
        #
        # transformers 4.49's ``generation_config.validate()`` rejects ``None`` for
        # these fields — it wants either a proper CompileConfig instance or the
        # attribute absent. So we ``delattr`` them instead of nulling them. Same
        # effect: transformers falls back to eager generation.
        if hasattr(self._model, "generation_config"):
            gen_conf = self._model.generation_config
            # Gemma 2's checkpoint ships ``cache_implementation="hybrid"``, which
            # triggers transformers 4.49 to JIT-compile the generation forward via
            # TorchDynamo. Dynamo cannot trace bitsandbytes' ``Params4bit.t()``
            # (4-bit quantized weight transpose), and the call fails. The accepted
            # non-compile value is ``None`` — ``validate()`` short-circuits with
            # ``if self.cache_implementation is not None and ... not in ALL_IMPLS``.
            # ``delattr`` is NOT safe (``validate()`` reads the attribute directly),
            # nor is any non-None value (they're all compile-triggering). So we keep
            # the attribute present and set it to ``None``. ``compile_config`` is only
            # consulted when ``cache_implementation`` asks for a compiled cache, so
            # nulling ``cache_implementation`` makes its value moot — we leave it
            # untouched to avoid its own validator path.
            if hasattr(gen_conf, "cache_implementation"):
                gen_conf.cache_implementation = None
            # Llama 3.1's generation_config ships with ``temperature=0.6`` and
            # ``top_p=0.9``, which triggers per-call warnings on greedy generate()
            # unless reset. ``validate()`` accesses these unconditionally, so we
            # keep them present at the sampling-neutral default of 1.0.
            if getattr(gen_conf, "temperature", 1.0) != 1.0:
                with contextlib.suppress(AttributeError):
                    gen_conf.temperature = 1.0
            if getattr(gen_conf, "top_p", 1.0) != 1.0:
                with contextlib.suppress(AttributeError):
                    gen_conf.top_p = 1.0

        peak_gb = _peak_gpu_memory_gb()
        logger.info(
            "{name} loaded. layers={L}, hidden={H}, peak_mem={mem}",
            name=cfg.name,
            L=self._model.config.num_hidden_layers,
            H=self._model.config.hidden_size,
            mem=f"{peak_gb:.2f} GB" if peak_gb is not None else "n/a (cpu)",
        )

        if cfg.warmup_nan_guard:
            self._warmup_nan_guard()

    # ------------------------ LLMBackend properties ------------------------

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model_id(self) -> str:
        return self._cfg.model_id

    @property
    def num_layers(self) -> int:
        return int(self._model.config.num_hidden_layers)

    @property
    def hidden_dim(self) -> int:
        return int(self._model.config.hidden_size)

    # ------------------------ Subclass extension points ------------------------

    def _render_chat(self, messages: list[ChatMessage]) -> str:
        """Default: use the tokenizer's native chat template unchanged.

        Subclasses that need to fold the ``system`` role (Gemma 2) override this.
        """
        return self._tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _stop_token_ids(self) -> list[int]:
        """Default: only the tokenizer's canonical eos_token_id.

        Subclasses that have multiple valid stops (Llama 3.1's ``<|eot_id|>``) override
        this.
        """
        eos = self._tokenizer.eos_token_id
        return [eos] if eos is not None else []

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Default behavior: system as a ``system`` role, history untouched.

        Used by :class:`~persona_rag.models.llama.LlamaBackend`. Gemma overrides to
        fold ``system_text`` into the first user turn.
        """
        messages: list[ChatMessage] = []
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_text))
        return self._render_chat(messages)

    # ------------------------ Generation ------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Protocol-compliant generate.

        ``temperature=0.0`` (with any ``top_p``) dispatches to greedy decoding — the
        idiom transformers uses. For sampling, pass ``temperature>0``.
        """
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
            top_p=top_p,
            do_sample=temperature > 0,
            seed=seed,
        )
        return self.generate_batch([prompt], cfg=cfg)[0]

    def chat(self, messages: list[ChatMessage], *, cfg: GenerationConfig | None = None) -> str:
        rendered = self._render_chat(messages)
        cfg = cfg or GenerationConfig()
        return self.generate(
            rendered,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        cfg = cfg or GenerationConfig()
        if cfg.seed is not None:
            _seed_everything(cfg.seed)

        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._cfg.max_input_tokens,
        ).to(self._model.device)

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        stops = self._stop_token_ids()
        if stops:
            gen_kwargs["eos_token_id"] = stops if len(stops) > 1 else stops[0]
        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        with torch.inference_mode():
            out = self._model.generate(**enc, **gen_kwargs)

        prompt_lens = enc["input_ids"].shape[1]
        new_tokens = out[:, prompt_lens:]
        return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # ------------------------ Hidden states ------------------------

    def get_hidden_states(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
        pool: HiddenStatePool = "mean",
        over: HiddenStateScope = "generation",
        max_new_tokens: int = 64,
        seed: int | None = 0,
    ) -> dict[int, torch.Tensor]:
        """Capture hidden states from selected layers.

        ``over="prompt"`` runs a single forward pass (cheap, deterministic).
        ``over="generation"`` and ``over="all"`` run greedy generation with
        ``output_hidden_states=True`` so the per-step hidden states are captured,
        then stacked along the sequence dimension. ``max_new_tokens`` controls how
        far we generate for those two scopes; 64 is a reasonable default for persona-
        vector extraction and is overridable per-call.

        Returns a dict keyed by layer index. Tensor shape depends on ``pool``:
        ``(hidden_dim,)`` for ``"mean"`` or ``"last"``; ``(n_tokens, hidden_dim)`` for
        ``"none"``. Tensors are on CPU, detached.
        """
        if pool not in ("mean", "last", "none"):
            raise ValueError(f"pool must be one of mean|last|none, got {pool!r}")
        if over not in ("generation", "prompt", "all"):
            raise ValueError(f"over must be one of generation|prompt|all, got {over!r}")

        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._cfg.max_input_tokens,
        ).to(self._model.device)
        int(enc["input_ids"].shape[1])

        # layer indices: 0 = embedding output; 1..L = transformer blocks.
        # ``hidden_states`` tuple has length L+1 both for forward and for generate()
        # (per step).
        total_layers = self.num_layers + 1
        wanted = layers if layers is not None else list(range(total_layers))
        for idx in wanted:
            if not 0 <= idx < total_layers:
                raise ValueError(
                    f"Layer {idx} out of range; valid 0..{total_layers - 1} "
                    f"(0=embedding, 1..{total_layers - 1}=transformer blocks)"
                )

        if over == "prompt":
            # One forward pass. hidden_states[l] has shape (1, prompt_len, hidden_dim).
            with torch.inference_mode():
                out = self._model(**enc, output_hidden_states=True, return_dict=True)
            per_layer: dict[int, torch.Tensor] = {}
            for idx in wanted:
                hs = out.hidden_states[idx].squeeze(0)  # (prompt_len, hidden_dim)
                per_layer[idx] = hs.detach().to("cpu", dtype=torch.float32)
            return {idx: _pool(per_layer[idx], pool) for idx in wanted}

        # over in {"generation", "all"} — need to run generation.
        if seed is not None:
            _seed_everything(seed)
        with torch.inference_mode():
            gen_out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=(
                    self._stop_token_ids()
                    if len(self._stop_token_ids()) > 1
                    else (self._stop_token_ids()[0] if self._stop_token_ids() else None)
                ),
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # generate() returns hidden_states as: Tuple[step] -> Tuple[layer] -> Tensor.
        # For step 0 (the prompt pass), each layer tensor is (1, prompt_len, hidden_dim).
        # For subsequent steps, each layer tensor is (1, 1, hidden_dim) — the single
        # newly-generated token. Stack them appropriately per scope.
        per_step_hs = gen_out.hidden_states  # tuple of tuples
        if not per_step_hs:
            # No new tokens generated (e.g., immediate EOS). Fall back to prompt only.
            with torch.inference_mode():
                out = self._model(**enc, output_hidden_states=True, return_dict=True)
            prompt_layers = {
                idx: out.hidden_states[idx].squeeze(0).detach().to("cpu", dtype=torch.float32)
                for idx in wanted
            }
            return {idx: _pool(prompt_layers[idx], pool) for idx in wanted}

        # Assemble per-layer token-sequence tensors.
        per_layer_scope: dict[int, torch.Tensor] = {}
        for idx in wanted:
            # Step 0: prompt tokens.
            step0_full = per_step_hs[0][idx].squeeze(0)  # (prompt_len, hidden_dim)
            # Steps 1..K: one generated token each.
            gen_rows = [per_step_hs[s][idx].squeeze(0) for s in range(1, len(per_step_hs))]
            # Each gen_rows[i] is shape (1, hidden_dim) or (hidden_dim,); normalize.
            gen_rows = [r.reshape(-1, r.shape[-1]) for r in gen_rows]
            if over == "generation":
                stacked = torch.cat(gen_rows, dim=0) if gen_rows else step0_full[-1:].clone()
            else:  # "all"
                stacked = torch.cat([step0_full, *gen_rows], dim=0)
            per_layer_scope[idx] = stacked.detach().to("cpu", dtype=torch.float32)

        return {idx: _pool(per_layer_scope[idx], pool) for idx in wanted}

    # ------------------------ NaN guard ------------------------

    def _warmup_nan_guard(self) -> None:
        """Run 10 short prompts, check logits for NaN/inf. Raise if any found.

        Load-time NaN guard. The 30-prompt stability suite in the smoke test is
        the broader stability check; this is the cheap pre-flight check that
        catches a broken quantization stack before any user call.
        """
        warmup_prompts = [
            "Hello.",
            "The capital of France is",
            "One plus one equals",
            "Water is composed of",
            "The Sun rises in the",
            "Shakespeare wrote plays such as",
            "In mathematics, pi is approximately",
            "A triangle has",
            "The Pacific Ocean is located between",
            "Rain falls from",
        ]
        logger.info(
            "{name}: running NaN/inf warm-up guard on {n} prompts",
            name=self.name,
            n=len(warmup_prompts),
        )

        enc = self._tokenizer(
            warmup_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self._model.device)

        with torch.inference_mode():
            out = self._model(**enc, return_dict=True)
        logits = out.logits
        has_nan = bool(torch.isnan(logits).any().item())
        has_inf = bool(torch.isinf(logits).any().item())
        if has_nan or has_inf:
            raise NaNGuardError(
                f"{self.name}: NaN/inf detected in warm-up logits "
                f"(nan={has_nan}, inf={has_inf}). Likely a quantization / transformers "
                f"version issue — see decisions #004, #005, #006. "
                f"Check that transformers>=4.49 and attn_implementation='eager' are both in effect."
            )
        logger.info(
            "{name}: warm-up OK (logit min={lo:.2f}, max={hi:.2f})",
            name=self.name,
            lo=float(logits.min().item()),
            hi=float(logits.max().item()),
        )

    def check_logits_finite(self, prompt: str) -> tuple[bool, dict[str, float]]:
        """Public NaN/inf check — single prompt, returns (ok, stats).

        Used by the smoke test and by ``tests/test_gemma.py`` + ``tests/test_llama.py``
        to exercise the guard path without forcing a raise at load.
        """
        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._cfg.max_input_tokens,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model(**enc, return_dict=True)
        logits = out.logits
        has_nan = bool(torch.isnan(logits).any().item())
        has_inf = bool(torch.isinf(logits).any().item())
        stats = {
            "min": float(logits.min().item()) if not (has_nan or has_inf) else float("nan"),
            "max": float(logits.max().item()) if not (has_nan or has_inf) else float("nan"),
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
        return (not has_nan and not has_inf), stats

    # ------------------------ Misc ------------------------

    def save_load_report(self, path: Path) -> None:
        """Write a tiny JSON report of the load for the smoke test to pick up."""
        peak_gb = _peak_gpu_memory_gb()
        data = {
            "name": self.name,
            "model_id": self.model_id,
            "revision": self._cfg.revision,
            "compute_dtype": self._cfg.compute_dtype,
            "attn_implementation": self._cfg.attn_implementation,
            "load_in_4bit": self._cfg.load_in_4bit,
            "bnb_4bit_quant_type": self._cfg.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self._cfg.bnb_4bit_use_double_quant,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "peak_gpu_mem_gb": peak_gb,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------
# Pooling helper
# ------------------------------------------------------------------------------


def _pool(x: torch.Tensor, pool: HiddenStatePool) -> torch.Tensor:
    """Apply pool to a (n_tokens, hidden_dim) tensor. Returns either
    (n_tokens, hidden_dim) for ``"none"`` or (hidden_dim,) for mean/last.
    """
    if x.dim() != 2:
        raise ValueError(f"expected (n_tokens, hidden_dim) tensor, got shape {tuple(x.shape)}")
    if pool == "none":
        return x
    if pool == "mean":
        return x.mean(dim=0)
    if pool == "last":
        return x[-1]
    raise ValueError(f"unknown pool {pool!r}")  # pragma: no cover — caught earlier
