"""Grounded answer synthesis using a chat-template-driven LLM.

The generator builds a single prompt from a query plus a list of retrieved
chunks, instructs the model to ground its answer in those chunks and to cite
them by chunk identifier, and returns the decoded answer text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from option8_rag.types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. Answer the user's "
    "question using only the provided context passages. If the answer "
    "cannot be derived from the context, say so explicitly. Cite the "
    "passages you used inline using their chunk identifiers in square "
    "brackets, e.g. [c1]."
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Decoder configuration for the synthesiser."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


_DEFAULT_GENERATION = GenerationConfig()


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Bundle of an answer text and the prompt that produced it (for logs)."""

    answer: str
    prompt_messages: list[dict[str, str]]


class GroundedGenerator:
    """Llama-style chat-template generator with a grounded RAG prompt.

    Args:
        model_id: HuggingFace model id.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        dtype: Compute dtype name. ``"float16"`` on V100.
        attn_implementation: Passed through to transformers
            ``AutoModelForCausalLM.from_pretrained``. Default ``"sdpa"`` is
            fine for Llama 3.1; Gemma 2 must use ``"eager"``.
        generation: Decoder knobs.
        context_top_k: How many retrieved chunks to feed into the prompt.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        generation: GenerationConfig = _DEFAULT_GENERATION,
        context_top_k: int = 5,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.generation = generation
        self.context_top_k = context_top_k
        self._tokenizer = None
        self._model = None
        self._supports_system_role_cached: bool | None = None

    # -- chat-template helpers -----------------------------------------

    def _supports_system_role(self) -> bool:
        """Return True if the loaded chat template accepts a `system` role."""

        if self._supports_system_role_cached is not None:
            return self._supports_system_role_cached
        try:
            self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [
                    {"role": "system", "content": "probe"},
                    {"role": "user", "content": "probe"},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            self._supports_system_role_cached = True
        except Exception as exc:
            if "System role not supported" in str(exc):
                self._supports_system_role_cached = False
            else:
                raise
        return self._supports_system_role_cached

    @staticmethod
    def _fold_system_into_first_user(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Prepend any system-turn content to the first user turn."""

        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if not sys_parts or not rest:
            return rest or messages
        sys_blob = "\n\n".join(sys_parts)
        first = rest[0]
        rewritten = {**first, "content": f"{sys_blob}\n\n{first['content']}"}
        return [rewritten, *rest[1:]]

    # -- model loading -------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        torch_dtype = _torch_dtype(self.dtype)
        logger.info(
            "loading generator model_id={model_id} device={device} dtype={dtype} attn={attn}",
            model_id=self.model_id,
            device=device,
            dtype=self.dtype,
            attn=self.attn_implementation,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch_dtype,
            "attn_implementation": self.attn_implementation,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        if device == "cpu":
            model = model.to("cpu")
        model.eval()

        # Some HF model defaults ship with `do_sample=True` baked into
        # generation_config.json (Llama-3.1 ships temperature=0.6,
        # top_p=0.9). Even when we pass `do_sample=False` to .generate()
        # the model's own config has been observed to bleed into the
        # decoding path on certain transformers releases. Pin the config
        # explicitly to remove the ambiguity.
        try:
            from transformers import GenerationConfig as HfGenerationConfig

            model.generation_config = HfGenerationConfig(
                do_sample=self.generation.do_sample,
                temperature=self.generation.temperature if self.generation.do_sample else None,
                top_p=self.generation.top_p if self.generation.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception:
            logger.warning("could not set generation_config; relying on model default")

        self._tokenizer = tokenizer
        self._model = model
        # Light sanity: allocate device cache.
        if device == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release the model and its GPU memory so the next stage has room."""

        import gc

        self._tokenizer = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:  # pragma: no cover
            pass

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:  # pragma: no cover
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- public API ----------------------------------------------------

    def generate(
        self,
        *,
        query: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Generate a grounded answer."""

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        messages = self._build_messages(query=query, retrieved=retrieved)
        if not self._supports_system_role():
            messages = self._fold_system_into_first_user(messages)
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model.device)

        import torch

        # Greedy decoding: pass sampling knobs only when do_sample=True
        # so transformers doesn't warn about an ignored temperature.
        gen_kwargs: dict[str, object] = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.generation.do_sample:
            gen_kwargs["temperature"] = self.generation.temperature
            gen_kwargs["top_p"] = self.generation.top_p

        # torch.compile / dynamo is incompatible with accelerate's
        # CPU-offload pre-forward hooks (we hit
        # `torch._dynamo.exc.Unsupported: call_method UserFunctionVariable
        # _compiled_fn` on Gemma-2-9b under offload). Disable dynamo for
        # generation; we don't benefit from compiled inference here
        # (16-token judge calls, 512-token answers).
        try:
            import torch._dynamo as torch_dynamo

            torch_dynamo.config.suppress_errors = True
        except ImportError:  # pragma: no cover
            pass

        with torch.inference_mode():
            output_ids = self._model.generate(inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs.shape[-1] :]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GeneratedAnswer(answer=answer, prompt_messages=messages)

    # -- prompt construction ------------------------------------------

    def _build_messages(
        self,
        *,
        query: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> list[dict[str, str]]:
        snippets = list(retrieved)[: self.context_top_k]
        if not snippets:
            user_content = (
                f"Question: {query}\n\n"
                "(no context passages were retrieved)\n\n"
                "Please answer the question if possible, otherwise state "
                "that you don't have enough information."
            )
        else:
            blocks: list[str] = []
            for i, item in enumerate(snippets, start=1):
                tag = f"c{i}"
                blocks.append(
                    f"[{tag}] (chunk_id={item.chunk.chunk_id})\n{item.chunk.text}",
                )
            joined = "\n\n".join(blocks)
            user_content = (
                f"Context passages:\n\n{joined}\n\n"
                f"Question: {query}\n\n"
                "Answer the question using only the context above. Cite "
                "passages with their tags in square brackets."
            )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]


def _torch_dtype(dtype: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return mapping[dtype]
