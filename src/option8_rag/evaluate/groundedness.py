"""LLM-as-judge groundedness scorer.

The judge is a chat-template LLM (different family from the generator to
limit self-grading bias). For each (question, answer, retrieved-passages)
triplet the judge is asked twice with two different rubrics; both outputs
are parsed to integers in ``{0, 1, 2}``, normalised to ``[0, 1]``, and
averaged. If a parse fails the corresponding sub-score is dropped; if
both fail the triplet receives ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from option8_rag.types import RetrievedChunk

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Your job is to decide whether an "
    "answer to a question is supported by the provided context passages. "
    "You must respond with a single integer rating only, no explanation."
)

CLAIM_RUBRIC = (
    "Rate how well the answer is grounded in the context passages.\n"
    "0 — the answer makes claims that are not supported by any passage, "
    "or directly contradicts a passage.\n"
    "1 — the answer is partially grounded; some claims are supported and "
    "some are not.\n"
    "2 — every substantive claim in the answer is supported by at least "
    "one passage.\n\n"
    "Respond with a single integer: 0, 1, or 2."
)

OVERALL_RUBRIC = (
    "Decide whether the answer is faithful to the context as a whole.\n"
    "0 — not faithful: the answer adds material that is not in the "
    "context, contradicts the context, or fabricates citations.\n"
    "1 — mostly faithful: the answer is broadly consistent with the "
    "context but contains minor unsupported additions.\n"
    "2 — fully faithful: the answer is completely supported by the "
    "context, with no fabrication.\n\n"
    "Respond with a single integer: 0, 1, or 2."
)


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Decoder configuration for the LLM-as-judge."""

    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


_DEFAULT_JUDGE_CONFIG = JudgeConfig()


@dataclass(frozen=True, slots=True)
class GroundednessScore:
    """Per-triplet groundedness output."""

    claim_score: float | None
    overall_score: float | None
    fused: float | None


class GroundednessJudge:
    """LLM-as-judge groundedness scorer (chat-template model).

    Args:
        model_id: HuggingFace model id of the judge LLM.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        dtype: ``"float16"`` on V100.
        attn_implementation: For Gemma 2, must be ``"eager"``.
        config: Decoder configuration.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        dtype: str = "float16",
        attn_implementation: str = "eager",
        config: JudgeConfig = _DEFAULT_JUDGE_CONFIG,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.config = config
        self._tokenizer = None
        self._model = None
        self._supports_system_role_cached: bool | None = None

    def _supports_system_role(self) -> bool:
        """Detect whether the judge's chat template accepts a `system` role.

        Probes the template once with a trivial system message and caches
        the result. Gemma's template raises ``TemplateError: System role
        not supported`` and that's the signal we want to catch; any other
        Jinja error we treat as "supported" so the real error surfaces at
        the actual call site.
        """

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
                # Re-raise on unexpected template failures so we don't
                # silently misclassify them as "no system support".
                raise
        return self._supports_system_role_cached

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        torch_dtype = _torch_dtype(self.dtype)
        logger.info(
            "loading judge model_id={model_id} device={device} dtype={dtype} attn={attn}",
            model_id=self.model_id,
            device=device,
            dtype=self.dtype,
            attn=self.attn_implementation,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        kwargs: dict[str, object] = {
            "torch_dtype": torch_dtype,
            "attn_implementation": self.attn_implementation,
        }
        if device == "cuda":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if device == "cpu":
            model = model.to("cpu")
        model.eval()

        # Pin generation_config so the model's shipped defaults
        # (Gemma-2-9b ships do_sample=True, temperature=0.6, top_p=0.9)
        # cannot bleed into our greedy-decoding path.
        try:
            from transformers import GenerationConfig as HfGenerationConfig

            model.generation_config = HfGenerationConfig(
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception:
            logger.warning("could not set judge generation_config; relying on model default")

        self._tokenizer = tokenizer
        self._model = model
        if device == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release the judge model and its GPU memory."""

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

    def score(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> GroundednessScore:
        """Return a groundedness score for one (q, a, retrieved) triplet."""

        self._load()
        ctx = _format_context(retrieved)
        claim = self._ask(question=question, answer=answer, context=ctx, rubric=CLAIM_RUBRIC)
        overall = self._ask(
            question=question,
            answer=answer,
            context=ctx,
            rubric=OVERALL_RUBRIC,
        )
        fused = _fuse(claim, overall)
        return GroundednessScore(claim_score=claim, overall_score=overall, fused=fused)

    # -- internals -----------------------------------------------------

    def _ask(self, *, question: str, answer: str, context: str, rubric: str) -> float | None:
        assert self._tokenizer is not None
        assert self._model is not None

        user_body = (
            f"Context passages:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer to evaluate: {answer}\n\n"
            f"{rubric}"
        )
        # Some chat templates (notably Gemma's) reject a `system` role and
        # raise "System role not supported" inside Jinja. Detect the case
        # by trying with a system message and falling back to a single
        # user turn that prepends the system prompt. Cached so we only
        # pay the failed-template cost once per judge instance.
        if not self._supports_system_role():
            messages = [
                {
                    "role": "user",
                    "content": f"{JUDGE_SYSTEM_PROMPT}\n\n{user_body}",
                },
            ]
        else:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_body},
            ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model.device)

        import torch

        # Greedy decoding: keep `do_sample=False` and drop the sampling
        # knobs to avoid the transformers warning about temperature being
        # ignored when sampling is off.
        gen_kwargs: dict[str, object] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        # Disable dynamo: it's incompatible with accelerate's CPU-offload
        # hooks (raises torch._dynamo.exc.Unsupported on Gemma-2 under
        # offload) and we get nothing from compiled inference for ~16
        # output tokens.
        try:
            import torch._dynamo as torch_dynamo

            torch_dynamo.config.suppress_errors = True
        except ImportError:  # pragma: no cover
            pass

        with torch.inference_mode():
            output_ids = self._model.generate(inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs.shape[-1] :]
        decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _parse_rating(decoded)


def _parse_rating(text: str) -> float | None:
    """Parse the first integer in ``{0, 1, 2}`` and return it normalised to ``[0, 1]``."""

    match = re.search(r"[0-2]", text)
    if match is None:
        return None
    value = int(match.group())
    return value / 2.0


def _fuse(claim: float | None, overall: float | None) -> float | None:
    valid = [s for s in (claim, overall) if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _format_context(retrieved: Sequence[RetrievedChunk]) -> str:
    if not retrieved:
        return "(no context passages)"
    blocks: list[str] = []
    for i, item in enumerate(retrieved, start=1):
        blocks.append(f"[c{i}] (chunk_id={item.chunk.chunk_id})\n{item.chunk.text}")
    return "\n\n".join(blocks)


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
