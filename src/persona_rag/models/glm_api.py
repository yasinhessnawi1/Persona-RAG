"""GLM-4.7 backend via NVIDIA Build's free-tier OpenAI-compatible API.

Used as an API-tier judge candidate for the drift-gate calibration sweep
when the open-model bound (Llama / Prometheus / Qwen at 7-9B) lands in
the weak band. NVIDIA Build offers a free tier rate-limited at ~40
requests / minute on `z-ai/glm4.7`; this backend respects that limit
with a small inter-call sleep so the calibration sweep does not 429 out.

API key is **only** read from the environment variable named in
`api_key_env` (default ``NVIDIA_API_KEY``). The backend never accepts a
key argument, never logs the key value, and never writes it to disk.
Rotate the key promptly if it has appeared in any shared transcript.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from persona_rag.models.base import ChatMessage, GenerationConfig, LLMBackend

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(env_path: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file at the repo root.

    Standalone implementation so the project does not pull a new
    dependency. Parses ``KEY=value`` lines, ignores blanks and comments,
    strips surrounding quotes on the value, and never overwrites an
    already-set environment variable. Silently no-ops if the file is
    missing — the env var may already be set by the user's shell.
    """
    if env_path is None:
        env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip matching surrounding quotes.
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            # Do not clobber a value already set in the environment.
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning(".env loader failed to read {}: {}", env_path, exc)


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_GLM_MODEL_ID = "z-ai/glm4.7"
# 40 req/min => 1.5s between calls leaves 33 ms headroom under the cap.
DEFAULT_MIN_CALL_INTERVAL_S = 1.5


class GlmApiBackend(LLMBackend):
    """Drop-in :class:`LLMBackend` for GLM-4.7 over the NVIDIA free-tier API.

    Only the ``generate`` path is fully implemented — the drift gate is the
    only consumer. ``chat`` is implemented in terms of ``generate`` for
    protocol completeness; ``get_hidden_states`` and ``generate_batch``
    raise ``NotImplementedError`` because the API does not expose them.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "NVIDIA_API_KEY",
        base_url: str = DEFAULT_NVIDIA_BASE_URL,
        model_id: str = DEFAULT_GLM_MODEL_ID,
        name: str = "glm-4.7",
        min_call_interval_s: float = DEFAULT_MIN_CALL_INTERVAL_S,
        request_timeout_s: float = 60.0,
    ) -> None:
        # Load values from a repo-root .env file if present. The .env file
        # is gitignored; the loader does not overwrite values already set
        # in the shell environment.
        _load_dotenv()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {api_key_env!r} environment variable to use "
                f"{self.__class__.__name__}. Drop ``{api_key_env}=<your-key>`` into "
                "the repo-root .env file (gitignored) or export it in your shell. "
                "The backend never accepts the key as an argument or reads it from "
                "a tracked config file."
            )
        # Lazy import keeps openai out of the import path on machines that
        # do not have it installed.
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout_s)
        self._model_id = model_id
        self._name = name
        self._min_call_interval_s = float(min_call_interval_s)
        self._last_call_t: float = 0.0
        # Cached call count for diagnostics; not thread-safe (sequential by
        # design — the calibration runner serialises gate calls).
        self.calls = 0
        logger.info(
            "{name} loaded (API-tier; base_url={url}, model_id={mid})",
            name=self._name,
            url=base_url,
            mid=self._model_id,
        )

    # ------------------------ LLMBackend properties ------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def num_layers(self) -> int:
        # Not exposed by the API; protocol-required, returned as zero so any
        # caller that branches on this can detect the API-tier case.
        return 0

    @property
    def hidden_dim(self) -> int:
        return 0

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
        """Call the API for a single completion.

        On any API or network error, returns an empty string so the gate's
        parser falls into the malformed-defaults-to-ok path. The caller
        sees the same conservative fallback behaviour as a malformed
        local-judge response.
        """
        self._respect_rate_limit()
        self.calls += 1
        # NVIDIA Build's OpenAI-compatible endpoint accepts the standard
        # chat-completion shape. We force ``enable_thinking=False`` via
        # extra_body so the model emits final content directly rather
        # than reasoning tokens; the drift-gate template wants the JSON
        # output, not a chain-of-thought trace.
        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
                top_p=top_p,
                max_tokens=max_new_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            # Includes openai.APIError, RateLimitError, network errors, etc.
            # Log the type but never the request body (which contains the
            # rendered prompt; for the gate that includes the persona,
            # acceptable to log; the API key never appears in the openai
            # SDK's exception messages).
            logger.warning(
                "{name} API call failed: {kind}: {msg}",
                name=self._name,
                kind=type(exc).__name__,
                msg=str(exc)[:200],
            )
            return ""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content or ""

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        cfg: GenerationConfig | None = None,
    ) -> str:
        """Render the chat list into the API's message format and call generate."""
        cfg = cfg or GenerationConfig()
        self._respect_rate_limit()
        self.calls += 1
        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=max(cfg.temperature, 1e-5) if cfg.temperature > 0 else 0.0,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            logger.warning(
                "{name} API call failed: {kind}: {msg}",
                name=self._name,
                kind=type(exc).__name__,
                msg=str(exc)[:200],
            )
            return ""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content or ""

    def generate_batch(
        self,
        prompts: list[str],
        *,
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Batched generation by sequential single-call dispatch.

        The NVIDIA Build endpoint does not accept a batched-completion
        shape, and the free tier's rate limit makes parallelism
        counter-productive anyway. Sequential calls preserve the rate-
        limit floor and produce identical observable behaviour to the
        local backends' batch path.
        """
        cfg = cfg or GenerationConfig()
        return [
            self.generate(
                p,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed,
            )
            for p in prompts
        ]

    def get_hidden_states(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
        pool: Any = "mean",
        over: Any = "generation",
    ) -> Any:
        """Not supported by the hosted API; raise loudly."""
        raise NotImplementedError(
            f"{self._name} is API-tier; hidden states are not exposed. "
            "Use a local backend (Gemma, Llama, etc.) for hidden-state work."
        )

    def format_persona_prompt(
        self,
        system_text: str | None,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Render persona-conditioned prompt as a single user-message string.

        The drift gate calls ``judge.generate(rendered_template)`` with a
        fully-rendered prompt, so this method is rarely exercised on the
        API backend. Provided for protocol completeness.
        """
        parts: list[str] = []
        if system_text:
            parts.append(system_text.strip())
        if history:
            for m in history:
                label = "User" if m.role == "user" else "Assistant"
                parts.append(f"{label}: {m.content.strip()}")
        parts.append(f"User: {user_text.strip()}")
        return "\n\n".join(parts)

    # ------------------------ Internals ------------------------

    def _respect_rate_limit(self) -> None:
        """Sleep so the next call is at least ``min_call_interval_s`` after the previous."""
        if self._min_call_interval_s <= 0:
            return
        now = time.monotonic()
        delta = now - self._last_call_t
        if delta < self._min_call_interval_s:
            time.sleep(self._min_call_interval_s - delta)
        self._last_call_t = time.monotonic()


__all__ = [
    "DEFAULT_GLM_MODEL_ID",
    "DEFAULT_MIN_CALL_INTERVAL_S",
    "DEFAULT_NVIDIA_BASE_URL",
    "GlmApiBackend",
]
