"""Sentence-transformers wrapper with separate query/passage encoding paths.

The wrapper enforces the asymmetric-retrieval prompt convention used by BGE
and E5 models: queries and passages are encoded through distinct methods
that apply the model-specific prefix internally. Callers cannot accidentally
use a single ``encode`` and forget the prefix — there is no such method.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger


class TextEncoder:
    """Asymmetric-retrieval encoder.

    Args:
        model_id: HuggingFace model id (e.g. ``BAAI/bge-large-en-v1.5``).
        query_prefix: Prefix prepended to each query before encoding.
            BGE: ``"Represent this sentence for searching relevant passages: "``.
            E5: ``"query: "``. Empty string for symmetric models.
        passage_prefix: Prefix prepended to each passage before encoding.
            BGE: empty. E5: ``"passage: "``.
        normalize: Whether to L2-normalise embeddings.
        batch_size: Per-batch encoding size.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        dtype: Compute dtype for the underlying model. ``"float16"`` is
            required on V100. ``"float32"`` on CPU.
    """

    def __init__(
        self,
        *,
        model_id: str,
        query_prefix: str,
        passage_prefix: str,
        normalize: bool = True,
        batch_size: int = 64,
        device: str = "auto",
        dtype: str = "float16",
    ) -> None:
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.dtype = dtype
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        logger.info(
            "loading sentence-transformers model_id={model_id} device={device} dtype={dtype}",
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
        )
        kwargs: dict[str, object] = {"device": self.device}
        # sentence-transformers accepts a `model_kwargs` dict to pass through to
        # the underlying transformers AutoModel; this is how we force fp16 on V100.
        if self.dtype != "float32":
            kwargs["model_kwargs"] = {"torch_dtype": _torch_dtype(self.dtype)}
        self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension reported by the loaded model."""

        model = self._load()
        return int(model.get_sentence_embedding_dimension())

    # -- public API ----------------------------------------------------

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode queries; the query prefix is applied internally.

        Returns a ``(len(texts), dim)`` float32 numpy array.
        """

        return self._encode([self.query_prefix + (t or "") for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode passages; the passage prefix is applied internally."""

        return self._encode([self.passage_prefix + (t or "") for t in texts])

    # -- internals -----------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dep
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
