"""Shared CLI helpers: logging setup, seed setting, path resolution."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with a single stderr sink at the requested level."""

    logger.remove()
    logger.add(sys.stderr, level=level)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (if importable) for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_paths(cfg: DictConfig) -> dict[str, Path]:
    """Resolve the ``paths.*`` block to absolute :class:`Path` instances."""

    raw: dict[str, Any] = OmegaConf.to_container(cfg.paths, resolve=True)  # type: ignore[assignment]
    out: dict[str, Path] = {}
    for k, v in raw.items():
        out[k] = Path(str(v)).expanduser().resolve()
        out[k].mkdir(parents=True, exist_ok=True)
    return out


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Materialise a Hydra config to a plain dict (resolved)."""

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
