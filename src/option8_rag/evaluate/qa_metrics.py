"""Short-answer evaluation: exact match and token-level F1.

Implementation follows the SQuAD/NQ convention: lower-case, strip
punctuation, drop articles, normalise whitespace.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation."""

    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(t for t in text.split() if t not in _ARTICLES)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """1.0 if prediction matches any reference after normalisation."""

    if not references:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(r) for r in references))


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Best F1 across references, comparing on whitespace tokens of normalised text."""

    if not references:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def aggregate(
    *,
    predictions: Iterable[str],
    references: Iterable[Sequence[str]],
) -> dict[str, float]:
    """Macro EM and F1 over a parallel iterable of predictions and references."""

    em_scores: list[float] = []
    f1_scores: list[float] = []
    for pred, refs in zip(predictions, references, strict=True):
        em_scores.append(exact_match(pred, list(refs)))
        f1_scores.append(token_f1(pred, list(refs)))
    if not em_scores:
        return {"em": 0.0, "f1": 0.0}
    return {
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }
