"""Short-answer evaluation: exact match, token-level F1, and contains-match.

Implementation follows the SQuAD/NQ convention: lower-case, strip
punctuation, drop articles, normalise whitespace.

The ``contains_match`` metric is a deliberately permissive check that
credits a prediction whenever (after normalisation) it contains the
reference as a substring. It exists because instruction-tuned RAG
generators tend to wrap the bare fact in a sentence ("According to
[c1], the answer is X") which extractive EM/F1 punish severely without
that meaning the system was wrong. Contains-match is paired with EM/F1
in reports — it is *not* a replacement.
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


def contains_match(prediction: str, references: Sequence[str]) -> float:
    """1.0 if any normalised reference is a substring of the normalised prediction.

    This is the lenient counterpart to :func:`exact_match`. It credits a
    sentence-wrapped answer that still contains the bare fact, e.g.
    prediction "the course leader is Morten Goodwin" and reference
    "Morten Goodwin" both normalise to forms where the latter is a
    substring of the former.
    """

    if not references:
        return 0.0
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    for r in references:
        ref_norm = normalize_answer(r)
        if ref_norm and ref_norm in pred_norm:
            return 1.0
    return 0.0


def aggregate(
    *,
    predictions: Iterable[str],
    references: Iterable[Sequence[str]],
) -> dict[str, float]:
    """Macro EM, F1, and contains-match over parallel predictions and references."""

    em_scores: list[float] = []
    f1_scores: list[float] = []
    contains_scores: list[float] = []
    for pred, refs in zip(predictions, references, strict=True):
        ref_list = list(refs)
        em_scores.append(exact_match(pred, ref_list))
        f1_scores.append(token_f1(pred, ref_list))
        contains_scores.append(contains_match(pred, ref_list))
    if not em_scores:
        return {"em": 0.0, "f1": 0.0, "contains": 0.0}
    return {
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "contains": sum(contains_scores) / len(contains_scores),
    }
