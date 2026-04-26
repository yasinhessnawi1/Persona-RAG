"""BM25 tokenization helper — single code path for index AND query (avoids score collapse).

Mismatched tokenization between index time and query time silently collapses
BM25 scores (any term that gets lowercased on one side but not the other no
longer matches). We centralise the call to ``bm25s.tokenize`` here so both legs
of the retrieval go through the same configuration.
"""

from __future__ import annotations

from collections.abc import Iterable

# bm25s is imported lazily inside `tokenize` so this module can be inspected at
# import time without the dependency installed (relevant for type checking).


def tokenize(texts: Iterable[str] | str) -> list[list[str]]:
    """Tokenize one or many texts using bm25s' default English pipeline.

    Returns a list of token-lists. A single string input still returns a
    list-of-one so the caller doesn't branch on type.
    """
    import bm25s

    texts = [texts] if isinstance(texts, str) else list(texts)
    if not texts:
        return []
    # bm25s.tokenize returns a Tokenized namedtuple-like with .ids / .vocab
    # when stopwords is enabled, OR a list-of-token-lists when return_ids=False.
    # We force return_ids=False to get plain str tokens — easier to debug and
    # to assert in tests.
    out = bm25s.tokenize(texts, lower=True, stopwords="en", return_ids=False)
    # bm25s in 0.3.x returns list[list[str]] when return_ids=False.
    return list(out)
