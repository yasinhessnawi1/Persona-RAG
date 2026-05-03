"""Tokenizer wrapper for the off-spec Free-Transformer experiment.

Re-uses the Gemma-2 tokenizer (already required for the rest of the project) so
the persona text is tokenised the same way it would be downstream. We do NOT
load any model weights here — the tokenizer is a pure rust/sentencepiece
component and works on Darwin without bitsandbytes.

Default model id: ``google/gemma-2-9b-it``. Authentication is handled by the
Hugging Face cache that ``huggingface-cli login`` already populated for the
project.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer


@dataclass
class TokenizerBundle:
    """Container for the components the trainer needs."""

    tokenizer: AutoTokenizer
    pad_token_id: int
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tokens, mask) of length max_len. Right-pad with pad_token_id."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [self.tokenizer.bos_token_id or self.eos_token_id, *ids]
        ids = [*ids[: max_len - 1], self.eos_token_id]
        attn = [1] * len(ids) + [0] * (max_len - len(ids))
        ids = [*ids, *([self.pad_token_id] * (max_len - len(ids)))]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def load_tokenizer(model_id: str = "google/gemma-2-9b-it") -> TokenizerBundle:
    tok = AutoTokenizer.from_pretrained(model_id)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
        tok.pad_token_id = pad_id
    return TokenizerBundle(
        tokenizer=tok,
        pad_token_id=pad_id,
        eos_token_id=tok.eos_token_id,
        vocab_size=tok.vocab_size,
    )
