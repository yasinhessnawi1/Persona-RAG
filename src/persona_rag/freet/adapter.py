"""Free-Transformer Z-injection on a frozen pretrained backbone.

Architecture (see `docs/research/free_transformer_adapter.md`):

  * Pretrained Gemma-2-9B (or any HF causal LM) loaded in 4-bit, FROZEN.
  * After block L/2 of the backbone, a non-causal encoder block reads the
    hidden state and produces per-sequence persona logits.
  * Softmax + straight-through one-hot Z (n_personas categories).
  * `z_to_residual` projects Z to a (T, dim) residual R.
  * R is added to the hidden_states fed to block L/2 + 1 via a forward
    pre-hook on that block.
  * Trainable parameters: encoder block, persona head, ζ query embedding,
    z_to_residual. ~150M new params on top of a 9B frozen base.

The adapter does not subclass / patch the backbone's attention layer — the
hook keeps the implementation robust to transformers version changes. This
is a documented deviation from the paper's K/V-only injection (§3.2) — the
adapter adds R to the residual stream rather than to K/V alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from persona_rag.freet.model import (
    FreeTransformerConfig,
    FreeTransformerEncoder,
    SupervisedPersonaHead,
)


@dataclass
class FreetAdapterConfig:
    """Adapter-only hyperparameters. Backbone shape is read from the loaded model."""

    n_personas: int = 3
    encoder_mlp_ratio: int = 4
    encoder_norm_eps: float = 1e-5
    inject_after_layer: int | None = None
    """Index of the backbone block AFTER WHICH R is injected (i.e. R is added
    to the input of block ``inject_after_layer + 1``). Defaults to L // 2."""


# --------------------------------------------------------------------------- #
# Hook registry                                                               #
# --------------------------------------------------------------------------- #


class _ResidualInjector:
    """A small holder that buffers R and adds it to the next forward call.

    A forward pre-hook is registered on the chosen backbone block. Each
    forward pass, the adapter's encoder produces R; we stash it on the
    injector; the hook adds it to the block's first positional argument
    (the hidden_states tensor). After the hook fires we clear R so a
    subsequent forward without setting R is a no-op.
    """

    def __init__(self) -> None:
        self._r: torch.Tensor | None = None

    def set(self, r: torch.Tensor | None) -> None:
        self._r = r

    def hook(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        if self._r is None:
            return args, kwargs
        r = self._r
        self._r = None  # one-shot
        # transformers' decoder layers receive `hidden_states` either as the
        # first positional arg or as a keyword. Handle both.
        if args:
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return args, kwargs
            new = hidden + r.to(hidden.dtype)
            args = (new, *args[1:])
            return args, kwargs
        if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs = {**kwargs, "hidden_states": kwargs["hidden_states"] + r.to(kwargs["hidden_states"].dtype)}
            return args, kwargs
        return args, kwargs


# --------------------------------------------------------------------------- #
# Backbone introspection                                                      #
# --------------------------------------------------------------------------- #


def _backbone_decoder_blocks(model: nn.Module) -> nn.ModuleList:
    """Locate the list of decoder blocks on a HF causal LM.

    Works for the standard transformers shape: ``model.model.layers`` for
    Llama / Gemma / Qwen-style architectures. Also accepts ``model.layers``.
    """
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(
            "could not find decoder block list on backbone — expected "
            "model.model.layers or model.layers"
        )
    return layers


def _backbone_hidden_dim(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "hidden_size"):
        raise ValueError("backbone has no .config.hidden_size")
    return int(cfg.hidden_size)


def _backbone_num_heads(model: nn.Module) -> tuple[int, int]:
    """Return (num_attention_heads, num_kv_heads). Falls back if KV-heads not exposed."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("backbone has no .config")
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_q))
    return n_q, n_kv


# --------------------------------------------------------------------------- #
# The adapter module                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class FreetAdapterOutput:
    logits: torch.Tensor
    persona_logits: torch.Tensor
    z: torch.Tensor


class FreetAdapter(nn.Module):
    """Wraps a frozen HF causal LM with a Z-injection encoder + head.

    The backbone is expected to already be on a CUDA device (loaded in 4-bit
    via bitsandbytes for V100). All backbone parameters are set to
    ``requires_grad=False`` here as a safety belt.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: FreetAdapterConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        layers = _backbone_decoder_blocks(backbone)
        n_layers = len(layers)
        if cfg.inject_after_layer is None:
            inject_idx = n_layers // 2
        else:
            inject_idx = int(cfg.inject_after_layer)
        if not (0 <= inject_idx < n_layers - 1):
            raise ValueError(
                f"inject_after_layer must be in [0, {n_layers - 2}]; got {inject_idx}"
            )
        self.inject_after_layer = inject_idx

        dim = _backbone_hidden_dim(backbone)
        n_q_heads, n_kv_heads = _backbone_num_heads(backbone)

        # Build a FreeTransformerConfig that the encoder block can consume.
        # Only the fields used by the encoder + its TransformerBlock are
        # populated meaningfully; the rest carry placeholders.
        self._encoder_cfg = FreeTransformerConfig(
            vocab_size=1,                       # unused (no LM head on encoder)
            dim=dim,
            n_layers=2,                         # unused (encoder has its own block)
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            mlp_ratio=cfg.encoder_mlp_ratio,
            max_seq_len=4096,
            norm_eps=cfg.encoder_norm_eps,
            latent_bits=1,                      # unused (supervised mode)
            inject_after_layer=0,               # unused
            pad_token_id=0,
            tie_embeddings=False,
            latent_mode="supervised",
            n_personas=cfg.n_personas,
        )
        self.encoder = FreeTransformerEncoder(self._encoder_cfg)
        self.persona_head = SupervisedPersonaHead(cfg.n_personas)
        self.z_to_residual = nn.Linear(cfg.n_personas, dim, bias=False)
        # Init z_to_residual small so first-step residual is near zero —
        # backbone behaviour should match its pretrained baseline at step 0.
        nn.init.normal_(self.z_to_residual.weight, std=0.001)

        # Hook plumbing.
        self._injector = _ResidualInjector()
        self._hook_handle = layers[inject_idx + 1].register_forward_pre_hook(
            self._injector.hook, with_kwargs=True
        )
        self._capture: dict[str, torch.Tensor] = {}
        self._capture_handle = layers[inject_idx].register_forward_hook(
            self._capture_hook
        )

        # RoPE for the encoder's non-causal attention. Must match the encoder
        # config's head_dim (which we set to backbone's). Built on demand on
        # first forward to land on the correct device.
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    # ----- internal hooks -----

    def _capture_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: tuple | torch.Tensor,
    ) -> None:
        # transformers' decoder layers return a tuple whose first element is
        # the hidden state. Some return just the tensor.
        hidden = output[0] if isinstance(output, tuple) else output
        if isinstance(hidden, torch.Tensor):
            self._capture["x_half"] = hidden

    def _ensure_rope(self, t: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self._encoder_cfg.head_dim
        if (
            self._rope_cos is None
            or self._rope_cos.shape[0] < t
            or self._rope_cos.device != device
        ):
            from persona_rag.freet.model import _build_rope_cache  # local import to avoid cycles

            cos, sin = _build_rope_cache(
                max(t, 4096), head_dim, self._encoder_cfg.rope_theta, device
            )
            self._rope_cos = cos.to(dtype)
            self._rope_sin = sin.to(dtype)
        return self._rope_cos[:t].to(dtype), self._rope_sin[:t].to(dtype)

    # ----- public API -----

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @contextmanager
    def _arm_residual(self, r: torch.Tensor) -> Iterator[None]:
        """Stash R for the next backbone forward pass; clear afterwards."""
        self._injector.set(r)
        try:
            yield
        finally:
            self._injector.set(None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        fixed_z_index: int | None = None,
    ) -> FreetAdapterOutput:
        """Run a forward pass.

        Args:
            input_ids: (B, T) int64 token ids.
            attention_mask: (B, T) 1/0 mask.
            fixed_z_index: when set, skip the encoder and clamp Z to a one-hot
                of this index. Used for generation-time Z-steering tests.

        Returns:
            FreetAdapterOutput with LM logits, per-sequence persona logits,
            and the (B, T, n_personas) one-hot Z.
        """
        b, t = input_ids.shape
        # 1) First-half forward pass to capture x_half. We need to run the
        #    backbone twice when training: once to get x_half (encoder
        #    input), once to get final logits with R injected. To keep this
        #    single-pass, we use the captured x_half from the previous
        #    call's pre-injection forward. But that would require two
        #    sequential forwards — unavoidable for a clean implementation.
        #
        # Pragmatic approach: do ONE backbone forward; the capture hook
        # records x_half at block L/2; the residual injector injects R
        # *during the same forward* into block L/2+1's input. Because
        # forward_hook runs *after* the layer completes (so x_half is the
        # output of block L/2), and forward_pre_hook on block L/2+1 runs
        # *before* its forward, the capture happens before the next layer
        # starts — but R wasn't computed yet. So we need two passes:
        # pass 1 captures x_half with R disabled; pass 2 runs with R armed.
        #
        # Cost: 2x backbone forward per training step. Mitigation: use
        # gradient checkpointing only on the second pass; first pass is
        # under no_grad. Acceptable tradeoff for clean code.
        self._capture.clear()
        with torch.no_grad():
            self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if "x_half" not in self._capture:
            raise RuntimeError("capture hook did not fire — backbone shape unexpected")
        x_half = self._capture["x_half"]  # (B, T, dim) on backbone device
        x_half = x_half.to(next(self.encoder.parameters()).dtype)

        # 2) Run the encoder on x_half to get per-token persona logits.
        cos, sin = self._ensure_rope(t, x_half.device, x_half.dtype)
        encoder_logits = self.encoder(x_half, cos, sin, attn_mask=attention_mask)

        # 3) Pool / classify / sample → Z.
        if fixed_z_index is not None:
            if not (0 <= fixed_z_index < self.cfg.n_personas):
                raise ValueError(
                    f"fixed_z_index out of range [0, {self.cfg.n_personas - 1}]: {fixed_z_index}"
                )
            persona_logits = encoder_logits.float().mean(dim=1)  # (B, n_personas), reported only
            z_seq = torch.zeros(b, self.cfg.n_personas, device=x_half.device, dtype=x_half.dtype)
            z_seq[:, fixed_z_index] = 1.0
            z = z_seq.unsqueeze(1).expand(b, t, self.cfg.n_personas).contiguous()
        else:
            z, persona_logits = self.persona_head(encoder_logits, attn_mask=attention_mask)

        # 4) Project Z → R, run a second backbone forward with R armed.
        r = self.z_to_residual(z)
        with self._arm_residual(r):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = out.logits

        return FreetAdapterOutput(logits=logits, persona_logits=persona_logits, z=z)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run only the encoder side; return per-token persona logits.

        Used by the analysis script to extract per-sequence persona features
        without a second backbone forward.
        """
        self._capture.clear()
        self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        x_half = self._capture["x_half"].to(next(self.encoder.parameters()).dtype)
        _b, t, _d = x_half.shape
        cos, sin = self._ensure_rope(t, x_half.device, x_half.dtype)
        return self.encoder(x_half, cos, sin, attn_mask=attention_mask)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        fixed_z_index: int,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """Greedy / top-k autoregressive sampling with Z hard-clamped.

        Mirrors the from-scratch FreeTransformer.generate() shape.
        """
        device = input_ids.device
        b = input_ids.shape[0]
        ids = input_ids.clone()
        mask = attention_mask.clone()
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            out = self.forward(ids, mask, fixed_z_index=fixed_z_index)
            last_pos = mask.sum(dim=1) - 1
            row_idx = torch.arange(b, device=device)
            next_logits = out.logits[row_idx, last_pos].float()
            if temperature <= 0:
                next_tok = next_logits.argmax(dim=-1)
            else:
                next_logits = next_logits / temperature
                if top_k and top_k > 0:
                    topk_vals, _ = next_logits.topk(top_k, dim=-1)
                    cutoff = topk_vals[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if eos_token_id is not None:
                next_tok = torch.where(
                    finished, torch.full_like(next_tok, pad_token_id), next_tok
                )
                finished = finished | (next_tok == eos_token_id)
            ids = torch.cat([ids, next_tok.unsqueeze(-1)], dim=1)
            mask = torch.cat([mask, (~finished).long().unsqueeze(-1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return ids

    def close(self) -> None:
        """Detach the forward hooks. Call when discarding the adapter."""
        self._hook_handle.remove()
        self._capture_handle.remove()
