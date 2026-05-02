"""Free Transformer architecture, faithful to Fleuret 2025 (arXiv 2510.17558).

Decoder Transformer (Llama-3-shape: SwiGLU, RMSNorm, RoPE, GQA) extended with:

  * a non-causal encoder block that emits per-token sigmoid logits L_t ∈ R^H,
  * a Binary Mapper that turns L_t into a hard one-hot Z_t ∈ {0,1}^{2^H} with
    Bernoulli-product gradient pass-through (paper Eqs. 6-8),
  * an additive injection of Z (linearly mapped to R of shape (T, D)) into the
    keys and values of decoder block L/2 + 1.

Sized for V100-feasible from-scratch training (25-60M params). The encoder
runs only at training and KV-cache pre-fill time; at generation time Z is
sampled uniformly from the prior and the encoder is skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FreeTransformerConfig:
    """Hyperparameters for the Free Transformer."""

    vocab_size: int = 256_000
    dim: int = 384
    n_layers: int = 6
    n_q_heads: int = 6
    n_kv_heads: int = 2
    mlp_ratio: int = 4
    max_seq_len: int = 1024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    latent_bits: int = 8
    """H in the paper. Z has 2**H categories per token. Used only in unsupervised mode."""
    inject_after_layer: int | None = None
    """Decoder layer index after which Z is injected. Defaults to n_layers // 2."""
    pad_token_id: int = 0
    tie_embeddings: bool = True
    latent_mode: str = "unsupervised"
    """One of:
        "unsupervised" — paper's free-bits KL VAE; Z has 2**latent_bits categories.
        "supervised"   — Z is the persona_id one-hot; encoder is trained as a
                         classifier; no KL term. Decoder injection is identical.
    """
    n_personas: int = 3
    """Used only in supervised mode. Z has this many categories."""

    def __post_init__(self) -> None:
        if self.dim % self.n_q_heads != 0:
            raise ValueError("dim must be divisible by n_q_heads")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be a multiple of n_kv_heads (GQA)")
        if self.inject_after_layer is None:
            self.inject_after_layer = self.n_layers // 2
        if not (0 <= self.inject_after_layer < self.n_layers - 1):
            raise ValueError(
                "inject_after_layer must be in [0, n_layers - 2] so the next "
                "block can consume the modulated K/V"
            )
        if self.latent_bits <= 0 or self.latent_bits > 16:
            raise ValueError("latent_bits must be in [1, 16]")
        if self.latent_mode not in ("unsupervised", "supervised"):
            raise ValueError(
                f"latent_mode must be 'unsupervised' or 'supervised', got {self.latent_mode!r}"
            )
        if self.latent_mode == "supervised" and self.n_personas < 2:
            raise ValueError("n_personas must be >= 2 for supervised mode")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_q_heads

    @property
    def latent_dim(self) -> int:
        if self.latent_mode == "supervised":
            return self.n_personas
        return 1 << self.latent_bits


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, learned scale)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tensors of shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D_head). cos/sin: (T, D_head).
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos_b) + (_rotate_half(x) * sin_b)


class SwiGLU(nn.Module):
    """SwiGLU MLP — `down(silu(gate) * up)` over (dim) → (mlp_dim) → (dim)."""

    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = mlp_ratio * dim
        # 2/3 trick (Llama): keep parameter count similar to a 4x MLP.
        hidden = int(2 * hidden / 3)
        # Round up to nearest multiple of 64 for kernel friendliness.
        hidden = ((hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class GroupedQueryAttention(nn.Module):
    """GQA with optional causal mask and split q vs kv inputs.

    The Free Transformer's L/2+1 block needs `in_q != in_kv` (Z is added only
    on the K/V branch). All other blocks call with `in_kv=None` to mean
    `in_kv = in_q`.
    """

    def __init__(self, cfg: FreeTransformerConfig, *, causal: bool) -> None:
        super().__init__()
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.repeat = cfg.n_q_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_q_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_q_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x_kv is None:
            x_kv = x_q
        b, t, _ = x_q.shape

        q = self.wq(x_q).view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_kv).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_kv).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            causal_mask = torch.ones(t, t, device=x_q.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            # attn_mask: (B, T) with 1 = keep, 0 = pad.
            # Block keys at padded positions across all heads/queries.
            kv_keep = attn_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(~kv_keep, float("-inf"))

        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_q_heads * self.head_dim)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (RMSNorm + GQA + RMSNorm + SwiGLU).

    Supports the Free Transformer's split-input call (`in_kv != in_q`), used
    only on the layer where Z is injected.
    """

    def __init__(self, cfg: FreeTransformerConfig, *, causal: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg, causal=causal)
        self.mlp_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_ratio)

    def forward(
        self,
        x_q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        x_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm residual: y = x_q + attn(norm(x_q), norm(x_kv or x_q)).
        h_q = self.attn_norm(x_q)
        h_kv = self.attn_norm(x_kv) if x_kv is not None else None
        x = x_q + self.attn(h_q, h_kv, cos, sin, attn_mask=attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- #
# Binary Mapper (paper §3.4)                                                  #
# --------------------------------------------------------------------------- #


class SupervisedPersonaHead(nn.Module):
    """Maps per-token persona logits to a per-sequence one-hot Z over personas.

    The encoder emits logits of shape ``(B, T, n_personas)``. We mean-pool over
    valid tokens to get a per-sequence ``(B, n_personas)`` logit vector,
    softmax it, sample (training) or argmax (eval) into a hard one-hot index,
    then broadcast the one-hot back to ``(B, T, n_personas)`` so downstream
    code matches the unsupervised injection path.

    Gradient pass-through uses the standard straight-through trick:
    ``one_hot + softmax - softmax.detach()``.
    """

    def __init__(self, n_personas: int) -> None:
        super().__init__()
        self.n_personas = n_personas

    def forward(
        self,
        logits: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_st, persona_logits).

        Args:
            logits: (B, T, n_personas).
            attn_mask: (B, T) 1/0 mask. Padded positions are dropped from the pool.

        Returns:
            z_st: (B, T, n_personas) straight-through one-hot, broadcast across T.
            persona_logits: (B, n_personas) per-sequence logits for the persona
                classification head loss.
        """
        b, t, c = logits.shape
        if c != self.n_personas:
            raise ValueError(f"expected {self.n_personas} categories, got {c}")
        logits32 = logits.float()
        logits32 = torch.nan_to_num(logits32, nan=0.0, posinf=20.0, neginf=-20.0)
        # Mean-pool over valid tokens to get a per-sequence logit.
        if attn_mask is None:
            persona_logits = logits32.mean(dim=1)  # (B, C)
        else:
            mask = attn_mask.float().unsqueeze(-1)  # (B, T, 1)
            denom = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
            persona_logits = (logits32 * mask).sum(dim=1) / denom  # (B, C)
        soft = torch.softmax(persona_logits, dim=-1)  # (B, C)
        # Hard sample at training, argmax at eval — straight-through gradient
        # in both cases through `soft`.
        if self.training:
            index = torch.distributions.Categorical(probs=soft).sample()  # (B,)
        else:
            index = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(index, num_classes=c).float()  # (B, C)
        z_seq = hard + soft - soft.detach()  # (B, C) straight-through
        # Broadcast across T so downstream injection code is identical to unsupervised.
        z_st = z_seq.unsqueeze(1).expand(b, t, c).contiguous()
        return z_st.to(logits.dtype), persona_logits


class BinaryMapper(nn.Module):
    """Maps H independent Bernoulli logits to a one-hot of dimension 2^H.

    Forward (training): Bernoulli sampling per bit → integer index → one-hot
    `Y`, plus the joint Bernoulli probability `G` over all 2^H values, returned
    as `Y + G - G.detach()` so the gradient flows through `G` (paper Eq. 8).

    Forward (eval): same, but the bits are sampled deterministically by
    `torch.bernoulli`. At true generation time the encoder is bypassed and the
    sampling happens uniformly (see `FreeTransformer.sample_z`).
    """

    def __init__(self, latent_bits: int) -> None:
        super().__init__()
        self.latent_bits = latent_bits
        self.latent_dim = 1 << latent_bits
        # Precomputed binary encodings of [0, 2^H-1].
        powers = 1 << torch.arange(latent_bits)
        codes = (torch.arange(self.latent_dim).unsqueeze(-1) // powers) % 2  # (2^H, H)
        self.register_buffer("codes", codes.float(), persistent=False)

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one-hot Z and return its KL contribution.

        Args:
            logits: (B, T, H) tensor of pre-sigmoid Bernoulli logits.

        Returns:
            z: (B, T, 2^H) tensor of straight-through one-hots.
            kl_per_token: (B, T) tensor of KL divergences against the uniform
                prior, computed in float32 for stability.
        """
        _b, _t, h = logits.shape
        if h != self.latent_bits:
            raise ValueError(f"expected {self.latent_bits} bits, got {h}")
        # Cast everything to fp32 inside the mapper. fp16 sigmoids saturate
        # near 0/1 and the joint-probability product underflows.
        logits32 = logits.float()
        # Guard against fp16 NaN/Inf bleeding through from the encoder block —
        # at fp16 with random init the encoder linear head can produce non-finite
        # values on the very first batch, and torch.bernoulli triggers a CUDA
        # assert on any p outside [0,1] (NaN included). We replace non-finites
        # with 0 (≡ uniform Bernoulli per bit) and rely on training to recover.
        logits32 = torch.nan_to_num(logits32, nan=0.0, posinf=20.0, neginf=-20.0)
        probs = torch.sigmoid(logits32)  # P(B_h = 1)
        # Numerical floor/ceiling: torch.sigmoid(x).clamp(0,1) is identity in
        # theory but defensive against fp32 round-off producing 1.0+eps when x
        # is huge — the bernoulli kernel checks strict bounds.
        probs = probs.clamp(min=0.0, max=1.0)
        # Sample bits.
        bits = torch.bernoulli(probs)  # (B, T, H)
        # Convert to integer index, then one-hot.
        powers = (1 << torch.arange(h, device=logits.device)).float()  # (H,)
        index = (bits * powers).sum(dim=-1).long()  # (B, T)
        y = torch.nn.functional.one_hot(index, num_classes=self.latent_dim).float()
        # Joint Bernoulli probability over all 2^H values, computed via the
        # log-prob trick to stay numerically sane.
        # codes: (2^H, H); log_p_pos = log(sigmoid); log_p_neg = log(1-sigmoid).
        log_p_pos = torch.nn.functional.logsigmoid(logits32)  # (B, T, H)
        log_p_neg = torch.nn.functional.logsigmoid(-logits32)  # (B, T, H)
        # Log G_{b,t,d} = sum_h codes[d, h] * log_p_pos + (1-codes[d, h]) * log_p_neg.
        # `self.codes` is registered as a buffer, so .to(model_dtype) follows the
        # module dtype (fp16 in V100 training); pin it back to fp32 here so the
        # einsum dtype matches the fp32 logits we just promoted.
        codes32 = self.codes.float()
        log_g = torch.einsum("dh,bth->btd", codes32, log_p_pos) + torch.einsum(
            "dh,bth->btd", 1.0 - codes32, log_p_neg
        )  # (B, T, 2^H)
        g = log_g.exp()
        z_st = y + g - g.detach()
        # KL against uniform prior P(Z) = 1/2^H. KL(Q||P) = H*log2 + sum_z Q log Q.
        # For independent bits we can compute it bitwise:
        #   KL = sum_h [ p log(2p) + (1-p) log(2(1-p)) ]
        # which is exact and avoids the (B, T, 2^H) sum.
        log2 = math.log(2.0)
        eps = 1e-12
        kl_bits = (
            probs * (torch.log(probs + eps) + log2)
            + (1.0 - probs) * (torch.log(1.0 - probs + eps) + log2)
        )  # (B, T, H)
        kl_per_token = kl_bits.sum(dim=-1)  # (B, T)
        # Cast Z back to model dtype for the downstream linear.
        return z_st.to(logits.dtype), kl_per_token


# --------------------------------------------------------------------------- #
# Free Transformer encoder                                                    #
# --------------------------------------------------------------------------- #


class FreeTransformerEncoder(nn.Module):
    """Non-causal encoder block + linear head producing per-token Bernoulli logits.

    Queries are a learned constant token embedding ζ replicated to length T;
    keys and values come from the first L/2 decoder blocks' output.
    """

    def __init__(self, cfg: FreeTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.zeta = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        self.block = TransformerBlock(cfg, causal=False)
        self.head_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        head_out = cfg.latent_bits if cfg.latent_mode == "unsupervised" else cfg.n_personas
        self.head = nn.Linear(cfg.dim, head_out, bias=True)
        # Step-0 stability: initialize the encoder head so the encoder's per-token
        # output starts near zero. In unsupervised mode this gives sigmoid ≈ 0.5
        # per bit (uniform Bernoulli) so torch.bernoulli is happy. In supervised
        # mode this gives softmax ≈ uniform over personas, matching the prior of
        # the persona_id classifier at step 0.
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.001)

    def forward(
        self,
        x_half: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return per-token Bernoulli logits of shape (B, T, H)."""
        b, t, _ = x_half.shape
        zeta = self.zeta.expand(b, t, -1).contiguous()
        h = self.block(zeta, cos, sin, x_kv=x_half, attn_mask=attn_mask)
        return self.head(self.head_norm(h))


# --------------------------------------------------------------------------- #
# Top-level Free Transformer                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class FreeTransformerOutput:
    logits: torch.Tensor
    kl_per_token: torch.Tensor | None
    encoder_logits: torch.Tensor | None
    z: torch.Tensor | None
    persona_logits: torch.Tensor | None = None
    """Per-sequence persona classifier logits. Set only when
    ``cfg.latent_mode == 'supervised'`` and the encoder ran.
    """


class FreeTransformer(nn.Module):
    """Decoder Transformer with VAE-style latent random tensor Z."""

    def __init__(self, cfg: FreeTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=cfg.pad_token_id)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, causal=True) for _ in range(cfg.n_layers)]
        )
        self.encoder = FreeTransformerEncoder(cfg)
        if cfg.latent_mode == "unsupervised":
            self.binary_mapper: nn.Module = BinaryMapper(cfg.latent_bits)
            self.persona_head: nn.Module | None = None
        else:
            self.binary_mapper = nn.Identity()  # unused; kept so old ckpts can load
            self.persona_head = SupervisedPersonaHead(cfg.n_personas)
        self.z_to_residual = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # RoPE cache buffers — registered so .to(device) covers them.
        cos, sin = _build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, device=torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_for(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} > max_seq_len {self.cfg.max_seq_len}")
        return self.rope_cos[:t], self.rope_sin[:t]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        sample_z_from_prior: bool = False,
        return_encoder_logits: bool = False,
    ) -> FreeTransformerOutput:
        """Run a forward pass.

        Args:
            tokens: (B, T) int64 input token ids.
            attn_mask: (B, T) optional 1/0 mask (1 = keep, 0 = padding). Padded
                positions are excluded from attention on both sides.
            sample_z_from_prior: when True, the encoder is skipped and Z is
                sampled uniformly at random — matches the paper's inference
                behaviour. Returns kl_per_token=None.
            return_encoder_logits: when True, also returns the encoder's
                per-token H-dim logits (handy for the analysis script).
        """
        b, t = tokens.shape
        cos, sin = self._rope_for(t)

        x = self.tok_emb(tokens)
        inject_idx = self.cfg.inject_after_layer
        # First half of the decoder.
        for i in range(inject_idx + 1):
            x = self.blocks[i](x, cos, sin, attn_mask=attn_mask)
        x_half = x

        encoder_logits: torch.Tensor | None = None
        kl_per_token: torch.Tensor | None = None
        persona_logits: torch.Tensor | None = None
        if sample_z_from_prior:
            z = self._sample_z_uniform(b, t, x.device, x.dtype)
        else:
            encoder_logits = self.encoder(x_half, cos, sin, attn_mask=attn_mask)
            if self.cfg.latent_mode == "unsupervised":
                z, kl_per_token = self.binary_mapper(encoder_logits)
            else:
                assert self.persona_head is not None
                z, persona_logits = self.persona_head(encoder_logits, attn_mask=attn_mask)
        # R = W_z · z; injected as additive on K/V of the very next block.
        r = self.z_to_residual(z)

        # Z is injected by re-running the *same* layer that produced x_half,
        # but with K/V coming from x_half + r. The clean reading of the paper
        # is "the L/2+1th block gets queries=X_{L/2}, k/v=X_{L/2}+R", i.e. the
        # block one past the injection point. We follow that literal reading:
        # blocks[0..inject_idx] run before injection; the *next* block
        # (blocks[inject_idx + 1]) sees the modulated K/V.
        next_idx = inject_idx + 1
        if next_idx >= self.cfg.n_layers:
            raise ValueError("inject_after_layer must leave at least one block after injection")
        x = self.blocks[next_idx](x, cos, sin, x_kv=x + r, attn_mask=attn_mask)

        # Remaining decoder blocks.
        for i in range(next_idx + 1, self.cfg.n_layers):
            x = self.blocks[i](x, cos, sin, attn_mask=attn_mask)

        logits = self.lm_head(self.final_norm(x))

        return FreeTransformerOutput(
            logits=logits,
            kl_per_token=kl_per_token,
            encoder_logits=encoder_logits if return_encoder_logits else None,
            z=z,
            persona_logits=persona_logits,
        )

    def _sample_z_uniform(
        self, batch: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample Z uniformly from the prior over `2^H` indices (paper inference path)."""
        index = torch.randint(0, self.cfg.latent_dim, (batch, seq_len), device=device)
        z = torch.nn.functional.one_hot(index, num_classes=self.cfg.latent_dim).to(dtype)
        return z

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the encoder's per-token H-dim logits (B, T, H).

        Used by the analysis script to extract per-sequence persona features
        without running the second half of the decoder.
        """
        _b, t = tokens.shape
        cos, sin = self._rope_for(t)
        x = self.tok_emb(tokens)
        for i in range(self.cfg.inject_after_layer + 1):
            x = self.blocks[i](x, cos, sin, attn_mask=attn_mask)
        return self.encoder(x, cos, sin, attn_mask=attn_mask)


def free_transformer_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    kl_per_token: torch.Tensor,
    *,
    pad_token_id: int,
    free_bits_kappa: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute cross-entropy + token-wise free-bits KL.

    Returns (total_loss, metrics_dict). Metrics are detached floats.
    """
    # Cross-entropy on next-token prediction; logits are over the full T.
    # We assume the caller already shifted targets so that targets[t] is the
    # next token after logits[t].
    vocab = logits.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    # Free-bits per token (paper Eq. 5): max(0, KL_t - kappa) averaged over T.
    excess = (kl_per_token - free_bits_kappa).clamp_min(0.0)
    kl_term = excess.mean()
    total = ce + kl_term
    metrics = {
        "loss": float(total.detach()),
        "ce": float(ce.detach()),
        "kl_mean": float(kl_per_token.mean().detach()),
        "kl_excess": float(kl_term.detach()),
    }
    return total, metrics


def supervised_freet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    persona_logits: torch.Tensor,
    persona_labels: torch.Tensor,
    *,
    pad_token_id: int,
    persona_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for supervised-Z mode: next-token CE + persona-classifier CE.

    Args:
        logits: (B, T, V) decoder LM logits.
        targets: (B, T) shifted next-token targets.
        persona_logits: (B, n_personas) per-sequence encoder classifier logits.
        persona_labels: (B,) integer persona id labels.
        pad_token_id: ignore index for the next-token CE.
        persona_loss_weight: scalar weight on the persona CE term.

    Returns:
        (total_loss, metrics).
    """
    vocab = logits.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    persona_ce = torch.nn.functional.cross_entropy(
        persona_logits, persona_labels, reduction="mean"
    )
    total = ce + persona_loss_weight * persona_ce
    with torch.no_grad():
        persona_acc = (persona_logits.argmax(dim=-1) == persona_labels).float().mean()
    metrics = {
        "loss": float(total.detach()),
        "ce": float(ce.detach()),
        "persona_ce": float(persona_ce.detach()),
        "persona_acc": float(persona_acc.detach()),
    }
    return total, metrics
