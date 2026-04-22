"""
model.py — SpeechLM: GPT extended for speech-language modelling.

Architecture overview
─────────────────────
                   ┌─────────────────────┐
  text tokens ──▶  │  TextEmbedding      │ ─┐
                   └─────────────────────┘  │
                                            ├──▶ d_model ──▶ TransformerBlocks ──▶ LM head
                   ┌─────────────────────┐  │
  audio tokens ──▶ │  AudioEmbedding     │ ─┘
                   └─────────────────────┘

Key differences from vanilla GPT
──────────────────────────────────
1. Dual embedding tables: text and audio tokens are initialised
   differently (audio tokens need smaller init — they represent
   quantised acoustic features, not semantic units).

2. Rotary Position Embeddings (RoPE) instead of learned absolute
   positions. RoPE generalises better to unseen sequence lengths
   and is standard in modern LLMs (LLaMA, Mistral, etc.).

3. Pre-norm (LayerNorm before attention/FFN) instead of post-norm.
   Pre-norm trains more stably at large scale.

4. SwiGLU activation in FFN instead of GELU.
   SwiGLU is used in PaLM, LLaMA-2, Mistral — empirically better.

5. Optional causal masking pattern: full causal (text) vs.
   delayed-pattern masking for multi-codebook audio (Phase 2 hook).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig, VocabConfig, SpeechLMConfig, small_config


# ─────────────────────────────────────────────────────────────
# 1. Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────

def precompute_rope_freqs(d_head: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """
    Precompute complex rotation frequencies for RoPE.

    RoPE rotates query/key vectors by an angle that depends on
    their absolute position. The rotation is applied in complex space:
      x_rotated = x * exp(i * theta * position)

    Returns shape: (max_seq_len, d_head//2) as complex tensor.

    Why RoPE over learned positions?
      • No position embedding table to learn (saves params)
      • Relative distance is preserved: <Rq, Rk> depends only on (pos_q - pos_k)
      • Generalises to longer sequences than seen during training
    """
    assert d_head % 2 == 0
    # theta_i = 1 / base^(2i / d_head),  i = 0..d_head/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head)) #creates 32 different rotation speeds
    positions = torch.arange(max_seq_len).float()
    # outer product → (max_seq_len, d_head//2)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64. Converts each angle into a complex number cos(angle) + i*sin(angle). This is the mathematical representation of a rotation.


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a query or key tensor.

    x     : (B, n_heads, T, d_head)
    freqs : (T, d_head//2)  complex
    """
    T = x.shape[2]
    # View last dim as pairs → complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_b = freqs[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,d_head//2)
    x_rotated = x_complex * freqs_b
    return torch.view_as_real(x_rotated).reshape(x.shape).type_as(x)


# ─────────────────────────────────────────────────────────────
# 2. Attention (with RoPE)
# ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Compared to your vanilla GPT:
      • Q/K rotated by position-dependent frequencies (RoPE)
      • Uses F.scaled_dot_product_attention (Flash Attention kernel
        when available — ~2-4× memory efficient, same math)
      • No explicit causal mask tensor; SDPA handles it via is_causal=True
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        # Q, K, V projections fused into one matrix for efficiency
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)#Linear transformation after 12 attention heads are merged back together
        self.attn_drop = cfg.dropout  # passed to SDPA

    def forward(
        self,
        x: torch.Tensor,               # (B, T, d_model)
        rope_freqs: torch.Tensor,       # (max_seq_len, d_head//2) complex
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_head)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply RoPE to Q and K (NOT V — V carries content, not position)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Flash Attention (or fallback) — handles causal mask internally
        # dropout only applied during training
        dropout_p = self.attn_drop if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=(attn_mask is None),  # is_causal=False if custom mask passed
        )  # (B, n_heads, T, d_head)

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────
# 3. Feed-Forward Network with SwiGLU
# ─────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    Standard FFN: x → Linear → GELU → Linear
    SwiGLU:       x → (Linear_gate ⊙ SiLU(Linear_up)) → Linear_down

    The gating mechanism lets the network suppress irrelevant features
    multiplicatively. Empirically outperforms GELU at the same param count.

    Note: to keep param count equal to a standard 4×d_model FFN,
    the hidden dim here is 2/3 of d_ff. We keep it as d_ff for simplicity.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # gate and up projections (run in parallel)
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) ⊙ up — the "gated linear unit" part
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(self.drop(gate * up))


# ─────────────────────────────────────────────────────────────
# 4. Transformer Block (Pre-Norm)
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.

    Pre-norm:  x → LayerNorm → Attention → x + residual
    Post-norm: x → Attention → x + residual → LayerNorm

    Pre-norm is more stable to train because the residual stream
    stays in a consistent scale throughout training.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln1(x), rope_freqs, attn_mask)
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 5. Dual Embedding Table
# ─────────────────────────────────────────────────────────────

class SpeechEmbedding(nn.Module):
    """
    Separate embedding tables for text and audio tokens,
    both projected to d_model.

    Why separate tables?
      • Text tokens represent semantic units — we can initialise
        from a pretrained GPT-2 embedding (Phase 1 bonus).
      • Audio tokens represent quantised acoustic features —
        they benefit from smaller initialisation (std=0.01 vs 0.02).
      • Keeps the embedding gradient flows independent.

    Both tables output d_model-dimensional vectors, so the
    Transformer sees a uniform representation regardless of modality.
    """

    def __init__(self, vocab: VocabConfig, model: ModelConfig):
        super().__init__()
        self.text_vocab_size  = vocab.text_vocab_size
        self.audio_offset     = vocab.audio_token_offset
        self.special_offset   = vocab.special_token_offset
        self.total_vocab      = vocab.total_vocab_size
        self.d_model          = model.d_model

        # Text embedding: normal init (same as GPT-2)
        self.text_emb = nn.Embedding(vocab.text_vocab_size, model.d_model)
        nn.init.normal_(self.text_emb.weight, mean=0.0, std=0.02)

        # Audio embedding: smaller init — codes are dense in acoustic space
        self.audio_emb = nn.Embedding(vocab.encodec.num_audio_tokens, model.d_model)
        nn.init.normal_(self.audio_emb.weight, mean=0.0, std=0.01)

        # Special tokens (pad, bos, eos, audio_start, audio_end)
        self.special_emb = nn.Embedding(vocab.num_special, model.d_model)
        nn.init.normal_(self.special_emb.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T) integer tensor — can contain a mix of
                   text, audio, and special token IDs.

        Returns: (B, T, d_model)

        Strategy: route each token to the correct embedding table
        using boolean masks, then sum the results.
        (Zero-ing out the other tables' contributions via the mask.)
        """
        B, T = token_ids.shape
        out = torch.zeros(B, T, self.d_model, device=token_ids.device,
                          dtype=self.text_emb.weight.dtype)

        # Text token mask
        text_mask  = token_ids < self.text_vocab_size
        if text_mask.any():
            ids = token_ids.clamp(0, self.text_vocab_size - 1)
            out += self.text_emb(ids) * text_mask.unsqueeze(-1)

        # Audio token mask
        audio_mask = (token_ids >= self.audio_offset) & (token_ids < self.special_offset)
        if audio_mask.any():
            ids = (token_ids - self.audio_offset).clamp(0, self.audio_emb.num_embeddings - 1)
            out += self.audio_emb(ids) * audio_mask.unsqueeze(-1)

        # Special token mask
        special_mask = token_ids >= self.special_offset
        if special_mask.any():
            ids = (token_ids - self.special_offset).clamp(0)
            out += self.special_emb(ids) * special_mask.unsqueeze(-1)

        return out


# ─────────────────────────────────────────────────────────────
# 6. SpeechLM — the full model
# ─────────────────────────────────────────────────────────────

class SpeechLM(nn.Module):
    """
    Speech-Language Model: a GPT that jointly models text and audio tokens.

    Forward pass (training):
      input_ids  → SpeechEmbedding → TransformerBlocks → LayerNorm → LM head
                                                                       ↓
                                                              logits (B, T, vocab)

    The LM head weight is tied to text_emb.weight (weight tying).
    This is standard in GPT-2 and saves ~38M params at d_model=768.
    Note: weight tying only applies to the text embedding, not audio,
    because the output distribution is over the full vocabulary.
    We use a fresh Linear for the full vocab projection.

    Usage:
      cfg   = small_config()
      model = SpeechLM(cfg)
      logits, loss = model(input_ids, targets=targets)
    """

    def __init__(self, cfg: SpeechLMConfig):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        vc = cfg.vocab

        # Embedding
        self.embedding = SpeechEmbedding(vc, mc)

        # Transformer stack
        self.blocks = nn.ModuleList([TransformerBlock(mc) for _ in range(mc.n_layers)])

        # Final layer norm (pre-norm architecture needs this after last block)
        self.ln_f = nn.LayerNorm(mc.d_model)

        # LM head: project d_model → full vocab
        self.lm_head = nn.Linear(mc.d_model, vc.total_vocab_size, bias=False)

        # Precompute RoPE frequencies (not a parameter, registered as buffer)
        rope = precompute_rope_freqs(mc.d_head, mc.max_seq_len)
        self.register_buffer("rope_freqs", rope, persistent=False)

        # Dropout on embeddings
        self.embed_drop = nn.Dropout(mc.dropout)

        # Weight tying: lm_head reuses text embedding weights for text token logits.
        # We do partial weight tying: only the text portion of lm_head.
        # Full weight tying (all vocab) would be:
        #   self.lm_head.weight = self.embedding.text_emb.weight  # only if same size

        # Count and report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SpeechLM initialised: {n_params/1e6:.1f}M parameters")

    def forward(
        self,
        input_ids: torch.Tensor,                 # (B, T)
        targets:   Optional[torch.Tensor] = None, # (B, T) for training
        attn_mask: Optional[torch.Tensor] = None, # custom mask (Phase 2)
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: (logits, loss)
          logits : (B, T, vocab_size) — always returned
          loss   : scalar cross-entropy — only if targets provided
        """
        B, T = input_ids.shape
        assert T <= self.cfg.model.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.model.max_seq_len}"

        # Embed tokens
        x = self.embedding(input_ids)   # (B, T, d_model)
        x = self.embed_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, self.rope_freqs, attn_mask)

        # Final norm
        x = self.ln_f(x)                # (B, T, d_model)

        # Project to vocabulary
        logits = self.lm_head(x)        # (B, T, vocab_size)

        # Compute loss if targets given (training / evaluation)
        loss = None
        if targets is not None:
            # Flatten batch and time dims for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab)
                targets.view(-1),                   # (B*T,)
                ignore_index=-1,                    # -1 = padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,    # (B, T_prompt)
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 200,
    ) -> torch.Tensor:
        """
        Autoregressive generation — same as GPT-2, works for both
        text and audio tokens since they share the vocabulary.

        Returns: (B, T_prompt + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            ctx = input_ids[:, -self.cfg.model.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature  # (B, vocab)

            if top_k is not None:
                # Zero out logits below the top-k threshold
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))

    def configure_optimizer(self, weight_decay: float, lr: float, device: str):
        """
        AdamW with weight decay applied only to 2D params (weights),
        NOT to biases or LayerNorm parameters.
        This is the standard GPT-style optimiser setup.
        """
        decay_params   = [p for n, p in self.named_parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() <  2 and p.requires_grad]

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Use fused AdamW on CUDA (meaningfully faster)
        use_fused = (device == "cuda") and ("fused" in torch.optim.AdamW.__init__.__doc__ or True)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95),
                                       eps=1e-8, fused=use_fused and torch.cuda.is_available())
        print(f"Optimiser: {len(decay_params)} decay tensors, {len(nodecay_params)} no-decay tensors")
        return optimizer


# ─────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    cfg   = small_config()
    model = SpeechLM(cfg)

    # Simulate a mixed text+audio batch
    B, T  = 2, 64
    # Mix of text IDs (0-50256), audio IDs (50257+), and special tokens
    ids   = torch.randint(0, cfg.vocab.total_vocab_size, (B, T))
    tgts  = torch.roll(ids, -1, dims=1)  # next-token targets

    logits, loss = model(ids, targets=tgts)
    print(f"logits shape : {logits.shape}")   # (2, 64, total_vocab_size)
    print(f"loss         : {loss.item():.4f}")

    # Test generation
    prompt   = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"generated shape: {generated.shape}")  # (1, 30)