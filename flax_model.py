"""
flax_model.py — Flax re-implementation of SpeechLM's transformer block.

Flax vs PyTorch modules:
  PyTorch: class MyModel(nn.Module):
               def __init__(self): self.linear = nn.Linear(...)
               def forward(self, x): return self.linear(x)
           model = MyModel()
           out   = model(x)              # weights live inside model

  Flax:    class MyModel(nn.Module):
               @nn.compact
               def __call__(self, x): return nn.Dense(d)(x)
           model  = MyModel()
           params = model.init(key, x)["params"]   # params returned explicitly
           out    = model.apply({"params": params}, x)

  The key difference: params are an explicit pytree (nested dict),
  not hidden inside the object. This is what makes Flax composable
  with jit, vmap, and gradient transforms (optax).

@nn.compact decorator:
  Allows you to define submodules inline in __call__ rather than
  in __init__. The first call traces the module and allocates params.
  Subsequent calls with .apply() reuse the same structure.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

try:
    import flax.linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    print("Install Flax: pip install flax")


# ─────────────────────────────────────────────────────────────
# RoPE helper (stateless, no params needed)
# ─────────────────────────────────────────────────────────────

def apply_rope_flax(x, cos_freqs, sin_freqs):
    """RoPE rotation — identical math to PyTorch version."""
    T, H, D = x.shape
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    cos = cos_freqs[:T, jnp.newaxis, :]
    sin = sin_freqs[:T, jnp.newaxis, :]
    x_rot = jnp.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return x_rot.reshape(T, H, D)


# ─────────────────────────────────────────────────────────────
# Flax Modules
# ─────────────────────────────────────────────────────────────

class FlaxCausalSelfAttention(nn.Module):
    """
    Causal self-attention as a Flax Module.

    Flax module attributes are declared as class-level fields.
    They become part of the module's "static" configuration
    (compiled into the XLA graph, not trainable params).
    """
    d_model: int
    n_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, cos_freqs, sin_freqs, deterministic: bool = True):
        """
        Args:
            x           : (T, d_model)
            cos_freqs   : (max_T, d_head//2)
            sin_freqs   : (max_T, d_head//2)
            deterministic: if False, apply dropout (training mode)

        Returns: (T, d_model)

        @nn.compact means Dense layers are created on first call
        and reused on subsequent calls via .apply().
        """
        T, C  = x.shape
        d_head = C // self.n_heads

        # QKV projection — nn.Dense is Flax's equivalent of nn.Linear
        qkv = nn.Dense(3 * self.d_model, use_bias=True)(x)   # (T, 3*d_model)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape to (T, n_heads, d_head)
        def to_heads(t):
            return t.reshape(T, self.n_heads, d_head)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # RoPE
        q = apply_rope_flax(q, cos_freqs, sin_freqs)
        k = apply_rope_flax(k, cos_freqs, sin_freqs)

        # Attention scores
        scale  = jnp.sqrt(d_head).astype(x.dtype)
        q_t = jnp.transpose(q, (1, 0, 2))   # (H, T, d_head)
        k_t = jnp.transpose(k, (1, 0, 2))
        v_t = jnp.transpose(v, (1, 0, 2))

        scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 2, 1))) / scale

        # Causal mask
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(causal[jnp.newaxis], scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = nn.Dropout(rate=self.dropout_rate)(weights, deterministic=deterministic)

        out = jnp.matmul(weights, v_t)                   # (H, T, d_head)
        out = jnp.transpose(out, (1, 0, 2)).reshape(T, C)  # (T, d_model)

        return nn.Dense(self.d_model, use_bias=True)(out)


class FlaxSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block in Flax."""
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        gate = jax.nn.silu(nn.Dense(self.d_ff)(x))
        up   = nn.Dense(self.d_ff)(x)
        out  = nn.Dropout(rate=self.dropout_rate)(gate * up, deterministic=deterministic)
        return nn.Dense(self.d_model)(out)


class FlaxTransformerBlock(nn.Module):
    """Pre-norm transformer block in Flax."""
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, cos_freqs, sin_freqs, deterministic: bool = True):
        # Pre-norm + attention
        x = x + FlaxCausalSelfAttention(
            self.d_model, self.n_heads, self.dropout_rate
        )(nn.LayerNorm()(x), cos_freqs, sin_freqs, deterministic)

        # Pre-norm + FFN
        x = x + FlaxSwiGLUFFN(
            self.d_model, self.d_ff, self.dropout_rate
        )(nn.LayerNorm()(x), deterministic)

        return x


class FlaxSpeechLM(nn.Module):
    """
    Minimal Flax version of SpeechLM for benchmarking.
    Covers the transformer body (embedding → blocks → norm → logits).
    """
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, input_ids, cos_freqs, sin_freqs, deterministic: bool = True):
        """
        input_ids : (T,) integer array
        Returns   : (T, vocab_size) logits
        """
        x = nn.Embed(self.vocab_size, self.d_model)(input_ids)

        for _ in range(self.n_layers):
            x = FlaxTransformerBlock(
                self.d_model, self.n_heads, self.d_ff, self.dropout_rate
            )(x, cos_freqs, sin_freqs, deterministic)

        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size, use_bias=False)(x)


# ─────────────────────────────────────────────────────────────
# Parameter initialisation + forward pass helper
# ─────────────────────────────────────────────────────────────

def init_flax_model(key, vocab_size, d_model, n_heads, n_layers, d_ff,
                    max_seq_len, seq_len=64):
    """
    Initialise a FlaxSpeechLM and return (model, params, cos_freqs, sin_freqs).

    Flax init convention:
        params = model.init(key, dummy_input, ...)["params"]
    The "params" key separates learnable params from other state
    (e.g., batch norm running stats would be in "batch_stats").
    """
    if not FLAX_AVAILABLE:
        raise ImportError("pip install flax")

    from jax_attention import precompute_rope_freqs_jax
    d_head   = d_model // n_heads
    cos_f, sin_f = precompute_rope_freqs_jax(d_head, max_seq_len)

    model    = FlaxSpeechLM(vocab_size, d_model, n_heads, n_layers,
                            d_ff, max_seq_len)
    dummy_ids = jnp.zeros(seq_len, dtype=jnp.int32)
    params   = model.init(key, dummy_ids, cos_f, sin_f)["params"]

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"FlaxSpeechLM: {n_params/1e6:.1f}M parameters")

    return model, params, cos_f, sin_f


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not FLAX_AVAILABLE:
        print("Install Flax: pip install flax")
        exit()

    import jax

    vocab, d, h, layers, ff, T = 58_454, 256, 8, 4, 1024, 64
    key = jax.random.PRNGKey(0)

    model, params, cos_f, sin_f = init_flax_model(
        key, vocab, d, h, layers, ff, max_seq_len=512, seq_len=T
    )

    ids    = jax.random.randint(key, (T,), 0, vocab)
    logits = model.apply({"params": params}, ids, cos_f, sin_f)
    print(f"Logits shape: {logits.shape}")   # (T, vocab)
    print("Flax model smoke test passed.")