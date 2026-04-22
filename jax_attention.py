"""
jax_attention.py — Multi-head causal self-attention in pure JAX.

Why re-implement in JAX?
  This isn't just a porting exercise. JAX's jit/vmap/pmap transform
  system exposes a different mental model for thinking about:
    • What it means for a computation to be "compiled"
    • How batching is separate from the function definition (vmap)
    • How parallelism is explicit and composable (pmap)

  Benchmarking both implementations teaches you *where* the performance
  comes from — XLA compilation, memory layout, kernel fusion — rather
  than treating PyTorch as a black box.

Key JAX concepts used here:
  jax.jit       — compile a function to XLA (run once slow, then fast)
  jax.vmap      — auto-vectorise over a batch dimension
  jax.lax.scan  — efficient loop that doesn't unroll into the graph
  jnp           — JAX's NumPy-compatible array API

Install: pip install jax[cuda12] flax  (or jax[cpu] for testing)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. RoPE in JAX
# ─────────────────────────────────────────────────────────────

def precompute_rope_freqs_jax(d_head: int, max_seq_len: int, base: float = 10_000.0):
    """
    Precompute RoPE frequency matrix.

    Returns: (max_seq_len, d_head//2) complex64 array.

    JAX note: jnp operations are lazy until evaluated — this is
    fine at module load time since we call it once and cache.
    """
    inv_freq = 1.0 / (base ** (jnp.arange(0, d_head, 2).astype(jnp.float32) / d_head))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs     = jnp.outer(positions, inv_freq)          # (T, d_head//2)
    # JAX complex: combine cos/sin rather than using polar notation
    # We'll apply RoPE via the (cos, sin) rotation formulation
    return jnp.cos(freqs), jnp.sin(freqs)              # each (T, d_head//2)


def apply_rope_jax(x, cos_freqs, sin_freqs):
    """
    Apply RoPE to a query or key array.

    x          : (T, n_heads, d_head)  — note: JAX convention is often (T, H, D)
    cos_freqs  : (T, d_head//2)
    sin_freqs  : (T, d_head//2)

    Returns same shape as x.

    Rotation formula (equivalent to complex multiplication):
      x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
      x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    """
    T, H, D = x.shape
    # Split into even/odd components
    x0 = x[..., 0::2]   # (T, H, D//2)
    x1 = x[..., 1::2]   # (T, H, D//2)

    # Broadcast freqs: (T, 1, D//2)
    cos = cos_freqs[:T, jnp.newaxis, :]
    sin = sin_freqs[:T, jnp.newaxis, :]

    # Apply rotation
    x_rot = jnp.stack([
        x0 * cos - x1 * sin,
        x0 * sin + x1 * cos,
    ], axis=-1)  # (T, H, D//2, 2)

    return x_rot.reshape(T, H, D)


# ─────────────────────────────────────────────────────────────
# 2. Scaled dot-product attention
# ─────────────────────────────────────────────────────────────

def scaled_dot_product_attention_jax(q, k, v, mask=None):
    """
    Scaled dot-product attention — pure JAX, no libraries.

    Args:
        q, k, v: (T, H, D_head)
        mask   : optional (T, T) boolean mask — True = keep, False = mask out

    Returns: (T, H, D_head)

    This is numerically identical to F.scaled_dot_product_attention
    (same math, same scaling). The performance difference between
    JAX and PyTorch comes from XLA compilation + kernel fusion,
    not algorithm differences.
    """
    D_head = q.shape[-1]
    scale  = jnp.sqrt(D_head).astype(q.dtype)

    # q: (T, H, D) → (H, T, D) for matmul convenience
    q = jnp.transpose(q, (1, 0, 2))   # (H, T_q, D)
    k = jnp.transpose(k, (1, 0, 2))   # (H, T_k, D)
    v = jnp.transpose(v, (1, 0, 2))   # (H, T_v, D)

    # Attention scores: (H, T_q, T_k)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale

    # Causal mask: upper triangle → -inf
    T = q.shape[1]
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores      = jnp.where(causal_mask[jnp.newaxis], scores, -1e9)

    if mask is not None:
        scores = jnp.where(mask[jnp.newaxis], scores, -1e9)

    weights = jax.nn.softmax(scores, axis=-1)   # (H, T_q, T_k)

    # Weighted sum: (H, T_q, D)
    out = jnp.matmul(weights, v)

    # Back to (T, H, D)
    return jnp.transpose(out, (1, 0, 2))


# ─────────────────────────────────────────────────────────────
# 3. Full attention layer (stateless — params passed explicitly)
# ─────────────────────────────────────────────────────────────

def multi_head_attention_jax(
    params: dict,
    x: jnp.ndarray,          # (T, d_model)
    cos_freqs: jnp.ndarray,
    sin_freqs: jnp.ndarray,
    n_heads: int,
) -> jnp.ndarray:
    """
    Multi-head self-attention — pure function, no hidden state.

    params keys:
        qkv_w : (d_model, 3*d_model)
        qkv_b : (3*d_model,)
        out_w : (d_model, d_model)
        out_b : (d_model,)

    JAX insight: because this is a pure function (same inputs → same outputs,
    no side effects), jit() can safely compile and cache it. Any function
    that reads from global state or has side effects breaks jit.
    """
    T, C = x.shape
    d_head = C // n_heads

    # QKV projection
    qkv = x @ params["qkv_w"] + params["qkv_b"]   # (T, 3*C)
    q, k, v = jnp.split(qkv, 3, axis=-1)           # each (T, C)

    # Reshape to (T, n_heads, d_head)
    def to_heads(t):
        return t.reshape(T, n_heads, d_head)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)

    # Apply RoPE
    q = apply_rope_jax(q, cos_freqs, sin_freqs)
    k = apply_rope_jax(k, cos_freqs, sin_freqs)

    # Attention
    out = scaled_dot_product_attention_jax(q, k, v)   # (T, n_heads, d_head)

    # Merge heads
    out = out.reshape(T, C)                            # (T, C)

    # Output projection
    return out @ params["out_w"] + params["out_b"]


# ─────────────────────────────────────────────────────────────
# 4. JIT-compiled and vmap-batched versions
# ─────────────────────────────────────────────────────────────

def make_batched_attention(n_heads: int):
    """
    Returns a jit-compiled, vmap-batched attention function.

    vmap insight:
        multi_head_attention_jax operates on a single sequence (T, d_model).
        vmap(fn, in_axes=(None, 0, None, None, None)) says:
          "apply fn to each element of the batch axis (axis 0) of x,
           while keeping params, cos_freqs, sin_freqs the same for all"

        This is the functional alternative to writing a batch loop.
        vmap generates vectorised XLA code — no Python loop overhead.
    """
    def _attn_single(params, x, cos_freqs, sin_freqs):
        return multi_head_attention_jax(params, x, cos_freqs, sin_freqs, n_heads)

    # vmap over the batch dimension of x (in_axes=1 means axis 0 of x varies)
    batched = vmap(_attn_single, in_axes=(None, 0, None, None))

    # jit-compile the batched function
    return jit(batched)


# ─────────────────────────────────────────────────────────────
# 5. Parameter initialisation
# ─────────────────────────────────────────────────────────────

def init_attention_params(key: jax.random.PRNGKey, d_model: int) -> dict:
    """
    Randomly initialise attention parameters.

    JAX random note:
        JAX uses *explicit* random keys — no global random state.
        You must split the key and pass subkeys to each init call.
        This makes randomness reproducible and composable.

        key → split → key1 (for qkv_w), key2 (for out_w)

    Why explicit keys?
        jit() traces functions — if random state were implicit/global,
        jit would capture the state at trace time and produce the same
        random numbers on every call. Explicit keys avoid this pitfall.
    """
    k1, k2 = jax.random.split(key)
    std = 0.02
    return {
        "qkv_w": jax.random.normal(k1, (d_model, 3 * d_model)) * std,
        "qkv_b": jnp.zeros(3 * d_model),
        "out_w": jax.random.normal(k2, (d_model, d_model)) * std,
        "out_b": jnp.zeros(d_model),
    }


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("Install JAX: pip install jax[cuda12]  (or jax[cpu] for testing)")
        exit()

    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")

    d_model, n_heads, T, B = 256, 8, 64, 4
    d_head = d_model // n_heads

    key    = jax.random.PRNGKey(0)
    params = init_attention_params(key, d_model)

    cos_f, sin_f = precompute_rope_freqs_jax(d_head, T)

    # Single sequence
    x_single = jax.random.normal(key, (T, d_model))
    out      = multi_head_attention_jax(params, x_single, cos_f, sin_f, n_heads)
    print(f"\nSingle sequence: input {x_single.shape} → output {out.shape}")

    # Batched + JIT
    batched_attn = make_batched_attention(n_heads)
    x_batch = jax.random.normal(key, (B, T, d_model))

    # First call: triggers JIT compilation (slow)
    import time
    t0  = time.perf_counter()
    out_batch = batched_attn(params, x_batch, cos_f, sin_f)
    out_batch.block_until_ready()   # JAX is async — must block to measure
    t1  = time.perf_counter()
    print(f"Batch (first, JIT compile): {(t1-t0)*1000:.1f}ms  shape={out_batch.shape}")

    # Second call: uses compiled version (fast)
    t0  = time.perf_counter()
    out_batch = batched_attn(params, x_batch, cos_f, sin_f)
    out_batch.block_until_ready()
    t1  = time.perf_counter()
    print(f"Batch (second, compiled)  : {(t1-t0)*1000:.1f}ms")
    print("\nJAX attention smoke test passed.")