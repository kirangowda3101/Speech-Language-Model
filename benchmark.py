"""
benchmark.py — PyTorch vs JAX attention benchmarking.

Measures:
  1. Forward pass throughput (tokens/sec) across batch sizes
  2. Peak memory usage
  3. JIT warm-up cost vs steady-state latency
  4. Scaling behaviour with sequence length

Run:
    python benchmark.py --device cuda --seq_lens 64,128,256,512,1024
"""

import argparse
import time
import json
import gc
from typing import List

import torch
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available — will benchmark PyTorch only")


# ─────────────────────────────────────────────────────────────
# PyTorch benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_pytorch_attention(
    d_model: int,
    n_heads: int,
    seq_len: int,
    batch_size: int,
    n_warmup: int = 5,
    n_trials: int = 20,
    device: str = "cuda",
) -> dict:
    """Benchmark PyTorch CausalSelfAttention forward pass."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from config import small_config
    from model import CausalSelfAttention, precompute_rope_freqs

    cfg    = small_config()
    cfg.model.d_model  = d_model
    cfg.model.n_heads  = n_heads
    cfg.model.max_seq_len = seq_len * 2

    attn   = CausalSelfAttention(cfg.model).to(device)
    freqs  = precompute_rope_freqs(cfg.model.d_head, seq_len * 2).to(device)
    x      = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            out = attn(x, freqs)
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure
    times = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = attn(x, freqs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem = 0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

    tokens_per_sec = (batch_size * seq_len) / np.mean(times)

    return {
        "framework":     "pytorch",
        "d_model":       d_model,
        "n_heads":       n_heads,
        "seq_len":       seq_len,
        "batch_size":    batch_size,
        "mean_ms":       np.mean(times) * 1000,
        "std_ms":        np.std(times)  * 1000,
        "tokens_per_sec": int(tokens_per_sec),
        "peak_mem_mb":   peak_mem,
    }


# ─────────────────────────────────────────────────────────────
# JAX benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_jax_attention(
    d_model: int,
    n_heads: int,
    seq_len: int,
    batch_size: int,
    n_warmup: int = 5,
    n_trials: int = 20,
) -> dict:
    """Benchmark JAX batched+JIT attention forward pass."""
    if not JAX_AVAILABLE:
        return {"framework": "jax", "error": "JAX not available"}

    from jax_attention import (
        init_attention_params,
        precompute_rope_freqs_jax,
        make_batched_attention,
    )

    key      = jax.random.PRNGKey(0)
    params   = init_attention_params(key, d_model)
    d_head   = d_model // n_heads
    cos_f, sin_f = precompute_rope_freqs_jax(d_head, seq_len)
    x        = jax.random.normal(key, (batch_size, seq_len, d_model))

    batched_attn = make_batched_attention(n_heads)

    # JIT warm-up (first call compiles)
    t_compile_start = time.perf_counter()
    out = batched_attn(params, x, cos_f, sin_f)
    out.block_until_ready()
    compile_ms = (time.perf_counter() - t_compile_start) * 1000

    # Additional warmup
    for _ in range(n_warmup):
        out = batched_attn(params, x, cos_f, sin_f)
        out.block_until_ready()

    # Measure steady-state
    times = []
    for _ in range(n_trials):
        t0  = time.perf_counter()
        out = batched_attn(params, x, cos_f, sin_f)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    tokens_per_sec = (batch_size * seq_len) / np.mean(times)

    return {
        "framework":      "jax",
        "d_model":        d_model,
        "n_heads":        n_heads,
        "seq_len":        seq_len,
        "batch_size":     batch_size,
        "compile_ms":     compile_ms,
        "mean_ms":        np.mean(times) * 1000,
        "std_ms":         np.std(times)  * 1000,
        "tokens_per_sec": int(tokens_per_sec),
    }


# ─────────────────────────────────────────────────────────────
# Numerical correctness check
# ─────────────────────────────────────────────────────────────

def check_numerical_agreement(d_model: int = 128, n_heads: int = 4, T: int = 32):
    """
    Verify that JAX and PyTorch attention produce numerically close outputs
    when given identical weights and inputs.

    If max absolute difference > 1e-4, something is wrong in the port.
    """
    if not JAX_AVAILABLE:
        print("Skipping numerical check — JAX not available")
        return

    print("\nNumerical agreement check (JAX vs PyTorch)...")

    from jax_attention import (
        init_attention_params, precompute_rope_freqs_jax,
        multi_head_attention_jax
    )
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from config import small_config
    from model import CausalSelfAttention, precompute_rope_freqs

    # Shared random weights
    key    = jax.random.PRNGKey(42)
    params = init_attention_params(key, d_model)

    # Copy weights to PyTorch
    cfg           = small_config()
    cfg.model.d_model    = d_model
    cfg.model.n_heads    = n_heads
    cfg.model.max_seq_len = T * 2

    pt_attn = CausalSelfAttention(cfg.model)

    with torch.no_grad():
        qkv_w = np.array(params["qkv_w"])
        out_w = np.array(params["out_w"])
        pt_attn.qkv_proj.weight.copy_(torch.from_numpy(qkv_w.T))
        pt_attn.out_proj.weight.copy_(torch.from_numpy(out_w.T))
        pt_attn.qkv_proj.bias.zero_()
        pt_attn.out_proj.bias.zero_()

    # Shared input
    x_np  = np.random.randn(T, d_model).astype(np.float32)
    x_jax = jnp.array(x_np)
    x_pt  = torch.from_numpy(x_np).unsqueeze(0)  # (1, T, d_model)

    # JAX forward
    d_head = d_model // n_heads
    cos_f, sin_f = precompute_rope_freqs_jax(d_head, T)
    out_jax = multi_head_attention_jax(params, x_jax, cos_f, sin_f, n_heads)

    # PyTorch forward
    freqs_pt = precompute_rope_freqs(d_head, T)
    with torch.no_grad():
        out_pt = pt_attn(x_pt, freqs_pt)[0].numpy()  # (T, d_model)

    diff = np.abs(np.array(out_jax) - out_pt).max()
    print(f"  Max absolute difference: {diff:.6f}")
    print(f"  Status: {'PASS' if diff < 1e-3 else 'FAIL (check RoPE implementation)'}")


# ─────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--d_model",  type=int, default=768)
    parser.add_argument("--n_heads",  type=int, default=12)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--seq_lens", type=str, default="64,128,256,512")
    parser.add_argument("--output",   type=str, default="benchmark_results.json")
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    results  = []

    # Numerical check first
    check_numerical_agreement()

    print(f"\nBenchmarking attention: d_model={args.d_model}, "
          f"n_heads={args.n_heads}, batch={args.batch}")
    print(f"Sequence lengths: {seq_lens}")
    print("-" * 60)
    print(f"{'Framework':<12} {'seq_len':<10} {'mean_ms':<10} {'tokens/sec':<14} {'mem_mb':<10}")
    print("-" * 60)

    for seq_len in seq_lens:
        # PyTorch
        r_pt = benchmark_pytorch_attention(
            args.d_model, args.n_heads, seq_len, args.batch, device=args.device
        )
        results.append(r_pt)
        print(f"{'PyTorch':<12} {seq_len:<10} {r_pt['mean_ms']:<10.2f} "
              f"{r_pt['tokens_per_sec']:<14,} {r_pt.get('peak_mem_mb',0):<10.1f}")

        # JAX
        if JAX_AVAILABLE:
            r_jax = benchmark_jax_attention(
                args.d_model, args.n_heads, seq_len, args.batch
            )
            results.append(r_jax)
            speedup = r_pt["mean_ms"] / r_jax["mean_ms"]
            print(f"{'JAX':<12} {seq_len:<10} {r_jax['mean_ms']:<10.2f} "
                  f"{r_jax['tokens_per_sec']:<14,} {'—':<10}  "
                  f"({speedup:.2f}× vs PyTorch)")

        print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    print("Run plot_results.py to visualise.")


if __name__ == "__main__":
    main()