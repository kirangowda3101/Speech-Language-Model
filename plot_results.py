"""
plot_results.py — Plot benchmark_results.json from benchmark.py.

Produces two charts:
  1. Throughput (tokens/sec) vs sequence length — PyTorch vs JAX
  2. Latency (ms) vs sequence length — with error bars

Usage:
    python plot_results.py --input benchmark_results.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def plot_benchmark(results: list, output_prefix: str = "benchmark"):
    pt_results  = [r for r in results if r["framework"] == "pytorch"]
    jax_results = [r for r in results if r["framework"] == "jax" and "error" not in r]

    if not pt_results:
        print("No PyTorch results found.")
        return

    pt_seqs  = [r["seq_len"]       for r in pt_results]
    pt_tps   = [r["tokens_per_sec"] for r in pt_results]
    pt_ms    = [r["mean_ms"]        for r in pt_results]
    pt_std   = [r["std_ms"]         for r in pt_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PyTorch vs JAX Attention — Benchmark Results", fontsize=13)

    # ── Chart 1: Throughput ──────────────────────────────────
    ax = axes[0]
    ax.plot(pt_seqs, pt_tps, "o-", label="PyTorch", color="#534AB7", linewidth=2)

    if jax_results:
        jax_seqs = [r["seq_len"]        for r in jax_results]
        jax_tps  = [r["tokens_per_sec"] for r in jax_results]
        ax.plot(jax_seqs, jax_tps, "s--", label="JAX (jit+vmap)", color="#1D9E75", linewidth=2)

    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs sequence length")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(pt_seqs)
    ax.set_xticklabels(pt_seqs)

    # ── Chart 2: Latency with error bars ─────────────────────
    ax2 = axes[1]
    ax2.errorbar(pt_seqs, pt_ms, yerr=pt_std, fmt="o-",
                 label="PyTorch", color="#534AB7", linewidth=2, capsize=4)

    if jax_results:
        jax_ms  = [r["mean_ms"] for r in jax_results]
        jax_std = [r["std_ms"]  for r in jax_results]
        ax2.errorbar(jax_seqs, jax_ms, yerr=jax_std, fmt="s--",
                     label="JAX (jit+vmap)", color="#1D9E75", linewidth=2, capsize=4)

    ax2.set_xlabel("Sequence length (tokens)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency vs sequence length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(pt_seqs)
    ax2.set_xticklabels(pt_seqs)

    plt.tight_layout()
    out_path = f"{output_prefix}_charts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # ── Print speedup table ───────────────────────────────────
    if jax_results:
        print("\nSpeedup summary (PyTorch latency / JAX latency):")
        print(f"  {'seq_len':<10} {'PT (ms)':<12} {'JAX (ms)':<12} {'speedup':<10}")
        for pt, jx in zip(pt_results, jax_results):
            if pt["seq_len"] == jx["seq_len"]:
                speedup = pt["mean_ms"] / jx["mean_ms"]
                print(f"  {pt['seq_len']:<10} {pt['mean_ms']:<12.2f} "
                      f"{jx['mean_ms']:<12.2f} {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="benchmark_results.json")
    parser.add_argument("--output", type=str, default="benchmark")
    args = parser.parse_args()

    results = load_results(args.input)
    plot_benchmark(results, output_prefix=args.output)


if __name__ == "__main__":
    main()