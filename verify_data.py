"""
verify_data.py — Sanity check the data pipeline before launching training.

Run this after preprocess.py completes and before starting train.py.
It checks:
  1. Token ID ranges (no out-of-vocab IDs)
  2. Sequence length distribution
  3. DataLoader throughput (tokens/sec, batches/sec)
  4. GPU transfer speed (CPU → CUDA)
  5. Sample decoded sequences (visual inspection)

Usage:
    python verify_data.py --tokens_root /data/librispeech_tokens
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

from config import small_config
from tokenizer import SpeechLMTokenizer
from dataset import LibriSpeechTokenDataset
from dataloader import build_dataloader


def check_token_ranges(tokens_root: Path, cfg, tok: SpeechLMTokenizer, n_files: int = 500):
    """Verify all token IDs are within valid vocab range."""
    print("\n[1] Token ID range check")
    print(f"    Expected range: [0, {cfg.vocab.total_vocab_size})")
    print(f"    Sampling {n_files} files...")

    npy_files = list(tokens_root.rglob("*.npy"))[:n_files]
    if not npy_files:
        print("    No .npy files found — run preprocess.py first.")
        return

    all_min, all_max = float('inf'), float('-inf')
    type_counts = Counter()

    for f in npy_files:
        arr = np.load(f)
        all_min = min(all_min, int(arr.min()))
        all_max = max(all_max, int(arr.max()))
        for tid in arr[:100]:   # spot-check first 100 tokens
            if tok.is_text_token(int(tid)):
                type_counts["text"] += 1
            elif tok.is_audio_token(int(tid)):
                type_counts["audio"] += 1
            elif tok.is_special_token(int(tid)):
                type_counts["special"] += 1
            else:
                type_counts["INVALID"] += 1

    ok = (all_min >= 0 and all_max < cfg.vocab.total_vocab_size
          and type_counts.get("INVALID", 0) == 0)

    print(f"    Token ID min: {all_min}  max: {all_max}")
    print(f"    Token type distribution (sampled): {dict(type_counts)}")
    print(f"    Status: {'PASS' if ok else 'FAIL'}")
    return ok


def check_sequence_lengths(dataset: LibriSpeechTokenDataset, n_samples: int = 200):
    """Print sequence length statistics."""
    print(f"\n[2] Sequence length distribution (n={n_samples})")
    lengths = []
    for i in range(min(n_samples, len(dataset))):
        item    = dataset[i]
        lengths.append(item["length"])

    lengths = np.array(lengths)
    print(f"    mean={lengths.mean():.0f}  median={np.median(lengths):.0f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    print(f"    % fully filled (= seq_len): "
          f"{100*(lengths == dataset.seq_len).mean():.1f}%")
    print(f"    % under 50% filled        : "
          f"{100*(lengths < dataset.seq_len//2).mean():.1f}%")


def benchmark_throughput(loader, device: str = "cpu", n_batches: int = 20):
    """Measure DataLoader → GPU transfer throughput."""
    print(f"\n[3] DataLoader throughput benchmark ({n_batches} batches, device={device})")
    dev = torch.device(device)

    # Warmup
    for i, (inp, tgt, _) in enumerate(loader):
        if i >= 2:
            break

    total_tokens = 0
    start = time.perf_counter()

    for i, (inp, tgt, lengths) in enumerate(loader):
        if i >= n_batches:
            break
        inp = inp.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        total_tokens += inp.numel()

    elapsed = time.perf_counter() - start
    tokens_per_sec = total_tokens / elapsed
    batches_per_sec = n_batches / elapsed

    print(f"    {tokens_per_sec:,.0f} tokens/sec")
    print(f"    {batches_per_sec:.1f} batches/sec")
    print(f"    {total_tokens:,} tokens in {elapsed:.2f}s")

    # Rule of thumb: training needs ~5× this throughput to keep GPU busy
    # If tokens/sec < 500k, consider more DataLoader workers
    if tokens_per_sec < 500_000:
        print(f"    WARNING: throughput may bottleneck training. "
              f"Try increasing num_workers.")
    else:
        print(f"    Throughput looks healthy.")


def inspect_sample(dataset: LibriSpeechTokenDataset, tok: SpeechLMTokenizer, idx: int = 0):
    """Decode and print a sample sequence for visual inspection."""
    print(f"\n[4] Sample sequence (item {idx})")
    item      = dataset[idx]
    input_ids = item["input_ids"]
    length    = item["length"]

    print(f"    Total length: {len(input_ids)}, actual tokens: {length}")

    # Show first 30 token IDs and their types
    preview = input_ids[:30].tolist()
    types   = []
    for t in preview:
        if tok.is_special_token(t):
            types.append("S")
        elif tok.is_audio_token(t):
            types.append("A")
        else:
            types.append("T")

    print(f"    Token IDs (first 30): {preview}")
    print(f"    Types (T=text, A=audio, S=special): {''.join(types)}")

    # Decode text portion if present
    text_ids = [t for t in input_ids.tolist() if tok.is_text_token(t)]
    if text_ids:
        try:
            decoded = tok.decode_text(text_ids)
            print(f"    Text portion: {decoded!r}")
        except Exception:
            pass

    audio_ids_list = [t for t in input_ids.tolist() if tok.is_audio_token(t)]
    print(f"    Audio tokens: {len(audio_ids_list)} "
          f"(≈ {len(audio_ids_list) / (75*8):.2f}s of audio)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_root", type=str, required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["train-clean-100"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    cfg = small_config()
    tok = SpeechLMTokenizer(cfg.vocab)
    tokens_root = Path(args.tokens_root)

    # 1. Token range check
    check_token_ranges(tokens_root, cfg, tok)

    # 2. Build dataset
    print(f"\nLoading dataset from {tokens_root}...")
    dataset = LibriSpeechTokenDataset(
        tokens_root, args.splits, cfg, is_val=False
    )
    if len(dataset) == 0:
        print("Dataset is empty — check tokens_root and splits.")
        return

    # 3. Sequence length distribution
    check_sequence_lengths(dataset)

    # 4. Sample inspection
    inspect_sample(dataset, tok, idx=0)

    # 5. Throughput benchmark
    loader = build_dataloader(
        dataset, cfg.training,
        shuffle=False,
        num_workers=args.num_workers,
    )
    benchmark_throughput(loader, device=args.device)

    # 6. Estimated total tokens
    est_tokens = dataset.estimate_total_tokens()
    print(f"\n[5] Dataset size estimate")
    print(f"    ~{est_tokens/1e9:.2f}B tokens total")
    print(f"    ~{est_tokens / (cfg.training.batch_size * cfg.model.max_seq_len):,.0f} "
          f"batches per epoch (batch_size={cfg.training.batch_size})")

    print("\nAll checks complete. Ready for Phase 4 — DDP training.")


if __name__ == "__main__":
    main()