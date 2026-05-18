"""
preprocess.py — Offline audio tokenisation for LibriSpeech.

Run this ONCE on the HPC before launching training.
It walks all .flac files, encodes them with EnCodec,
and saves flat token arrays as .npy files.

On a single GPU (A100), encoding 960 hours takes ~2-4 hours.
Use --workers to parallelise across CPU cores for the I/O-bound parts.

Usage:
    # Encode train-clean-100 only (fastest, ~100hrs)
    python preprocess.py \\
        --librispeech_root /data/LibriSpeech \\
        --output_root      /data/librispeech_tokens \\
        --splits           train-clean-100

    # Encode all 960hrs (train-clean-100 + train-clean-360 + train-other-500)
    python preprocess.py \\
        --librispeech_root /data/LibriSpeech \\
        --output_root      /data/librispeech_tokens \\
        --splits           train-clean-100 train-clean-360 train-other-500 \\
        --device           cuda \\
        --workers          8

Output layout:
    /data/librispeech_tokens/
        train-clean-100/
            1234-56789-0001.npy   ← int32 array, shape (K*T,)
            1234-56789-0001.txt   ← transcript text
            ...
        manifest.json             ← metadata (duration, token counts, etc.)
"""

import argparse
import io
import json
import re
import time
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import small_config
from tokenizer import SpeechLMTokenizer
from audio_utils import (
    load_audio, peak_normalise, TARGET_SR,
    iter_librispeech_flac
)

try:
    from encodec_wrapper import EnCodecWrapper
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False


def encode_one_file(
    flac_path: Path,
    out_dir: Path,
    wrapper: "EnCodecWrapper",
    tok: SpeechLMTokenizer,
    transcript: str,
) -> dict | None:
    """
    Encode one .flac file → save .npy + .txt.

    Returns a metadata dict on success, None on failure.
    """
    npy_path = out_dir / (flac_path.stem + ".npy")
    txt_path = out_dir / (flac_path.stem + ".txt")

    # Skip if already processed (allows resuming interrupted jobs)
    if npy_path.exists():
        existing = np.load(npy_path)
        return {"file": flac_path.stem, "tokens": len(existing), "skipped": True}

    try:
        # Load and normalise
        waveform, sr = load_audio(flac_path, target_sr=TARGET_SR)
        waveform     = peak_normalise(waveform)
        duration_sec = waveform.shape[-1] / sr

        # Encode with EnCodec
        codes     = wrapper.encode(waveform, sample_rate=sr)   # (K, T)
        audio_ids = tok.encode_audio_codes(codes.numpy())       # flat list

        # Save token array
        np.save(npy_path, np.array(audio_ids, dtype=np.int32))

        # Save transcript
        if transcript:
            txt_path.write_text(transcript)

        return {
            "file":     flac_path.stem,
            "tokens":   len(audio_ids),
            "duration": duration_sec,
            "frames":   codes.shape[1],
            "skipped":  False,
        }

    except Exception as e:
        print(f"  ERROR {flac_path.name}: {e}")
        return None


def preprocess_split(
    split: str,
    librispeech_root: Path,
    output_root: Path,
    wrapper: "EnCodecWrapper",
    tok: SpeechLMTokenizer,
) -> dict:
    """
    Process all files in one LibriSpeech split.
    Returns summary statistics.
    """
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files and transcripts
    print(f"\nScanning {split}...")
    items = []
    for waveform, sr, speaker_id, transcript in iter_librispeech_flac(
        librispeech_root, split
    ):
        # We need the path — re-scan to get it
        pass  # we'll scan differently below

    # Direct scan for paths + transcripts
    split_dir = librispeech_root / split
    transcripts = {}
    for trans_file in split_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

    flac_files = sorted(split_dir.rglob("*.flac"))
    print(f"  Found {len(flac_files):,} .flac files")

    # Process sequentially (EnCodec is GPU-bound; parallelising won't help much)
    stats = {"total": 0, "skipped": 0, "errors": 0,
             "total_tokens": 0, "total_duration": 0.0}

    for flac_path in tqdm(flac_files, desc=split, unit="file"):
        utt_id     = flac_path.stem
        transcript = transcripts.get(utt_id, "")
        result     = encode_one_file(flac_path, out_dir, wrapper, tok, transcript)

        if result is None:
            stats["errors"] += 1
        else:
            stats["total"]         += 1
            stats["total_tokens"]  += result["tokens"]
            stats["total_duration"] += result.get("duration", 0)
            if result["skipped"]:
                stats["skipped"] += 1

    print(f"  Done: {stats['total']:,} files, "
          f"{stats['total_tokens']:,} tokens, "
          f"{stats['total_duration']/3600:.1f}h audio, "
          f"{stats['errors']} errors, {stats['skipped']} skipped")
    return stats


_GIGASPEECH_TAGS = re.compile(r"<[^>]+>")


def _clean_gigaspeech_text(text: str) -> str:
    """Remove GigaSpeech punctuation tags like <COMMA>, <PERIOD>, etc."""
    return _GIGASPEECH_TAGS.sub("", text).strip()


def preprocess_gigaspeech(
    output_root: Path,
    wrapper: "EnCodecWrapper",
    tok: SpeechLMTokenizer,
    subset: str = "m",
    hf_split: str = "train",
) -> dict:
    """
    Download and preprocess GigaSpeech via HuggingFace datasets.

    Produces the same .npy / .txt output layout as preprocess_split(),
    written to output_root/gigaspeech-{subset}/.

    Args:
        output_root : root directory for token files
        wrapper     : EnCodecWrapper instance
        tok         : SpeechLMTokenizer instance
        subset      : GigaSpeech size bucket — xs/s/m/l/xl (default: m ≈ 1000h)
        hf_split    : HuggingFace split to use (default: train)
    """
    try:
        import datasets
    except ImportError:
        raise ImportError("pip install datasets")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("pip install soundfile")

    try:
        import torchaudio.functional as AF
    except ImportError:
        raise ImportError("pip install torchaudio")

    split_name = f"gigaspeech-{subset}"
    out_dir = output_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # streaming=True skips automatic audio decoding (which would invoke torchcodec).
    # We decode each example manually via soundfile below.
    print(f"\nLoading GigaSpeech '{subset}' ({hf_split}) from HuggingFace (streaming)...")
    ds = datasets.load_dataset(
        "speechcolab/gigaspeech", subset,
        split=hf_split,
        streaming=True,
    )
    ds = ds.cast_column("audio", datasets.features.Audio(decode=False))

    stats = {"total": 0, "skipped": 0, "errors": 0,
             "total_tokens": 0, "total_duration": 0.0}

    for example in tqdm(ds, desc=split_name, unit="file"):
        seg_id   = example["segment_id"]
        npy_path = out_dir / f"{seg_id}.npy"
        txt_path = out_dir / f"{seg_id}.txt"

        if npy_path.exists():
            existing = np.load(npy_path)
            stats["total"]        += 1
            stats["total_tokens"] += len(existing)
            stats["skipped"]      += 1
            continue

        try:
            arr, sr  = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32", always_2d=False)
            waveform = torch.from_numpy(arr).float().unsqueeze(0)  # (1, samples)

            if sr != TARGET_SR:
                waveform = AF.resample(waveform, sr, TARGET_SR)

            waveform     = peak_normalise(waveform)
            duration_sec = waveform.shape[-1] / TARGET_SR

            codes     = wrapper.encode(waveform, sample_rate=TARGET_SR)
            audio_ids = tok.encode_audio_codes(codes.cpu().numpy())

            np.save(npy_path, np.array(audio_ids, dtype=np.int32))

            transcript = _clean_gigaspeech_text(example["text"])
            if transcript:
                txt_path.write_text(transcript)

            stats["total"]          += 1
            stats["total_tokens"]   += len(audio_ids)
            stats["total_duration"] += duration_sec

        except Exception as e:
            print(f"  ERROR {seg_id}: {e}")
            stats["errors"] += 1

    print(f"  Done: {stats['total']:,} files, "
          f"{stats['total_tokens']:,} tokens, "
          f"{stats['total_duration']/3600:.1f}h audio, "
          f"{stats['errors']} errors, {stats['skipped']} skipped")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data to token arrays")
    parser.add_argument("--dataset", type=str, default="librispeech",
                        choices=["librispeech", "gigaspeech"],
                        help="Dataset to preprocess")
    parser.add_argument("--librispeech_root", type=str,
                        help="Path to LibriSpeech root (required for --dataset librispeech)")
    parser.add_argument("--output_root",      type=str, required=True,
                        help="Where to write .npy token files")
    parser.add_argument("--splits", nargs="+",
                        default=["train-clean-100"],
                        help="LibriSpeech splits to process (ignored for gigaspeech)")
    parser.add_argument("--gigaspeech_subset", type=str, default="m",
                        choices=["xs", "s", "m", "l", "xl"],
                        help="GigaSpeech size bucket (default: m ≈ 1000h)")
    parser.add_argument("--device",    type=str, default="cuda",
                        help="cuda / cpu")
    parser.add_argument("--bandwidth", type=float, default=6.0,
                        help="EnCodec bandwidth in kbps (controls num codebooks)")
    args = parser.parse_args()

    if not ENCODEC_AVAILABLE:
        print("ERROR: encodec not installed. Run: pip install encodec")
        return

    if args.dataset == "librispeech" and not args.librispeech_root:
        parser.error("--librispeech_root is required when --dataset=librispeech")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg     = small_config()
    tok     = SpeechLMTokenizer(cfg.vocab)
    wrapper = EnCodecWrapper(bandwidth=args.bandwidth, device=args.device)

    print(f"\nPreprocessing {args.dataset} → {output_root}")
    print(f"Device: {args.device}, bandwidth: {args.bandwidth}kbps")
    print(f"Tokens per second of audio: {wrapper.tokens_per_second()}")

    all_stats  = {}
    start_time = time.time()

    if args.dataset == "librispeech":
        librispeech_root = Path(args.librispeech_root)
        print(f"Splits: {args.splits}")
        for split in args.splits:
            all_stats[split] = preprocess_split(
                split, librispeech_root, output_root, wrapper, tok
            )
    else:
        all_stats["gigaspeech"] = preprocess_gigaspeech(
            output_root, wrapper, tok, subset=args.gigaspeech_subset
        )

    elapsed = time.time() - start_time

    manifest = {
        "dataset":         args.dataset,
        "bandwidth":       args.bandwidth,
        "tokens_per_sec":  wrapper.tokens_per_second(),
        "num_codebooks":   wrapper.num_codebooks,
        "elapsed_seconds": elapsed,
        "stats":           all_stats,
    }
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Total elapsed: {elapsed/3600:.1f}h")
    print("\nNext step: launch training with train.py")


if __name__ == "__main__":
    main()