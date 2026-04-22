"""
audio_utils.py — Waveform preprocessing for the SpeechLM pipeline.

Why this file exists:
  Raw audio from LibriSpeech (or any dataset) needs to be:
    1. Loaded and converted to mono float32
    2. Resampled to 24kHz (EnCodec's required sample rate)
    3. Normalised (peak or RMS) to avoid clipping
    4. Chunked into fixed-length windows for batching
    5. Padded/trimmed to fit the model's max_seq_len

  These are all pre-EnCodec operations. Keeping them here keeps
  encodec_wrapper.py clean and focused on just the codec.
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, List, Tuple
import warnings

try:
    import torchaudio
    import torchaudio.functional as AF
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("torchaudio not installed. Run: pip install torchaudio")


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

TARGET_SR    = 24_000   # EnCodec 24kHz model
FRAME_RATE   = 75       # EnCodec frames/second at 24kHz
NUM_CODEBOOKS = 8       # at 6kbps bandwidth

# Max context in audio tokens: keep this ≤ model.max_seq_len
# At 75fps × 8 codebooks = 600 tokens/sec
# For max_seq_len=2048: 2048/600 ≈ 3.4 seconds of audio fits per chunk
MAX_AUDIO_TOKENS  = 1536   # 2 seconds of audio (75*8*2 = 1200, with room for text)
MAX_AUDIO_FRAMES  = MAX_AUDIO_TOKENS // NUM_CODEBOOKS   # 192 frames ≈ 2.56 seconds
MAX_AUDIO_SAMPLES = MAX_AUDIO_FRAMES * (TARGET_SR // FRAME_RATE)  # ≈ 61,440 samples


# ─────────────────────────────────────────────────────────────
# 1. Load and convert to mono float32
# ─────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return (waveform, sample_rate).

    waveform shape: (1, samples) — always mono, float32, range [-1, 1]

    Why mono?
        LibriSpeech is mono speech. EnCodec 24kHz operates on mono.
        Stereo would double our token count with no benefit for speech.

    Why float32?
        Consistent with PyTorch's default. EnCodec normalises internally.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("pip install torchaudio")

    waveform, sr = torchaudio.load(str(path))  # (channels, samples)

    # Convert to mono by averaging channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
        sr = target_sr

    # Ensure float32
    waveform = waveform.float()

    return waveform, sr


# ─────────────────────────────────────────────────────────────
# 2. Normalisation
# ─────────────────────────────────────────────────────────────

def peak_normalise(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """
    Normalise waveform so the absolute peak equals target_peak.

    Why not always normalise to 1.0?
        Leaving a small headroom (0.95) prevents clipping after
        any subsequent processing (e.g., mixing, augmentation).
    """
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform  # silence — don't divide by near-zero
    return waveform * (target_peak / peak)


def rms_normalise(waveform: torch.Tensor, target_db: float = -23.0) -> torch.Tensor:
    """
    Normalise waveform to a target RMS level in dBFS.
    -23 dBFS is the EBU R128 broadcast loudness standard.
    Good default for speech data.
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-8:
        return waveform
    target_rms = 10 ** (target_db / 20)
    return waveform * (target_rms / rms)


# ─────────────────────────────────────────────────────────────
# 3. Chunking
# ─────────────────────────────────────────────────────────────

def chunk_waveform(
    waveform: torch.Tensor,                  # (1, samples) or (samples,)
    chunk_samples: int = MAX_AUDIO_SAMPLES,
    overlap_samples: int = 0,
    drop_last: bool = False,
) -> List[torch.Tensor]:
    """
    Split a long waveform into fixed-length chunks.

    Args:
        waveform       : input audio, shape (1, samples) or (samples,)
        chunk_samples  : samples per chunk
        overlap_samples: overlap between consecutive chunks (0 = no overlap)
        drop_last      : if True, discard the final chunk if shorter than chunk_samples

    Returns:
        List of (1, chunk_samples) tensors (last may be shorter if drop_last=False)

    Why chunk?
        EnCodec can handle arbitrary lengths, but our Transformer has a
        fixed context window. We need to break long utterances into pieces
        that fit within max_seq_len tokens.

    Why overlap?
        Optional. Useful for evaluation (avoids edge artifacts at boundaries).
        During training, no overlap is fine — the model sees different
        random crops anyway via the DataLoader.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)   # ensure (1, samples)

    samples = waveform.shape[-1]
    stride  = chunk_samples - overlap_samples
    chunks  = []

    start = 0
    while start < samples:
        end   = start + chunk_samples
        chunk = waveform[:, start:end]
        if drop_last and chunk.shape[-1] < chunk_samples:
            break
        chunks.append(chunk)
        start += stride

    return chunks


def pad_or_trim(
    waveform: torch.Tensor,
    target_samples: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad (with silence) or trim a waveform to exactly target_samples.

    Used to make all chunks the same length within a batch.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    n = waveform.shape[-1]
    if n == target_samples:
        return waveform
    elif n > target_samples:
        return waveform[:, :target_samples]
    else:
        pad = torch.full((*waveform.shape[:-1], target_samples - n), pad_value)
        return torch.cat([waveform, pad], dim=-1)


# ─────────────────────────────────────────────────────────────
# 4. Audio tokens ↔ waveform frame alignment
# ─────────────────────────────────────────────────────────────

def samples_to_frames(samples: int, sr: int = TARGET_SR, frame_rate: int = FRAME_RATE) -> int:
    """How many EnCodec frames does `samples` audio samples produce?"""
    return int(samples * frame_rate / sr)


def frames_to_samples(frames: int, sr: int = TARGET_SR, frame_rate: int = FRAME_RATE) -> int:
    """How many audio samples correspond to `frames` EnCodec frames?"""
    return int(frames * sr / frame_rate)


def frames_to_tokens(frames: int, num_codebooks: int = NUM_CODEBOOKS) -> int:
    """Number of flat token IDs produced by `frames` EnCodec frames."""
    return frames * num_codebooks


def tokens_to_frames(tokens: int, num_codebooks: int = NUM_CODEBOOKS) -> int:
    return tokens // num_codebooks


def duration_to_tokens(seconds: float) -> int:
    """Convenience: how many tokens does `seconds` of audio become?"""
    return int(seconds * FRAME_RATE * NUM_CODEBOOKS)


# ─────────────────────────────────────────────────────────────
# 5. Batch collation helper
# ─────────────────────────────────────────────────────────────

def collate_waveforms(
    waveforms: List[torch.Tensor],
    target_samples: Optional[int] = None,
) -> torch.Tensor:
    """
    Collate a list of waveforms into a batch tensor.

    If target_samples is None, pads all to the length of the longest.
    Returns: (B, 1, samples)
    """
    if target_samples is None:
        target_samples = max(w.shape[-1] for w in waveforms)

    padded = [pad_or_trim(w, target_samples) for w in waveforms]
    # Each is (1, samples) → stack to (B, 1, samples)
    return torch.stack(padded, dim=0)


# ─────────────────────────────────────────────────────────────
# 6. LibriSpeech-specific file iterator
# ─────────────────────────────────────────────────────────────

def iter_librispeech_flac(
    root: str | Path,
    split: str = "train-clean-100",
    max_duration_seconds: float = 30.0,
) -> Iterator[Tuple[torch.Tensor, int, str, str]]:
    """
    Iterate over LibriSpeech .flac files.

    Yields: (waveform, sample_rate, speaker_id, transcript)

    LibriSpeech structure:
      root/
        train-clean-100/
          speaker_id/
            chapter_id/
              speaker-chapter-utterance.flac
              speaker-chapter.trans.txt    ← transcripts here

    Args:
        root            : path to LibriSpeech root
        split           : one of train-clean-100, train-clean-360,
                          train-other-500, dev-clean, test-clean, ...
        max_duration_seconds: skip files longer than this (avoids OOM on outliers)

    Usage in Phase 3 (DataLoader):
        for waveform, sr, speaker, text in iter_librispeech_flac(root, "train-clean-100"):
            codes = encodec_wrapper.encode(waveform, sample_rate=sr)
            ...
    """
    root = Path(root)
    split_dir = root / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"LibriSpeech split not found: {split_dir}\n"
            f"Download from: https://www.openslr.org/12"
        )

    # Load transcripts: each .trans.txt has lines "SPEAKER-CHAPTER-UTT text..."
    transcripts = {}
    for trans_file in split_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

    # Yield each .flac file
    max_samples = int(max_duration_seconds * TARGET_SR)
    skipped = 0

    for flac_path in sorted(split_dir.rglob("*.flac")):
        utt_id     = flac_path.stem
        speaker_id = utt_id.split("-")[0]
        transcript = transcripts.get(utt_id, "")

        try:
            waveform, sr = load_audio(flac_path, target_sr=TARGET_SR, mono=True)
        except Exception as e:
            warnings.warn(f"Failed to load {flac_path}: {e}")
            continue

        # Skip overly long files
        if waveform.shape[-1] > max_samples:
            skipped += 1
            continue

        yield waveform, TARGET_SR, speaker_id, transcript

    if skipped > 0:
        print(f"Skipped {skipped} files longer than {max_duration_seconds}s")


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== audio_utils smoke test ===\n")

    # 1. Token count sanity checks
    print("Token count estimates:")
    for secs in [1, 2, 5, 10, 30, 60]:
        toks = duration_to_tokens(secs)
        print(f"  {secs:>3}s audio  →  {toks:>6} tokens  ({toks/2048*100:.0f}% of max_seq_len=2048)")

    print()

    # 2. Chunking
    sr   = TARGET_SR
    wave = torch.randn(1, sr * 10)   # 10 seconds of fake audio
    chunks = chunk_waveform(wave, chunk_samples=MAX_AUDIO_SAMPLES)
    print(f"10s waveform ({wave.shape[1]} samples) chunked into {len(chunks)} pieces:")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {c.shape[1]} samples = {c.shape[1]/sr:.2f}s")

    print()

    # 3. Normalisation
    loud_wave = torch.randn(1, sr) * 3.0   # very loud
    norm_peak = peak_normalise(loud_wave)
    norm_rms  = rms_normalise(loud_wave)
    print(f"Original peak  : {loud_wave.abs().max():.3f}")
    print(f"Peak-norm peak : {norm_peak.abs().max():.3f}  (target 0.95)")
    rms = lambda w: w.pow(2).mean().sqrt()
    print(f"RMS-norm RMS   : {20*torch.log10(rms(norm_rms)):.1f} dBFS  (target -23.0)")

    print()

    # 4. Frame alignment
    samples = MAX_AUDIO_SAMPLES
    frames  = samples_to_frames(samples)
    tokens  = frames_to_tokens(frames)
    print(f"Chunk of {samples} samples → {frames} frames → {tokens} tokens")
    print(f"Round-trip: {frames_to_samples(frames)} samples")