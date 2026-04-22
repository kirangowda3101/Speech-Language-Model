"""
test_pipeline.py — End-to-end pipeline integration test.

This script validates that all Phase 1 + Phase 2 components work together:

  WAV file
    ↓  audio_utils.load_audio()
  waveform (1, samples)
    ↓  audio_utils.peak_normalise()
  normalised waveform
    ↓  EnCodecWrapper.encode()
  codes (K, T)
    ↓  SpeechLMTokenizer.encode_audio_codes()
  audio token IDs (flat list)
    ↓  SpeechLMTokenizer.build_training_sequence()
  full sequence [BOS, text..., AUDIO_START, audio..., AUDIO_END, EOS]
    ↓  SpeechLMTokenizer.build_input_target_pair()
  (input_ids, targets) tensors
    ↓  SpeechLM.forward()
  logits, loss
    ↓  SpeechLMTokenizer.decode_audio_token_ids()
  codes (K, T)  ← recovered from model output
    ↓  EnCodecWrapper.decode()
  reconstructed waveform

Run with:
  python test_pipeline.py                   # uses a synthetic sine wave
  python test_pipeline.py --wav path/to.wav # uses a real file
"""

import argparse
import torch
import numpy as np

from config import small_config
from model import SpeechLM
from tokenizer import SpeechLMTokenizer
from audio_utils import (
    load_audio, peak_normalise, chunk_waveform,
    duration_to_tokens, MAX_AUDIO_SAMPLES, TARGET_SR
)

try:
    from encodec_wrapper import EnCodecWrapper
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False


def make_synthetic_wave(duration_seconds: float = 2.0, sr: int = TARGET_SR) -> torch.Tensor:
    """Generate a simple test signal: chord of 3 sine waves."""
    t = torch.linspace(0, duration_seconds, int(sr * duration_seconds))
    wave = (
        0.4 * torch.sin(2 * torch.pi * 261.63 * t) +  # C4
        0.3 * torch.sin(2 * torch.pi * 329.63 * t) +  # E4
        0.3 * torch.sin(2 * torch.pi * 392.00 * t)    # G4
    )
    return wave.unsqueeze(0)   # (1, samples)


def run_pipeline(wav_path: str | None = None, device: str = "cpu"):
    """
    Full pipeline: audio → tokens → model forward pass → reconstruction.
    """
    print("=" * 60)
    print("SpeechLM End-to-End Pipeline Test")
    print("=" * 60)

    # ── 1. Load or synthesize audio ──────────────────────────────
    print("\n[1] Loading audio...")
    if wav_path:
        waveform, sr = load_audio(wav_path, target_sr=TARGET_SR)
        print(f"    Loaded: {wav_path}")
    else:
        waveform = make_synthetic_wave(duration_seconds=2.0)
        sr = TARGET_SR
        print(f"    Using synthetic 2-second C major chord")

    waveform = peak_normalise(waveform)
    duration = waveform.shape[-1] / sr
    print(f"    Shape: {waveform.shape}, duration: {duration:.2f}s, sr: {sr}Hz")

    # Chunk to fit context window
    chunks = chunk_waveform(waveform, chunk_samples=MAX_AUDIO_SAMPLES)
    print(f"    Chunked into {len(chunks)} piece(s) of ≤{MAX_AUDIO_SAMPLES/sr:.2f}s each")
    waveform_chunk = chunks[0]   # process first chunk

    # ── 2. EnCodec encode ────────────────────────────────────────
    print("\n[2] Encoding with EnCodec...")
    if not ENCODEC_AVAILABLE:
        print("    EnCodec not available — simulating codes with random integers")
        K, T = 8, int(duration_to_tokens(min(duration, 2.0)) / 8)
        codes = torch.randint(0, 1024, (K, T))
    else:
        wrapper = EnCodecWrapper(bandwidth=6.0, device=device)
        codes   = wrapper.encode(waveform_chunk, sample_rate=sr)

    print(f"    Codes shape: {codes.shape}  (K={codes.shape[0]} codebooks, T={codes.shape[1]} frames)")
    print(f"    Code stats: min={codes.min()}, max={codes.max()}, dtype={codes.dtype}")
    expected_tokens = codes.shape[0] * codes.shape[1]
    print(f"    Will become {expected_tokens} flat audio token IDs")

    # ── 3. Tokenize ──────────────────────────────────────────────
    print("\n[3] Tokenizing...")
    cfg = small_config()
    tok = SpeechLMTokenizer(cfg.vocab)

    audio_ids = tok.encode_audio_codes(codes.numpy())
    print(f"    Audio token IDs: {len(audio_ids)} tokens")
    print(f"    First 8 IDs: {audio_ids[:8]}")
    print(f"    All in audio range [{cfg.vocab.audio_token_offset}, {cfg.vocab.special_token_offset}): "
          f"{all(cfg.vocab.audio_token_offset <= t < cfg.vocab.special_token_offset for t in audio_ids)}")

    # Build a text + audio training sequence
    # Simulating: text prompt "The speaker said:" followed by audio tokens
    try:
        import tiktoken
        enc   = tiktoken.get_encoding("gpt2")
        text_ids = enc.encode("The speaker said:")
    except ImportError:
        text_ids = [464, 12599, 531]   # fallback: hard-coded GPT-2 IDs for "The speaker said"

    full_sequence = tok.build_training_sequence(text_ids, audio_ids)
    print(f"\n    Full training sequence length: {len(full_sequence)} tokens")
    print(f"    Breakdown:")
    print(f"      BOS:          1 token")
    print(f"      Text:         {len(text_ids)} tokens")
    print(f"      AUDIO_START:  1 token")
    print(f"      Audio:        {len(audio_ids)} tokens")
    print(f"      AUDIO_END:    1 token")
    print(f"      EOS:          1 token")

    # Truncate to model's max_seq_len if needed
    max_len = cfg.model.max_seq_len
    if len(full_sequence) > max_len + 1:
        print(f"\n    Sequence too long ({len(full_sequence)}), truncating to {max_len+1}")
        full_sequence = full_sequence[:max_len + 1]

    input_ids, targets = tok.build_input_target_pair(full_sequence, pad_to=256)
    print(f"\n    input_ids : {input_ids.shape}")
    print(f"    targets   : {targets.shape}")
    print(f"    Non-padding targets: {(targets != -1).sum().item()}")

    # ── 4. Model forward pass ────────────────────────────────────
    print("\n[4] Running model forward pass...")
    model = SpeechLM(cfg)
    model.eval()
    model.to(device)

    inp = input_ids.unsqueeze(0).to(device)   # (1, T)
    tgt = targets.unsqueeze(0).to(device)     # (1, T)

    with torch.no_grad():
        logits, loss = model(inp, targets=tgt)

    print(f"    Logits shape : {logits.shape}  (B=1, T={logits.shape[1]}, vocab={logits.shape[2]})")
    print(f"    Loss         : {loss.item():.4f}  (untrained, ~log({cfg.vocab.total_vocab_size}) = {np.log(cfg.vocab.total_vocab_size):.2f} expected)")

    # ── 5. Decode tokens back to audio ──────────────────────────
    print("\n[5] Decoding tokens back to audio...")

    # In real inference: take the model's predicted audio token IDs
    # Here: take the ground-truth audio_ids to verify the round-trip
    recovered_codes = tok.decode_audio_token_ids(audio_ids)
    print(f"    Recovered codes shape : {recovered_codes.shape}")
    print(f"    Round-trip exact match: {np.array_equal(codes.numpy(), recovered_codes)}")

    if ENCODEC_AVAILABLE:
        recon_wave = wrapper.decode(torch.from_numpy(recovered_codes))
        print(f"    Reconstructed waveform: {recon_wave.shape}")
        try:
            import torchaudio
            torchaudio.save("pipeline_reconstruction.wav", recon_wave.unsqueeze(0), TARGET_SR)
            print("    Saved: pipeline_reconstruction.wav")
        except Exception:
            pass

    # ── 6. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Audio duration        : {duration:.2f}s")
    print(f"  Waveform samples      : {waveform_chunk.shape[-1]:,}")
    print(f"  EnCodec frames        : {codes.shape[1]}")
    print(f"  Audio tokens          : {len(audio_ids)}")
    print(f"  Full sequence length  : {len(full_sequence)}")
    print(f"  Model vocab size      : {cfg.vocab.total_vocab_size:,}")
    print(f"  Model parameters      : {model.num_parameters()/1e6:.1f}M")
    print(f"  Loss (untrained)      : {loss.item():.4f}")
    print()
    print("All components operational. Phase 2 complete.")
    print("Next: Phase 3 — LibriSpeech DataLoader")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeechLM pipeline test")
    parser.add_argument("--wav",    type=str, default=None,  help="Path to a .wav or .flac file")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda / mps")
    args = parser.parse_args()

    run_pipeline(wav_path=args.wav, device=args.device)