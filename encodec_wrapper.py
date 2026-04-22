"""
encodec_wrapper.py — Thin wrapper around Meta's EnCodec model.

What EnCodec does (conceptually):
  Waveform (float32 samples)
      │
      ▼
  Encoder CNN  →  continuous latent z  (shape: [B, D, T_frames])
      │
      ▼
  Residual Vector Quantization (RVQ)
      │   K codebooks applied sequentially:
      │   cb0 quantizes z,         residual₁ = z - cb0(z)
      │   cb1 quantizes residual₁, residual₂ = residual₁ - cb1(residual₁)
      │   ...
      ▼
  codes  (shape: [B, K, T_frames])  ← this is what we tokenize
      │
      ▼
  Decoder CNN  →  reconstructed waveform

Key numbers (EnCodec 24kHz, bandwidth=6kbps):
  • Frame rate     : 75 frames/second
  • Num codebooks  : 8
  • Codebook size  : 1024
  • So 1 second    : 75 frames × 8 codes = 600 tokens

Install: pip install encodec
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    print("EnCodec not installed. Run: pip install encodec")


class EnCodecWrapper(nn.Module):
    """
    Wraps EncodecModel with a clean encode/decode API
    that integrates directly with SpeechLMTokenizer.

    Usage:
        wrapper = EnCodecWrapper(bandwidth=6.0, device="cuda")

        # Encode
        codes = wrapper.encode(waveform)          # (K, T)

        # Decode
        audio = wrapper.decode(codes)             # (samples,)
    """

    def __init__(
        self,
        bandwidth: float = 6.0,   # kbps — controls how many codebooks are active
        device: str = "cpu",
    ):
        super().__init__()

        if not ENCODEC_AVAILABLE:
            raise ImportError("Install encodec: pip install encodec")

        # Load the 24kHz model (matches our VocabConfig defaults)
        # EnCodec has two variants: 24kHz (speech) and 48kHz (music).
        # We use 24kHz because LibriSpeech is speech data.
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()                      # always eval — we never train EnCodec
        self.model.to(device)

        self.device = device
        self.sample_rate = self.model.sample_rate       # 24000
        self.bandwidth   = bandwidth

        # How many codebooks are active at this bandwidth
        # EnCodec 24kHz: 1.5kbps→2cb, 3.0→4cb, 6.0→8cb, 12.0→16cb
        self.num_codebooks = self.model.quantizer.get_num_quantizers_for_bandwidth(
            self.model.frame_rate, bandwidth
        )

        print(
            f"EnCodecWrapper ready: 24kHz, {bandwidth}kbps, "
            f"{self.num_codebooks} codebooks, frame_rate={self.model.frame_rate}Hz"
        )

    @property
    def frame_rate(self) -> int:
        """Frames per second (75 for 24kHz EnCodec)."""
        return self.model.frame_rate

    @torch.no_grad()
    def encode(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a waveform to RVQ codes.

        Args:
            waveform   : (samples,) or (channels, samples) or (1, channels, samples)
                         float32 audio samples
            sample_rate: if provided and != 24000, will resample automatically

        Returns:
            codes: torch.Tensor of shape (K, T)
                   K = num_codebooks, T = num_frames
                   values in [0, codebook_size)

        Why @torch.no_grad()?
            EnCodec is a frozen pretrained model. We never backprop through it.
            Disabling grad tracking saves memory and speeds up the data pipeline.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        # Normalise shape to (1, channels, samples) — EnCodec's expected input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)   # (1, 1, samples)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)                # (1, channels, samples)
        # else already (1, channels, samples)

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = convert_audio(
                waveform, sample_rate, self.sample_rate,
                self.model.channels
            )

        waveform = waveform.to(self.device)

        # Encode → list of EncodedFrame (each is (codes, scale))
        # codes shape per frame: (B, K, T_chunk)
        encoded_frames = self.model.encode(waveform)

        # Concatenate chunks along time dimension
        # EnCodec chunks long audio; we stitch the codes back together
        # NEW
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        # codes: (B=1, K, T)

        return codes[0]  # strip batch dim → (K, T)

    @torch.no_grad()
    def decode(
        self,
        codes: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Decode RVQ codes back to a waveform.

        Args:
            codes: (K, T) integer tensor — values in [0, codebook_size)

        Returns:
            waveform: (samples,) float32 tensor, sample_rate=24000
        """
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes).long()

        # EnCodec expects (B, K, T)
        codes = codes.unsqueeze(0).to(self.device)   # (1, K, T)

        # Wrap in EncodedFrame format
        from encodec.model import EncodedFrame
        frames = [(codes, None)]  # (codes, scale=None)

        waveform = self.model.decode(frames)
        # waveform: (B=1, channels=1, samples)

        return waveform[0, 0].cpu()  # → (samples,)

    def tokens_per_second(self) -> int:
        """Number of audio tokens produced per second of audio."""
        return self.frame_rate * self.num_codebooks

    def frames_for_duration(self, seconds: float) -> int:
        """How many EnCodec frames does `seconds` of audio produce?"""
        return int(seconds * self.frame_rate)

    def tokens_for_duration(self, seconds: float) -> int:
        """How many flat token IDs does `seconds` of audio produce?"""
        return self.frames_for_duration(seconds) * self.num_codebooks


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ENCODEC_AVAILABLE:
        print("Install encodec first: pip install encodec")
        exit()

    import torchaudio

    wrapper = EnCodecWrapper(bandwidth=6.0, device="cpu")

    print(f"\nTokens per second of audio : {wrapper.tokens_per_second()}")
    print(f"Tokens for 10s of audio    : {wrapper.tokens_for_duration(10)}")
    print(f"Tokens for 60s of audio    : {wrapper.tokens_for_duration(60)}")

    # Synthesize a 1-second sine wave as a stand-in for real audio
    sr     = 24_000
    t      = torch.linspace(0, 1, sr)
    wave   = 0.5 * torch.sin(2 * torch.pi * 440 * t)   # 440Hz tone, 1 second

    print(f"\nInput waveform : {wave.shape}, sample_rate={sr}")

    codes = wrapper.encode(wave, sample_rate=sr)
    print(f"Encoded codes  : {codes.shape}  (K={codes.shape[0]}, T={codes.shape[1]})")
    print(f"Code range     : [{codes.min()}, {codes.max()}]")

    recon = wrapper.decode(codes)
    print(f"Decoded wave   : {recon.shape}")

    # Save for listening (optional)
    torchaudio.save("test_reconstruction.wav", recon.unsqueeze(0), sr)
    print("\nSaved test_reconstruction.wav — listen to verify quality")