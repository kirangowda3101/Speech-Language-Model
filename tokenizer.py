"""
tokenizer.py — SpeechLM tokenizer.

Handles:
  • Text → token IDs  (via tiktoken / GPT-2 BPE)
  • Audio codes → token IDs  (flat vocab offset mapping)
  • Special tokens (pad, bos, eos, audio_start, audio_end)

Design:
  We don't build a new tokenizer — we wrap tiktoken's GPT-2 BPE
  and extend it with audio token IDs using the flat vocab layout
  defined in config.py. This means we can initialise text embeddings
  from a pretrained GPT-2 checkpoint later without any ID remapping.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Union

from config import VocabConfig, SpeechLMConfig, small_config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")


class SpeechLMTokenizer:
    """
    Tokenizer for SpeechLM.

    Text tokenization: tiktoken cl100k_base (GPT-4 BPE, 100k vocab)
    or gpt2 (50k vocab) depending on config.text_vocab_size.

    Audio tokenization: maps (codebook, code) pairs to flat token IDs.

    Example usage:
        tok = SpeechLMTokenizer(cfg.vocab)

        # Text
        ids = tok.encode_text("Hello world")         # [15496, 995]
        txt = tok.decode_text(ids)                    # "Hello world"

        # Audio (from EnCodec output)
        # encodec_codes shape: (K, T) where K=num_codebooks, T=num_frames
        audio_ids = tok.encode_audio_codes(encodec_codes)  # (K*T,) or (T*K,)

        # Mixed sequence for training: [BOS] text_ids [AUDIO_START] audio_ids [AUDIO_END] [EOS]
        full_seq = tok.build_training_sequence(text_ids, audio_ids)
    """

    def __init__(self, vocab_cfg: VocabConfig):
        self.cfg = vocab_cfg

        # ── Special token IDs ───────────────────────────────────────
        base = vocab_cfg.special_token_offset
        self.pad_id         = base + 0
        self.bos_id         = base + 1
        self.eos_id         = base + 2
        self.audio_start_id = base + 3
        self.audio_end_id   = base + 4

        # ── Text tokenizer (tiktoken) ───────────────────────────────
        self._enc = None
        if TIKTOKEN_AVAILABLE:
            # Choose encoding based on configured text vocab size
            if vocab_cfg.text_vocab_size > 60_000:
                enc_name = "cl100k_base"   # GPT-4, ~100k vocab
            else:
                enc_name = "gpt2"          # GPT-2, 50,257 vocab
            self._enc = tiktoken.get_encoding(enc_name)
            print(f"Loaded tiktoken '{enc_name}' ({self._enc.n_vocab} text tokens)")

    # ── Text methods ────────────────────────────────────────────────

    def encode_text(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode a text string to a list of token IDs.

        Text IDs are in [0, text_vocab_size) — no offset needed.
        """
        if self._enc is None:
            raise RuntimeError("tiktoken not available. Install: pip install tiktoken")
        ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode_text(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode text token IDs back to a string."""
        if self._enc is None:
            raise RuntimeError("tiktoken not available.")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Filter to text-range only (ignore audio/special tokens)
        text_ids = [t for t in token_ids if t < self.cfg.text_vocab_size]
        return self._enc.decode(text_ids)

    # ── Audio methods ────────────────────────────────────────────────

    def encode_audio_codes(
        self,
        codes: Union[np.ndarray, torch.Tensor],
        interleave: bool = True,
    ) -> List[int]:
        """
        Convert EnCodec RVQ codes to flat token IDs.

        Args:
            codes     : shape (K, T) — K codebooks, T time frames
                        Each value is in [0, codebook_size).
            interleave: If True (default), interleave codebook tokens per frame:
                          frame0_cb0, frame0_cb1, ..., frame0_cbK,
                          frame1_cb0, frame1_cb1, ..., frame1_cbK, ...
                        This is the "delay pattern" used in AudioLM/MusicGen.

                        If False, flatten as (K*T,): all of cb0, then all of cb1, ...
                        Less common, harder to model autoregressively.

        Returns:
            List of integer token IDs, length K*T.

        Interleaving explanation:
            At each time step the model must predict K tokens (one per codebook).
            Interleaving means cb0 tokens are always followed by cb1 tokens for the
            same frame, giving the model a natural "fill in the remaining codebooks"
            structure. This is how MusicGen and AudioLM train.
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        K, T = codes.shape
        assert K == self.cfg.encodec.num_codebooks, \
            f"Expected {self.cfg.encodec.num_codebooks} codebooks, got {K}"

        token_ids = []
        if interleave:
            for t in range(T):
                for k in range(K):
                    token_ids.append(self.cfg.audio_token_id(k, int(codes[k, t])))
        else:
            for k in range(K):
                for t in range(T):
                    token_ids.append(self.cfg.audio_token_id(k, int(codes[k, t])))

        return token_ids

    def decode_audio_token_ids(
        self,
        token_ids: Union[List[int], torch.Tensor],
        interleave: bool = True,
    ) -> np.ndarray:
        """
        Inverse of encode_audio_codes.

        Returns codes array of shape (K, T).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Filter to audio range only
        audio_ids = [t for t in token_ids
                     if self.cfg.audio_token_offset <= t < self.cfg.special_token_offset]

        K = self.cfg.encodec.num_codebooks
        T = len(audio_ids) // K
        codes = np.zeros((K, T), dtype=np.int32)

        if interleave:
            for i, tid in enumerate(audio_ids):
                t = i // K
                k = i  % K
                _, code = self.cfg.decode_audio_token_id(tid)
                codes[k, t] = code
        else:
            for i, tid in enumerate(audio_ids):
                k = i // T
                t = i  % T
                _, code = self.cfg.decode_audio_token_id(tid)
                codes[k, t] = code

        return codes

    # ── Sequence builders ────────────────────────────────────────────

    def build_training_sequence(
        self,
        text_ids:  List[int],
        audio_ids: List[int],
    ) -> List[int]:
        """
        Build a training sequence in AudioLM style:

          [BOS] <text tokens> [AUDIO_START] <audio tokens> [AUDIO_END] [EOS]

        This is the standard format for speech-text joint modelling.
        The model learns to:
          1. Continue text autoregressively (standard LM)
          2. Transition from text to audio at [AUDIO_START]
          3. Generate audio tokens autoregressively
          4. Close the audio segment with [AUDIO_END]

        Returns a flat list of integer token IDs.
        """
        return (
            [self.bos_id]
            + text_ids
            + [self.audio_start_id]
            + audio_ids
            + [self.audio_end_id]
            + [self.eos_id]
        )

    def build_input_target_pair(
        self,
        sequence: List[int],
        pad_to: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From a sequence, build (input_ids, targets) for teacher-forcing.

        input_ids : sequence[:-1]
        targets   : sequence[1:]   (next token at each position)

        Optionally pads to `pad_to` length using pad_id on inputs
        and -1 on targets (cross_entropy ignores -1).
        """
        input_ids = sequence[:-1]
        targets   = sequence[1:]

        if pad_to > 0:
            pad_len = pad_to - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_id] * pad_len
                targets   = targets   + [-1]          * pad_len
            else:
                input_ids = input_ids[:pad_to]
                targets   = targets[:pad_to]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets,   dtype=torch.long),
        )

    # ── Utilities ────────────────────────────────────────────────────

    def is_text_token(self, token_id: int) -> bool:
        return 0 <= token_id < self.cfg.text_vocab_size

    def is_audio_token(self, token_id: int) -> bool:
        return self.cfg.audio_token_offset <= token_id < self.cfg.special_token_offset

    def is_special_token(self, token_id: int) -> bool:
        return token_id >= self.cfg.special_token_offset

    def vocab_size(self) -> int:
        return self.cfg.total_vocab_size

    def __repr__(self) -> str:
        return (
            f"SpeechLMTokenizer("
            f"text_vocab={self.cfg.text_vocab_size:,}, "
            f"audio_vocab={self.cfg.encodec.num_audio_tokens:,}, "
            f"special={self.cfg.num_special}, "
            f"total={self.cfg.total_vocab_size:,})"
        )


# ─────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = small_config()
    tok = SpeechLMTokenizer(cfg.vocab)
    print(tok)
    print()

    # Text round-trip
    if TIKTOKEN_AVAILABLE:
        text = "The quick brown fox"
        ids  = tok.encode_text(text)
        back = tok.decode_text(ids)
        print(f"Text encode: {text!r} → {ids}")
        print(f"Text decode: {ids} → {back!r}")
        print()

    # Audio round-trip
    K, T = cfg.vocab.encodec.num_codebooks, 10   # 10 frames
    fake_codes = np.random.randint(0, cfg.vocab.encodec.codebook_size, (K, T))
    audio_ids  = tok.encode_audio_codes(fake_codes)
    print(f"Audio codes shape : ({K}, {T})")
    print(f"Audio token IDs   : {audio_ids[:8]}...  (length {len(audio_ids)})")
    recovered  = tok.decode_audio_token_ids(audio_ids)
    print(f"Round-trip OK     : {np.array_equal(fake_codes, recovered)}")
    print()

    # Build training sequence
    text_ids   = tok.encode_text("Hello") if TIKTOKEN_AVAILABLE else [1, 2, 3]
    seq        = tok.build_training_sequence(text_ids, audio_ids)
    inp, tgt   = tok.build_input_target_pair(seq, pad_to=128)
    print(f"Training sequence length : {len(seq)}")
    print(f"Input tensor shape       : {inp.shape}")
    print(f"Target tensor shape      : {tgt.shape}")
    print(f"Special tokens: BOS={tok.bos_id}, EOS={tok.eos_id}, "
          f"AUDIO_START={tok.audio_start_id}, AUDIO_END={tok.audio_end_id}")
