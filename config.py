"""
config.py — SpeechLM hyperparameters and vocabulary layout.

Design note:
  EnCodec uses Residual Vector Quantization (RVQ) with K codebooks,
  each having V codes. We treat every (codebook, code) pair as a
  distinct token ID so the model sees a flat integer sequence —
  exactly like GPT sees BPE token IDs.

"""

from dataclasses import dataclass, field


@dataclass
class EnCodecConfig:
    num_codebooks: int = 8       # RVQ depth (EnCodec 24kHz default)
    codebook_size: int = 1024    # codes per codebook
    sample_rate:   int = 24_000  # Hz
    bandwidth:     float = 6.0   # kbps — controls num_codebooks used at runtime

    @property
    def num_audio_tokens(self) -> int:
        """Total distinct audio token IDs = K × V."""
        return self.num_codebooks * self.codebook_size


@dataclass
class VocabConfig:

    text_vocab_size:  int = 50_257   # GPT-2 BPE vocab (tiktoken cl100k: 100_277)
    encodec: EnCodecConfig = field(default_factory=EnCodecConfig)

    # Special tokens (appended after audio tokens)
    pad_token:        str = "<|pad|>"
    bos_token:        str = "<|startoftext|>"
    eos_token:        str = "<|endoftext|>"
    audio_start:      str = "<|startofaudio|>"
    audio_end:        str = "<|endofaudio|>"
    num_special:      int = 5

    @property
    def audio_token_offset(self) -> int:
        """First audio token ID = text_vocab_size."""
        return self.text_vocab_size

    @property
    def special_token_offset(self) -> int:
        return self.text_vocab_size + self.encodec.num_audio_tokens

    @property
    def total_vocab_size(self) -> int:
        return self.text_vocab_size + self.encodec.num_audio_tokens + self.num_special

    def audio_token_id(self, codebook: int, code: int) -> int:
        """
        Map (codebook index, code index) → flat token ID.

        Example: codebook=0, code=42  → 50_257 + 0*1024 + 42 = 50_299
                 codebook=3, code=7   → 50_257 + 3*1024 + 7  = 53_336
        """
        assert 0 <= codebook < self.encodec.num_codebooks
        assert 0 <= code     < self.encodec.codebook_size
        return self.audio_token_offset + codebook * self.encodec.codebook_size + code

    def decode_audio_token_id(self, token_id: int) -> tuple[int, int]:
        """Inverse of audio_token_id. Returns (codebook, code)."""
        assert self.audio_token_offset <= token_id < self.special_token_offset
        offset = token_id - self.audio_token_offset
        return divmod(offset, self.encodec.codebook_size)  # (codebook, code)


@dataclass
class ModelConfig:
    """
    Transformer architecture hyperparameters.

    Sized for a ~117M param model (GPT-2 small scale) — fits in
    ~6GB VRAM for training with mixed precision. Scale up for HPC runs.
    """
    d_model:     int = 768    # embedding dimension
    n_heads:     int = 12     # attention heads (d_model must be divisible)
    n_layers:    int = 12     # transformer blocks
    d_ff:        int = 3072   # feed-forward hidden dim (4 × d_model)
    max_seq_len: int = 2048   # context window
    dropout:     float = 0.1
    bias:        bool = True  # include bias in Linear layers (False = ~10% faster)

    # Audio-specific
    # Each timestep in EnCodec has K tokens (one per codebook).
    # We flatten them into K sequential positions before feeding to GPT.
    # This means 1 second of 24kHz audio ≈ 75 frames × 8 codebooks = 600 tokens.
    interleave_codebooks: bool = True  # flatten RVQ codes into sequence

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size:       int   = 8       # per GPU
    grad_accum_steps: int   = 4       # effective batch = batch_size × accum × n_gpus
    max_lr:           float = 3e-4
    min_lr:           float = 3e-5    # cosine decay floor
    warmup_steps:     int   = 2_000
    max_steps:        int   = 100_000
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    mixed_precision:  bool  = True    # bf16 on A100/H100, fp16 otherwise
    compile:          bool  = True    # torch.compile for ~20% speedup


@dataclass
class SpeechLMConfig:
    vocab:    VocabConfig    = field(default_factory=VocabConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Convenience constructors for common scales
# ---------------------------------------------------------------------------

def small_config() -> SpeechLMConfig:
    """~117M params. Good for local dev / ablations."""
    return SpeechLMConfig()  # defaults are already small-scale


def medium_config() -> SpeechLMConfig:
    """~350M params. Recommended for LibriSpeech 960h pre-training."""
    cfg = SpeechLMConfig()
    cfg.model.d_model  = 1024
    cfg.model.n_heads  = 16
    cfg.model.n_layers = 24
    cfg.model.d_ff     = 4096
    return cfg


def large_config() -> SpeechLMConfig:
    """~760M params. Use with 8+ A100s."""
    cfg = SpeechLMConfig()
    cfg.model.d_model  = 1280
    cfg.model.n_heads  = 20
    cfg.model.n_layers = 36
    cfg.model.d_ff     = 5120
    return cfg


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = small_config()
    v = cfg.vocab
    print(f"Total vocab size : {v.total_vocab_size:,}")
    print(f"Text tokens      : {v.text_vocab_size:,}")
    print(f"Audio tokens     : {v.encodec.num_audio_tokens:,}  ({v.encodec.num_codebooks} codebooks × {v.encodec.codebook_size})")
    print(f"Special tokens   : {v.num_special}")
    print()
    print(f"Audio token ID example: codebook=0, code=42  → {v.audio_token_id(0, 42)}")
    print(f"Decode back             → codebook, code = {v.decode_audio_token_id(v.audio_token_id(0, 42))}")
    print()
    m = cfg.model
    print(f"d_model={m.d_model}, n_heads={m.n_heads}, d_head={m.d_head}, n_layers={m.n_layers}")
    print(f"Context window: {m.max_seq_len} tokens")
    print(f"~1 sec of audio ≈ {75 * m.d_model // m.d_model * 8} tokens (75 frames × 8 codebooks)")