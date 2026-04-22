"""
dataset.py — LibriSpeech Dataset for SpeechLM pre-training.

Reads pre-encoded .npy token files produced by preprocess.py.
Each .npy file contains a 1D int32 array of flat audio token IDs
(already interleaved, as produced by SpeechLMTokenizer.encode_audio_codes).

Directory layout expected (produced by preprocess.py):
  tokens_root/
    train-clean-100/
      1234-56789-0001.npy    ← flat token array for one utterance
      1234-56789-0001.txt    ← transcript (optional, for text-audio training)
      ...
    train-clean-360/
      ...
    train-other-500/
      ...

Each item returned by __getitem__:
  {
    "input_ids": LongTensor (seq_len,),
    "targets":   LongTensor (seq_len,),   # next-token targets, -1 for padding
    "length":    int,                      # actual non-padded tokens
  }
"""

from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Dict
import random

from config import SpeechLMConfig, small_config
from tokenizer import SpeechLMTokenizer


class LibriSpeechTokenDataset(Dataset):
    """
    Dataset over pre-encoded LibriSpeech token files.

    Each example is a fixed-length window sampled from a tokenised utterance.
    If an utterance is shorter than seq_len, it is padded.
    If longer, a random crop is taken (so each epoch sees different crops).

    Args:
        tokens_root : root directory of pre-encoded .npy files
        splits      : list of LibriSpeech split names to include
        cfg         : SpeechLMConfig (for vocab and model settings)
        seq_len     : context window length (defaults to cfg.model.max_seq_len)
        include_text: if True, prepend transcript tokens before audio tokens
        split_seed  : random seed for deterministic train/val split
        val_fraction: fraction of data reserved for validation
        is_val      : if True, return the validation fraction
    """

    def __init__(
        self,
        tokens_root: str | Path,
        splits: List[str],
        cfg: SpeechLMConfig,
        seq_len: Optional[int] = None,
        include_text: bool = True,
        split_seed: int = 42,
        val_fraction: float = 0.001,   # ~0.1% held out for quick val
        is_val: bool = False,
    ):
        self.tokens_root  = Path(tokens_root)
        self.cfg          = cfg
        self.seq_len      = seq_len or cfg.model.max_seq_len
        self.include_text = include_text
        self.tok          = SpeechLMTokenizer(cfg.vocab)

        # Discover all .npy files across all splits
        all_files = []
        for split in splits:
            split_dir = self.tokens_root / split
            if not split_dir.exists():
                print(f"Warning: split directory not found: {split_dir}")
                continue
            npy_files = sorted(split_dir.glob("*.npy"))
            all_files.extend(npy_files)
            print(f"  {split}: {len(npy_files):,} utterances")

        if not all_files:
            raise FileNotFoundError(
                f"No .npy files found under {tokens_root}. "
                f"Run preprocess.py first."
            )

        # Deterministic train/val split
        rng = random.Random(split_seed)
        rng.shuffle(all_files)
        val_n   = max(1, int(len(all_files) * val_fraction))
        val_set = set(str(f) for f in all_files[:val_n])

        if is_val:
            self.files = [f for f in all_files if str(f) in val_set]
        else:
            self.files = [f for f in all_files if str(f) not in val_set]

        print(f"Dataset ({'val' if is_val else 'train'}): "
              f"{len(self.files):,} utterances across {splits}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one utterance and return (input_ids, targets) pair.

        Steps:
          1. Load pre-encoded audio token array from .npy
          2. Optionally prepend text tokens from paired .txt transcript
          3. Wrap in [BOS] ... [AUDIO_START] ... [AUDIO_END] [EOS] format
          4. Random crop or pad to seq_len + 1
          5. Split into input_ids[:-1] and targets[1:]
        """
        npy_path = self.files[idx]
        audio_ids = np.load(npy_path).tolist()

        # Load paired transcript if available and requested
        text_ids = []
        if self.include_text:
            txt_path = npy_path.with_suffix(".txt")
            if txt_path.exists():
                transcript = txt_path.read_text().strip()
                try:
                    text_ids = self.tok.encode_text(transcript)
                    # Cap text length — don't let long transcripts crowd out audio
                    max_text = self.seq_len // 4
                    text_ids = text_ids[:max_text]
                except Exception:
                    text_ids = []

        # Build full training sequence
        sequence = self.tok.build_training_sequence(text_ids, audio_ids)

        # Random crop to (seq_len + 1) so we can form input/target pairs
        target_len = self.seq_len + 1
        if len(sequence) >= target_len:
            # Random start so each epoch sees a different crop of long utterances
            max_start = len(sequence) - target_len
            start     = random.randint(0, max_start)
            sequence  = sequence[start : start + target_len]
        else:
            # Pad with pad_id (input) and -1 (target, ignored by cross-entropy)
            pad_needed = target_len - len(sequence)
            sequence   = sequence + [self.tok.pad_id] * pad_needed

        # Split into input / target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        targets   = torch.tensor(sequence[1:],  dtype=torch.long)

        # Mask padding in targets
        pad_mask          = input_ids == self.tok.pad_id
        targets[pad_mask] = -1

        return {
            "input_ids": input_ids,                        # (seq_len,)
            "targets":   targets,                          # (seq_len,)
            "length":    int((~pad_mask).sum()),           # actual tokens
        }

    def estimate_total_tokens(self) -> int:
        """
        Approximate total tokens across the dataset by sampling 200 files.
        Useful for computing steps_per_epoch without loading everything.
        """
        sample = self.files[:200]
        total  = sum(np.load(f).shape[0] for f in sample)
        avg    = total / len(sample)
        return int(avg * len(self.files))