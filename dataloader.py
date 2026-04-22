"""
dataloader.py — DataLoader factory for SpeechLM training.

Handles:
  • DDP-aware DistributedSampler (each GPU sees a non-overlapping shard)
  • Efficient collation (all sequences are already the same length — no padding needed)
  • Prefetching with multiple workers to hide disk I/O latency
  • Gradient accumulation awareness (effective batch size calculation)

Key concept — DistributedSampler:
  With DDP across N GPUs, you want each GPU to see a unique, non-overlapping
  subset of the data. PyTorch's DistributedSampler handles this automatically:
  it divides the dataset into N shards and gives each GPU rank its own shard.
  Without it, all GPUs would see the same data — wasted compute and incorrect
  gradient estimates.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from typing import Optional

from config import SpeechLMConfig, TrainingConfig
from dataset import LibriSpeechTokenDataset


def collate_fn(batch):
    """
    Collate a list of dataset items into a training batch.

    Because all sequences are already padded/cropped to the same seq_len
    in LibriSpeechTokenDataset.__getitem__, this is just a stack operation.

    Returns:
        input_ids : (B, seq_len) LongTensor
        targets   : (B, seq_len) LongTensor  (-1 = ignore in loss)
        lengths   : (B,) LongTensor           (actual non-padded lengths)
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    targets   = torch.stack([item["targets"]   for item in batch])
    lengths   = torch.tensor([item["length"]   for item in batch])
    return input_ids, targets, lengths


def build_dataloader(
    dataset: LibriSpeechTokenDataset,
    cfg: TrainingConfig,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """
    Build a DataLoader, DDP-aware if is_distributed=True.

    Args:
        dataset       : LibriSpeechTokenDataset instance
        cfg           : TrainingConfig (for batch_size)
        is_distributed: True when running with torch.distributed (DDP)
        rank          : this process's global rank (0..world_size-1)
        world_size    : total number of processes (GPUs)
        shuffle       : shuffle training data (set False for val)
        num_workers   : DataLoader worker processes (None = auto)

    Why num_workers matters:
        With num_workers=0 (default), data loading is synchronous —
        the GPU stalls waiting for each batch to load from disk.
        With num_workers=4+, workers preload batches in parallel so
        the GPU never waits. Rule of thumb: 4 workers per GPU on HPC.
    """
    if num_workers is None:
        # Auto-detect: use 4 workers per GPU, capped at CPU count
        import os
        cpu_count  = os.cpu_count() or 4
        num_workers = min(4, cpu_count // max(world_size, 1))

    if is_distributed:
        # DistributedSampler ensures each GPU gets a unique data shard.
        # shuffle=True in the sampler randomises within each GPU's shard.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,   # avoids uneven batches at epoch end
        )
        # NOTE: when using DistributedSampler, set shuffle=False in DataLoader
        # (the sampler handles shuffling)
        loader_shuffle = False
    else:
        sampler        = None
        loader_shuffle = shuffle

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=loader_shuffle if sampler is None else False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,    # copies batch tensors to pinned (page-locked) memory
                            # for faster CPU→GPU transfers via DMA
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # keep worker processes alive between epochs
        drop_last=True,     # ensures all batches are exactly batch_size
    )


def build_train_val_loaders(
    tokens_root: str,
    splits_train: list[str],
    cfg: SpeechLMConfig,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    num_workers: Optional[int] = None,
):
    """
    Convenience function: build both train and val DataLoaders.

    Returns: (train_loader, val_loader)
    """
    train_dataset = LibriSpeechTokenDataset(
        tokens_root, splits_train, cfg, is_val=False
    )
    val_dataset = LibriSpeechTokenDataset(
        tokens_root, splits_train, cfg, is_val=True
    )

    train_loader = build_dataloader(
        train_dataset, cfg.training,
        is_distributed=is_distributed,
        rank=rank, world_size=world_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = build_dataloader(
        val_dataset, cfg.training,
        is_distributed=is_distributed,
        rank=rank, world_size=world_size,
        shuffle=False, num_workers=num_workers,
    )

    # Print effective batch size for transparency
    eff_batch = (
        cfg.training.batch_size
        * cfg.training.grad_accum_steps
        * world_size
    )
    print(f"Effective batch size: {cfg.training.batch_size} per GPU "
          f"× {cfg.training.grad_accum_steps} grad accum "
          f"× {world_size} GPUs = {eff_batch} total")
    print(f"Train batches/epoch: {len(train_loader):,}")
    print(f"Val   batches/epoch: {len(val_loader):,}")

    return train_loader, val_loader


def set_epoch(loader: DataLoader, epoch: int):
    """
    Call this at the start of each epoch when using DistributedSampler.

    Why? DistributedSampler uses the epoch as a random seed for shuffling.
    Without calling set_epoch, all epochs see the same shuffle order.
    """
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)