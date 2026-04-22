"""
ddp_utils.py — Utilities for PyTorch Distributed Data Parallel (DDP).

DDP mental model:
  You launch N identical processes (one per GPU). Each process:
    1. Calls init_process_group() to establish communication
    2. Loads the same model and wraps it in DistributedDataParallel
    3. Gets a unique shard of data via DistributedSampler
    4. Runs the forward+backward pass independently
    5. DDP automatically all-reduces gradients across all processes
       (averages them) before optimizer.step()

  The result: every GPU sees a different batch but shares gradient
  information, giving the effect of a much larger batch size.

SLURM sets these environment variables automatically:
  RANK       — global rank of this process (0 = master)
  LOCAL_RANK — rank within this node (maps to GPU index)
  WORLD_SIZE — total number of processes
"""

import os
import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, bool]:
    """
    Initialise the DDP process group.

    Returns: (rank, local_rank, world_size, is_distributed)

    Call this at the very start of your training script, before
    any model or data loading.
    """
    # Check if we're actually running in a distributed context
    if "RANK" not in os.environ:
        # Single-GPU or CPU — no DDP
        return 0, 0, 1, False

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # NCCL backend: fastest for GPU-to-GPU communication (uses NVLink / PCIe)
    # Use "gloo" for CPU-only or debugging
    dist.init_process_group(backend="nccl")

    # Pin this process to its dedicated GPU
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, True


def cleanup():
    """Destroy the process group at the end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank: int) -> bool:
    """Only rank 0 should log, save checkpoints, and write to disk."""
    return rank == 0


def barrier():
    """
    Synchronise all processes at this point.

    Use before/after operations that must complete on all ranks before
    any rank proceeds (e.g., after rank 0 saves a checkpoint).
    """
    if dist.is_initialized():
        dist.barrier()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average a scalar tensor across all ranks.

    Use this to aggregate loss values for logging on rank 0:
        loss_avg = reduce_mean(loss.detach())
        if is_master(rank): log(loss_avg)
    """
    if not dist.is_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


def setup_device(local_rank: int, is_distributed: bool) -> torch.device:
    """Return the correct device for this process."""
    if is_distributed:
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(mixed_precision: bool, device: torch.device) -> torch.dtype:
    """
    Choose the right dtype for mixed precision training.

    bf16 is preferred on A100/H100 (better numerical range, no GradScaler needed).
    fp16 is the fallback for older GPUs (V100, RTX series).
    """
    if not mixed_precision or device.type == "cpu":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16   # A100, H100 — no GradScaler needed
    return torch.float16        # V100, RTX — needs GradScaler