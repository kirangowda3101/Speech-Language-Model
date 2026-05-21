"""
checkpoint.py — Save and resume training checkpoints.

Critical DDP rule:
  When a model is wrapped in DistributedDataParallel, the actual model
  weights live at model.module, not model. Always save model.module.state_dict()
  — otherwise the checkpoint is only loadable when DDP is active.

Checkpoint saves:
  - model weights (model.module.state_dict())
  - optimizer state (for resuming with the same momentum/variance)
  - training step and epoch (to resume from the right position)
  - loss history (for plotting)
  - config (so you know what architecture was trained)
"""

import torch
import json
from pathlib import Path
from typing import Optional


def save_checkpoint(
    path: str | Path,
    model,                   # may be DDP-wrapped or raw
    optimizer,
    step: int,
    epoch: int,
    loss: float,
    cfg_dict: dict,
    is_best: bool = False,
):
    """
    Save a training checkpoint. Only call on rank 0.

    Args:
        path     : file path to save to (e.g. "checkpoints/step_10000.pt")
        model    : the model — handles both DDP-wrapped and raw
        optimizer: AdamW optimizer
        step     : current global training step
        epoch    : current epoch
        loss     : current validation loss (for tracking best)
        cfg_dict : config as a dict (for reproducibility)
        is_best  : if True, also save a copy as "best.pt"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP: save model.module if DDP, else model directly
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "step":            step,
        "epoch":           epoch,
        "loss":            loss,
        "model_state":     raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          cfg_dict,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (step={step}, loss={loss:.4f})")

    if is_best:
        best_path = path.parent / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"New best checkpoint: {best_path}")

    # Delete all step_*.pt files except the one just saved
    for old in path.parent.glob("step_*.pt"):
        if old != path:
            old.unlink()
            print(f"Deleted old checkpoint: {old}")


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).

    Works whether model is DDP-wrapped or raw.
    Returns the checkpoint dict so the caller can retrieve step/epoch/loss.

    Usage:
        ckpt = load_checkpoint("checkpoints/best.pt", model, optimizer, device)
        start_step  = ckpt["step"]
        start_epoch = ckpt["epoch"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"Loaded checkpoint: {path} "
          f"(step={checkpoint.get('step', '?')}, "
          f"loss={checkpoint.get('loss', '?'):.4f})")
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Scan a checkpoint directory and return the most recent checkpoint
    (by step number), or None if no checkpoints exist.

    Useful for auto-resuming interrupted jobs on HPC.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = [
        p for p in checkpoint_dir.glob("step_*.pt")
        if p.stem.startswith("step_")
    ]
    if not checkpoints:
        return None

    # Sort by step number
    def step_num(p):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=step_num)