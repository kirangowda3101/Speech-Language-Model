"""
train.py — SpeechLM pre-training with PyTorch DDP.

Launch with torchrun (recommended):
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 train.py --tokens_root /data/librispeech_tokens

    # Multi-node (e.g. 2 nodes × 4 GPUs = 8 GPUs total)
    # (SLURM sets MASTER_ADDR / MASTER_PORT automatically via train.slurm)
    torchrun --nproc_per_node=4 --nnodes=2 train.py --tokens_root /data/librispeech_tokens

Key training loop details:
  1. Gradient accumulation: loss.backward() is called every micro-step,
     but optimizer.step() only happens every grad_accum_steps.
     The loss is divided by grad_accum_steps to keep gradients correctly scaled.

  2. DDP gradient sync: by default DDP all-reduces gradients at every
     backward(). During accumulation steps we DON'T want this (wasteful).
     We use model.no_sync() context to skip the sync on non-final steps.

  3. Mixed precision: torch.autocast wraps the forward pass.
     GradScaler is used only for fp16 (not needed for bf16).

  4. Gradient clipping: applied AFTER unscaling (if using GradScaler)
     and BEFORE optimizer.step(). This is the correct order.
"""

import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from contextlib import nullcontext

from config import small_config, medium_config, SpeechLMConfig
from model import SpeechLM
from dataloader import build_train_val_loaders, set_epoch
from lr_schedule import get_lr, apply_lr
from checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from ddp_utils import (
    init_distributed, cleanup, is_master, barrier,
    reduce_mean, setup_device, get_dtype
)


def evaluate(model, val_loader, device, dtype, max_batches: int = 50) -> float:
    """Run evaluation loop, return mean loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (inp, tgt, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                _, loss = model(inp, targets=tgt)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def train(args, cfg: SpeechLMConfig):
    # ── 1. Init distributed ─────────────────────────────────────
    rank, local_rank, world_size, is_dist = init_distributed()
    device = setup_device(local_rank, is_dist)
    dtype  = get_dtype(cfg.training.mixed_precision, device)

    if is_master(rank):
        print(f"Training on {world_size} GPU(s), dtype={dtype}")
        print(f"Effective batch size: "
              f"{cfg.training.batch_size * cfg.training.grad_accum_steps * world_size}")

    # ── 2. Build data loaders ────────────────────────────────────
    splits = args.splits.split(",")
    train_loader, val_loader = build_train_val_loaders(
        tokens_root=args.tokens_root,
        splits_train=splits,
        cfg=cfg,
        is_distributed=is_dist,
        rank=rank,
        world_size=world_size,
    )

    # ── 3. Build model ───────────────────────────────────────────
    model = SpeechLM(cfg).to(device)

    # torch.compile: fuses ops, reduces Python overhead (~20% speedup)
    # Requires PyTorch 2.0+. Disable if it causes issues.
    if cfg.training.compile and hasattr(torch, "compile"):
        if is_master(rank):
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap in DDP after compile
    if is_dist:
        model = DDP(model, device_ids=[local_rank])

    # ── 4. Optimizer ─────────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizer(
        weight_decay=cfg.training.weight_decay,
        lr=cfg.training.max_lr,
        device=device.type,
    )

    # GradScaler only needed for fp16 (bf16 doesn't underflow)
    use_scaler = (dtype == torch.float16)
    scaler     = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── 5. Resume from checkpoint if available ───────────────────
    start_step  = 0
    start_epoch = 0
    ckpt_dir    = Path(args.checkpoint_dir)

    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest and is_master(rank):
            print(f"Resuming from {latest}")
        if latest:
            ckpt       = load_checkpoint(latest, model, optimizer, device=str(device))
            start_step  = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
        barrier()  # all ranks wait until rank 0 finishes loading

    # ── 6. Training loop ─────────────────────────────────────────
    model.train()
    step         = start_step
    best_val_loss = float("inf")
    t0           = time.perf_counter()

    for epoch in range(start_epoch, 9999):
        set_epoch(train_loader, epoch)   # reshuffle DistributedSampler

        for micro_step_in_epoch, (inp, tgt, lengths) in enumerate(train_loader):
            if step >= cfg.training.max_steps:
                break

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # Gradient accumulation: determine if this is the last micro-step
            # before an optimizer update
            accum_step = micro_step_in_epoch % cfg.training.grad_accum_steps
            is_last_micro = (accum_step == cfg.training.grad_accum_steps - 1)

            # Skip DDP gradient sync on non-final accumulation steps (efficiency)
            sync_ctx = nullcontext() if not is_dist else (
                nullcontext() if is_last_micro else model.no_sync()
            )

            # Forward + backward
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, loss = model(inp, targets=tgt)

                # Divide loss by accumulation steps so gradients average correctly
                loss = loss / cfg.training.grad_accum_steps
                scaler.scale(loss).backward()

            # Optimizer step (only on final accumulation micro-step)
            if is_last_micro:
                # Unscale before gradient clipping (required for GradScaler)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip
                )

                # Update LR
                lr = get_lr(
                    step,
                    cfg.training.max_lr,
                    cfg.training.min_lr,
                    cfg.training.warmup_steps,
                    cfg.training.max_steps,
                )
                apply_lr(optimizer, lr)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # ── Logging ──────────────────────────────────────
                if step % args.log_interval == 0:
                    # Average loss across all GPUs for accurate reporting
                    loss_val = reduce_mean(
                        (loss * cfg.training.grad_accum_steps).detach()
                    ).item()
                    t1       = time.perf_counter()
                    dt       = t1 - t0
                    t0       = t1

                    if is_master(rank):
                        print(
                            f"step {step:>7} | "
                            f"loss {loss_val:.4f} | "
                            f"lr {lr:.2e} | "
                            f"{dt*1000/args.log_interval:.0f}ms/step"
                        )

                # ── Evaluation ───────────────────────────────────
                if step % args.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, dtype)
                    val_loss = reduce_mean(
                        torch.tensor(val_loss, device=device)
                    ).item()

                    if is_master(rank):
                        print(f"  val_loss: {val_loss:.4f}")
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss

                        save_checkpoint(
                            ckpt_dir / f"step_{step:07d}.pt",
                            model, optimizer, step, epoch,
                            val_loss,
                            cfg_dict={"model": vars(cfg.model),
                                      "vocab": vars(cfg.vocab)},
                            is_best=is_best,
                        )
                    barrier()

        if step >= cfg.training.max_steps:
            break

    if is_master(rank):
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_root",    type=str, required=True)
    parser.add_argument("--splits",         type=str,
                        default="train-clean-100,train-clean-360,train-other-500")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_size",     type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--resume",         action="store_true",
                        help="Auto-resume from latest checkpoint")
    parser.add_argument("--log_interval",   type=int, default=10)
    parser.add_argument("--eval_interval",  type=int, default=500)
    args = parser.parse_args()

    cfg = {"small": small_config, "medium": medium_config}[args.model_size]()
    train(args, cfg)


if __name__ == "__main__":
    main()