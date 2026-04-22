"""
lr_schedule.py — Learning rate schedule for SpeechLM pre-training.

Schedule: linear warmup → cosine decay → min_lr floor

  max_lr ─────╮
              │╲
              │  ╲  cosine decay
              │    ╲___________
  min_lr ─────────────────────── (held here for remainder)
              │    │
          warmup  max_steps

Why this schedule?
  • Warmup prevents large gradient updates at the start when weights
    are random — the loss landscape is steep and a high LR causes
    divergence. Ramping up over 2k steps gives the model time to
    find a reasonable region before full-speed training.
  • Cosine decay smoothly reduces LR as training converges, which
    empirically gives better final loss than a sharp step decay.
  • min_lr floor (typically max_lr/10) prevents the LR from reaching
    zero, which would stop learning entirely before training ends.

This is identical to the schedule used in GPT-3, LLaMA, and Chinchilla.
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """
    Compute the learning rate at a given training step.

    Args:
        step         : current training step (0-indexed)
        max_lr       : peak learning rate (e.g. 3e-4)
        min_lr       : floor learning rate (e.g. 3e-5)
        warmup_steps : steps for linear warmup (e.g. 2000)
        max_steps    : total training steps (e.g. 100_000)

    Returns: learning rate as a float

    Usage in training loop:
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    """
    # Phase 1: linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 3: after decay — hold at min_lr
    if step >= max_steps:
        return min_lr

    # Phase 2: cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def apply_lr(optimizer, lr: float):
    """Apply a learning rate to all param groups in an optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = lr