#!/bin/bash
#SBATCH --job-name=speechlm_giga_train
#SBATCH --partition=courses-gpu
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/ramanagarajayaram.k/logs/%j_train.out
#SBATCH --error=/scratch/ramanagarajayaram.k/logs/%j_train.err

# ── Environment ──────────────────────────────────────────────────────────────
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate speechlm

module load cuda/12.3.0

export HUGGING_FACE_HUB_TOKEN=$(cat /home/ramanagarajayaram.k/.cache/huggingface/token)

# ── Paths ─────────────────────────────────────────────────────────────────────
TOKENS_ROOT=/scratch/ramanagarajayaram.k/gigaspeech/tokens
CHECKPOINT_DIR=/home/ramanagarajayaram.k/checkpoints/speechlm_medium_giga
REPO_DIR=/home/ramanagarajayaram.k/Speech-Language-Model

mkdir -p "$CHECKPOINT_DIR" /scratch/ramanagarajayaram.k/logs

# ── DDP env vars ──────────────────────────────────────────────────────────────
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Self-chaining: pre-submit next job before training starts ─────────────────
cd "$REPO_DIR"
NEXT_JOB_ID=$(sbatch --dependency=afterany:$SLURM_JOB_ID --parsable train_gigaspeech.slurm)
echo "Pre-submitted next job: $NEXT_JOB_ID"

# ── Launch ────────────────────────────────────────────────────────────────────
srun torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port=$MASTER_PORT \
    train.py \
        --tokens_root    $TOKENS_ROOT \
        --splits         gigaspeech-m \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_size     medium \
        --resume \
        --log_interval   10 \
        --eval_interval  500

# ── Stop chaining if training is complete ─────────────────────────────────────
LATEST=$(ls "$CHECKPOINT_DIR"/step_*.pt 2>/dev/null | sort | tail -1)

if [ -z "$LATEST" ]; then
    echo "No checkpoint found — keeping next job $NEXT_JOB_ID queued."
else
    STEP_STR=$(basename "$LATEST" .pt | sed 's/step_//')
    STEP=$((10#$STEP_STR))
    echo "Latest checkpoint: step $STEP"

    if [ "$STEP" -ge 250000 ]; then
        echo "Training complete at step $STEP. Cancelling queued job $NEXT_JOB_ID."
        scancel "$NEXT_JOB_ID"
    else
        echo "Step $STEP < 250000 — continuing in next job $NEXT_JOB_ID."
    fi
fi
