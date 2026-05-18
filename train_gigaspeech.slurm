#!/bin/bash
#SBATCH --job-name=giga_train
#SBATCH --partition=courses-gpu
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/ramanagarajayaram.k/logs/%j_train.out
#SBATCH --error=/scratch/ramanagarajayaram.k/logs/%j_train.err

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate speechlm

module load cuda/12.3.0

cd ~/Speech-Language-Model

# Pre-submit the next chained job before training starts so it is queued
# regardless of whether this job exits cleanly or times out.
NEXT_JOB_ID=$(sbatch --dependency=afterany:$SLURM_JOB_ID --parsable train_gigaspeech.slurm)
echo "Pre-submitted next job: $NEXT_JOB_ID"

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    train.py \
        --tokens_root    /scratch/ramanagarajayaram.k/gigaspeech/tokens \
        --splits         gigaspeech-m \
        --checkpoint_dir /scratch/ramanagarajayaram.k/checkpoints/speechlm_medium_giga \
        --model_size     medium \
        --resume \
        --log_interval   10 \
        --eval_interval  500

# After training exits, check the latest checkpoint step.
# If we have reached max_steps (250000), cancel the queued chain job.
LATEST=$(ls /scratch/ramanagarajayaram.k/checkpoints/speechlm_medium_giga/step_*.pt 2>/dev/null \
         | sort | tail -1)

if [ -z "$LATEST" ]; then
    echo "No checkpoint found — keeping next job $NEXT_JOB_ID queued."
else
    STEP_STR=$(basename "$LATEST" .pt | sed 's/step_//')
    STEP=$((10#$STEP_STR))   # force base-10 so leading zeros don't break comparison
    echo "Latest checkpoint: step $STEP"

    if [ "$STEP" -ge 250000 ]; then
        echo "Training complete at step $STEP. Cancelling queued job $NEXT_JOB_ID."
        scancel "$NEXT_JOB_ID"
    else
        echo "Step $STEP < 250000 — continuing in next job $NEXT_JOB_ID."
    fi
fi
