#!/bin/bash
#SBATCH --job-name=speechlm_train
#SBATCH --nodes=2                    # number of nodes
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4                 # 4 GPUs per node (= 8 total)
#SBATCH --cpus-per-task=8            # CPU cores per GPU (for DataLoader workers)
#SBATCH --mem=256G                   # RAM per node
#SBATCH --time=48:00:00              # wall-clock limit
#SBATCH --partition=gpu              # your cluster's GPU partition name
#SBATCH --output=logs/%j_train.out   # stdout → logs/JOBID_train.out
#SBATCH --error=logs/%j_train.err    # stderr → logs/JOBID_train.err

# ── Environment ─────────────────────────────────────────────
module load cuda/12.1
module load python/3.11
source ~/venv/speechlm/bin/activate

# ── Paths (edit these) ───────────────────────────────────────
TOKENS_ROOT=/data/librispeech_tokens
CHECKPOINT_DIR=/checkpoints/speechlm_medium
REPO_DIR=/home/$USER/speech_lm

cd $REPO_DIR
mkdir -p logs $CHECKPOINT_DIR

# ── DDP communication setup ──────────────────────────────────
# SLURM assigns MASTER_ADDR for you via srun; torchrun needs it.
export MASTER_PORT=29500
export NCCL_DEBUG=INFO          # verbose NCCL logs (remove after debugging)
export NCCL_IB_DISABLE=0        # enable InfiniBand if available
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Launch with torchrun via srun ────────────────────────────
# srun launches one task per GPU across all nodes.
# torchrun handles process group init using the SLURM env vars.
srun torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port=$MASTER_PORT \
    train.py \
        --tokens_root    $TOKENS_ROOT \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_size     medium \
        --resume \
        --log_interval   10 \
        --eval_interval  500

# ── Notes ───────────────────────────────────────────────────
# To monitor: tail -f logs/<JOBID>_train.out
# To cancel:  scancel <JOBID>
# To resume:  sbatch train.slurm  (--resume flag auto-finds latest checkpoint)
