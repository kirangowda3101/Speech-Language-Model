#!/bin/bash
#SBATCH --job-name=giga_preprocess
#SBATCH --partition=gpu-short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/ramanagarajayaram.k/logs/%j_preprocess.out
#SBATCH --error=/scratch/ramanagarajayaram.k/logs/%j_preprocess.err

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate speechlm

module load cuda/12.3.0

cd ~/Speech-Language-Model

mkdir -p /scratch/ramanagarajayaram.k/hf_home /scratch/ramanagarajayaram.k/cache

export HF_HOME=/scratch/ramanagarajayaram.k/hf_home
export HF_DATASETS_CACHE=/scratch/ramanagarajayaram.k/gigaspeech/hf_cache
export TRANSFORMERS_CACHE=/scratch/ramanagarajayaram.k/hf_home
export XDG_CACHE_HOME=/scratch/ramanagarajayaram.k/cache
export HUGGING_FACE_HUB_TOKEN=$(cat /home/ramanagarajayaram.k/.cache/huggingface/token)

python preprocess.py \
    --dataset          gigaspeech \
    --gigaspeech_subset m \
    --output_root      /scratch/ramanagarajayaram.k/gigaspeech/tokens \
    --device           cuda \
    --bandwidth        6.0
