#!/bin/bash
#SBATCH --job-name=giga_preprocess
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h200:4
#SBATCH --nodelist=d4055
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/ramanagarajayaram.k/logs/%j_preprocess.out
#SBATCH --error=/scratch/ramanagarajayaram.k/logs/%j_preprocess.err

source /home/ramanagarajayaram.k/.conda/etc/profile.d/conda.sh
conda activate speechlm

module load cuda/12.3.0

cd ~/Speech-Language-Model

export HF_DATASETS_CACHE=/scratch/ramanagarajayaram.k/gigaspeech/hf_cache

python preprocess.py \
    --dataset          gigaspeech \
    --gigaspeech_subset m \
    --output_root      /scratch/ramanagarajayaram.k/gigaspeech/tokens \
    --device           cuda \
    --bandwidth        6.0
