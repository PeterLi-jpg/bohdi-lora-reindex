#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --job-name=bohdi_train
#SBATCH --mem=200G

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
export HF_TOKEN=${HF_TOKEN}  # needed for gated models
cd /orcd/home/002/sebasmos/code/bohdi-lora  # update this
export WANDB_MODE=offline

echo "$(date) | starting lora training on $(hostname)"
nvidia-smi --list-gpus

python scripts/train_lora.py --config configs/lora_medgemma27b.yaml

echo "$(date) | done"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
