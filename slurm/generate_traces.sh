#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/generate_traces_%j.out
#SBATCH --error=logs/generate_traces_%j.err
#SBATCH --job-name=bohdi_gen
#SBATCH --mem=200G

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
export HF_TOKEN=${HF_TOKEN}  # needed for gated models
cd /orcd/home/002/sebasmos/code/bohdi-lora  # update this

echo "$(date) | starting trace generation on $(hostname)"
nvidia-smi --list-gpus

python scripts/download_data.py

python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --exclude-ids data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl \
    --use-bodhi \
    --resume-from data/sft/raw_traces.jsonl

echo "$(date) | done"
