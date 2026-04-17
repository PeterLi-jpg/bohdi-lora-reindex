#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.err
#SBATCH --job-name=bohdi_filter
#SBATCH --mem=200G

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
cd /orcd/home/002/sebasmos/code/bohdi-lora  # update this

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Required for gated model access."
    echo "Run: export HF_TOKEN=hf_..."
    exit 1
fi
export HF_TOKEN

echo "$(date) | starting trace filtering on $(hostname)"
nvidia-smi --list-gpus

python scripts/download_data.py

python scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output-dir data/sft/ \
    --min-score 0.4 \
    --val-ratio 0.1 \
    --max-parse-failure-rate 0.05 \
    --graded-output data/sft/all_graded.jsonl

echo "$(date) | done"
