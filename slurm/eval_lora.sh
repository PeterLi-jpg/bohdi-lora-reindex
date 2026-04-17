#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --job-name=bohdi_eval
#SBATCH --mem=250G

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
cd /orcd/home/002/sebasmos/code/bohdi-lora  # update this

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. MedGemma requires gated access."
    echo "Run: export HF_TOKEN=hf_..."
    exit 1
fi
export HF_TOKEN

echo "$(date) | starting eval on $(hostname)"
nvidia-smi --list-gpus

python scripts/download_data.py

MODEL="google/medgemma-27b-text-it"
IDS="data/raw/hard_200_sample_ids.json"
LORA="checkpoints/best"

echo "--- base, no wrapper ---"
python scripts/eval_healthbench.py \
    --model $MODEL \
    --sample-ids $IDS \
    --grader-model mistralai/Mistral-7B-Instruct-v0.3 \
    --secondary-grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output eval/base_no_wrapper.json

echo "--- base + bodhi ---"
python scripts/eval_healthbench.py \
    --model $MODEL \
    --use-bodhi \
    --sample-ids $IDS \
    --grader-model mistralai/Mistral-7B-Instruct-v0.3 \
    --secondary-grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output eval/base_bodhi.json

echo "--- lora, no wrapper ---"
python scripts/eval_healthbench.py \
    --model $MODEL \
    --lora-path $LORA \
    --sample-ids $IDS \
    --grader-model mistralai/Mistral-7B-Instruct-v0.3 \
    --secondary-grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output eval/lora_no_wrapper.json

echo "--- lora + bodhi ---"
python scripts/eval_healthbench.py \
    --model $MODEL \
    --lora-path $LORA \
    --use-bodhi \
    --sample-ids $IDS \
    --grader-model mistralai/Mistral-7B-Instruct-v0.3 \
    --secondary-grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output eval/lora_bodhi.json

echo "$(date) | done"
