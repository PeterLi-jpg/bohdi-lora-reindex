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

set -euo pipefail

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
cd "${BOHDI_DIR:-${SLURM_SUBMIT_DIR:?ERROR: neither BOHDI_DIR nor SLURM_SUBMIT_DIR is set (needed to find the repo root)}}"

# pick up HF_TOKEN from a local .env if the login shell didn't export it
[ -f .env ] && source .env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. MedGemma requires gated access."
    echo "Run: export HF_TOKEN=hf_..."
    exit 1
fi
export HF_TOKEN

echo "$(date) | starting eval on $(hostname)"
nvidia-smi --list-gpus

# Fail fast on missing deps / gated-access / no-GPU before the 4-config sweep.
python scripts/preflight.py

python scripts/download_data.py

MODEL="google/medgemma-27b-text-it"
IDS="data/raw/hard_200_sample_ids.json"
LORA="checkpoints/best"

echo "--- base, no wrapper ---"
python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$IDS" --output eval/base_no_wrapper.json

echo "--- base + bodhi ---"
python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$IDS" --output eval/base_bodhi.json

echo "--- lora, no wrapper ---"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --sample-ids "$IDS" --output eval/lora_no_wrapper.json

echo "--- lora + bodhi ---"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --use-bodhi --sample-ids "$IDS" --output eval/lora_bodhi.json

echo "--- U-shape stratified analysis ---"
python scripts/eval_ushape.py \
    --eval-jsons eval/base_no_wrapper.json eval/base_bodhi.json eval/lora_no_wrapper.json eval/lora_bodhi.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/ushape.json

echo "--- plot U-shape figures ---"
python scripts/plot_ushape.py --input eval/ushape.json --out-dir eval/figures

echo "$(date) | done"
