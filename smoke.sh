#!/bin/bash
# Smoke test: run the full pipeline end-to-end on a tiny subset with a
# small Gemma model. Meant to catch bugs (template/tokenizer/dtype/path)
# locally before burning slurm time on the 27B run.
#
# Prereqs:
#   - HF_TOKEN set (gemma-2-2b-it is gated; accept terms on HF first)
#   - Python env with requirements installed (bash setup.sh)
#
# Runs in ~5-10 min on a single consumer GPU. CPU-only works but is slow.

set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Gemma models are gated."
    echo "  1. Accept terms at https://huggingface.co/google/gemma-3n-E4B-it"
    echo "  2. export HF_TOKEN=hf_..."
    exit 1
fi

MODEL="google/gemma-3n-E4B-it"
# small non-gated grader so smoke doesn't need a second gated access
GRADER="Qwen/Qwen2.5-0.5B-Instruct"
N_EXAMPLES=3

echo "=== smoke test | $(date) ==="
echo "model:   $MODEL"
echo "grader:  $GRADER"
echo "samples: $N_EXAMPLES"
echo

mkdir -p data/sft/smoke eval/smoke logs

echo "--- 1/4: download data ---"
python scripts/download_data.py

echo "--- 2/4: generate $N_EXAMPLES BOHDI traces ---"
python scripts/generate_traces.py \
    --model "$MODEL" \
    --datasets healthbench_hard \
    --output data/sft/smoke/raw_traces.jsonl \
    --use-bodhi \
    --max-examples $N_EXAMPLES

echo "--- 3/4: grade and filter (threshold lowered so nothing is dropped) ---"
python scripts/filter_traces.py \
    --input data/sft/smoke/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl \
    --grader-model "$GRADER" \
    --output-dir data/sft/smoke \
    --min-score -999 \
    --val-ratio 0.34

echo "--- 4a/4: train 1 epoch on the smoke set ---"
python scripts/train_lora.py --config configs/lora_gemma_smoke.yaml

echo "--- 4b/4: eval on $N_EXAMPLES examples ---"
python scripts/eval_healthbench.py \
    --model "$MODEL" \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --grader-model "$GRADER" \
    --output eval/smoke/lora.json \
    --max-examples $N_EXAMPLES

echo "--- 4c/4: U-shape stratification (tertiles on holdout only since n=$N_EXAMPLES) ---"
python scripts/eval_ushape.py \
    --eval-jsons eval/smoke/lora.json \
    --healthbench data/raw/healthbench_hard.jsonl \
    --tertile-on-holdout-only \
    --output eval/smoke/ushape.json

echo
echo "=== smoke test PASSED | $(date) ==="
echo "artifacts:"
echo "  data/sft/smoke/{train,val}.jsonl"
echo "  checkpoints/best/"
echo "  eval/smoke/lora.json"
echo "  eval/smoke/ushape.json"
