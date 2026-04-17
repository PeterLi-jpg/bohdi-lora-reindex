# BOHDI-LoRA

LoRA fine-tuning to internalize [BOHDI](https://github.com/sebasmos/bodhi-llms) epistemic virtues (humility, calibration, abstention) into model weights, replacing the prompt wrapper with weight-level alignment.

## Motivation

LLM overconfidence is reinforced through RLHF on benchmarks that reward confident answers over abstention or clarifying questions. The BOHDI prompt wrapper addresses this at inference time, but the underlying weights still favor overconfident behavior. This project uses SFT with LoRA to internalize BOHDI virtues directly into the model so it behaves humbly **without** the wrapper.

The current repository is designed to answer a narrower question than "does this solve calibration in general?": can we improve BOHDI-style behavior on held-out HealthBench Hard prompts without needing the BOHDI wrapper at inference time?

## Base Model

[google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) — Google's 27B medical Gemma model (text-only variant). Pre-trained on medical data, so the LoRA only needs to teach behavioral virtues rather than medical knowledge. Requires accepting Google's Health AI terms on HuggingFace.

## Training Data

HealthBench Hard (1000 examples) + HealthBench Full (5000 examples) combined = 5000 unique prompts, with 200 held out from HealthBench Hard for evaluation. That gives 4800 prompts for training data generation. These are run through the BOHDI wrapper, graded against the HealthBench rubrics, and filtered with a normalized rubric score that accounts for both positive credit and negative penalties.

Because the training data and reported benchmark come from the same HealthBench family, the resulting experiment should be interpreted as HealthBench-family adaptation unless it is backed by additional external evaluation.

## Pipeline

```bash
# first-time setup (creates dirs, installs deps, downloads data)
bash setup.sh

# 1. Generate BOHDI traces
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --exclude-ids data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl --use-bodhi

# 2. Grade and filter traces
python3 scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output-dir data/sft/

# 3. Train LoRA
python3 scripts/train_lora.py --config configs/lora_medgemma27b.yaml

# 4. Evaluate
python3 scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --grader-model mistralai/Mistral-7B-Instruct-v0.3 \
    --secondary-grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output eval/lora_no_wrapper.json
```

Slurm scripts for cluster execution are in `slurm/`. Update the `cd` path and submit with `sbatch`.

## Evaluation

Four configurations are compared on the 200-sample HealthBench Hard holdout:

| Configuration | Normalized Rubric Score | Positive-Credit Score |
|---|---|---|
| Base model | | |
| Base + BOHDI wrapper | | |
| **LoRA model (no wrapper)** | | |
| LoRA + BOHDI wrapper | | |

`Normalized Rubric Score` is the main selection and reporting metric in this repo. It normalizes each prompt between its worst possible rubric outcome (all negative penalties triggered) and its best possible rubric outcome (all positive credit earned, no penalties), which makes prompt-to-prompt comparisons more stable than dividing by positive points alone.

`Positive-Credit Score` reports how much of the positive-point rubric mass was earned, without applying penalty normalization.

Calibration metrics such as Brier score and ECE are intentionally omitted until the pipeline uses a model-derived confidence signal rather than a post-hoc rubric score.

The default evaluation grader is intentionally different from the filtering grader, and the evaluation script supports an optional secondary grader for cross-checking. The key result is still row 3: does the fine-tuned model exhibit BOHDI-style behavior on held-out HealthBench Hard prompts without the prompt wrapper?

## Repository Structure

```
bohdi-lora/
├── configs/          # Training hyperparameters
├── data/
│   ├── raw/          # HealthBench eval IDs
│   └── sft/          # Generated and filtered training data
├── eval/             # Evaluation outputs
├── scripts/          # Generation, filtering, training, and eval scripts
├── slurm/            # SBATCH job scripts
└── requirements.txt
```

## References

- [sebasmos/humbleai-healthbench](https://github.com/sebasmos/humbleai-healthbench) — BOHDI evaluation framework on HealthBench
- [sebasmos/bodhi-llms](https://github.com/sebasmos/bodhi-llms) — BOHDI wrapper package (`pip install bodhi-llm`)
- [HealthBench Hard](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl) — 1000 examples
- [HealthBench Full](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl) — 5000 examples
