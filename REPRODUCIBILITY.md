# Reproducibility Guide

Step-by-step instructions to reproduce every experiment in this repo. If a command is not written below, do not invent one — open an issue.

## 1. Prerequisites

### Hardware
- **Smoke test**: 1 GPU with >=16 GB VRAM (or CPU, slow)
- **Full run**: 1 GPU with >=80 GB VRAM for MedGemma-27B (H100 or A100-80G). Multi-GPU not required; single-device with `device_map="auto"`.

### Software
- Python 3.10 or 3.11 (tested on 3.11; 3.12 works but CUDA wheel coverage lags)
- CUDA 12.1+ for the GPU run
- Linux for the full run (slurm + autoawq). macOS is fine for smoke/dev (autoawq is skipped via platform marker).

### Hugging Face access
All three models are gated. Accept the terms on each page while logged into HF, then set `HF_TOKEN`:

| Model | Used for | URL |
|---|---|---|
| `google/medgemma-27b-text-it` | base model (paper target) | https://huggingface.co/google/medgemma-27b-text-it |
| `google/gemma-3n-E4B-it` | smoke / local iteration (Felipe: "4B has good reasoning") | https://huggingface.co/google/gemma-3n-E4B-it |
| `google/gemma-3n-E2B-it` | fallback if E4B OOMs on your box | https://huggingface.co/google/gemma-3n-E2B-it |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | grader (full) | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ |
| `Qwen/Qwen2.5-0.5B-Instruct` | grader (smoke) | ungated |

```bash
export HF_TOKEN=hf_...
```

## 2. Setup (one-time)

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
bash setup.sh
```

`setup.sh`:
1. Creates `logs/ data/raw/ data/sft/ eval/ checkpoints/`
2. `pip install -r requirements.txt`
3. Downloads HealthBench Hard + Full + Consensus into `data/raw/`

If `pip install` fails on `autoawq`, you are on macOS — that is expected and handled by a platform marker.

## 3a. Smoke test on GCP (recommended — ~20 min, ~$0.25)

Cheapest path: a single L4 GPU VM. Kill it the moment smoke passes.

```bash
# one-time: enable Compute Engine if you haven't
gcloud services enable compute.googleapis.com

# pick any GCP zone that has L4s: us-central1-a, us-east4-a, europe-west4-a
ZONE=us-central1-a
NAME=bohdi-smoke

# spin up the VM (Deep Learning VM image has CUDA + drivers preinstalled)
gcloud compute instances create $NAME \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"

# wait ~90s for first boot, then SSH in
gcloud compute ssh $NAME --zone=$ZONE

# --- inside the VM ---
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
export HF_TOKEN=hf_...      # paste yours; gemma-3n-E4B-it is gated
bash setup.sh
# Run smoke and write to a log. DO NOT use `bash smoke.sh | tee`: piping
# to tee makes the pipeline's exit code tee's (always 0), so smoke failures
# appear to pass. Use redirection or PIPESTATUS instead.
bash smoke.sh &> logs/smoke.log; echo "smoke exit=$?"
# (to watch progress from another terminal: tail -f logs/smoke.log)

# --- back on your laptop when done ---
gcloud compute instances delete $NAME --zone=$ZONE --quiet
```

If smoke passes, the pipeline is wired correctly. Scale up to a bigger GPU + real config when you're ready for a paper run.

## 3b. Smoke test on a local Mac (≥32 GB unified RAM)

Run this FIRST, every time you change any pipeline code. It exercises all four stages (generate, filter, train, eval) end-to-end with `gemma-3n-E4B-it` so bugs surface in minutes instead of after a 20-minute slurm queue. Per Felipe, 4B is the floor for meaningful BOHDI signal; E2B works too but its reasoning is often too weak to produce traces worth training on. Scale to MedGemma-27B once local runs show promise.

```bash
# default: 3 examples, ~10 min. For a meatier smoke:
N_EXAMPLES=10 bash smoke.sh
```

Validated on an M4 Max / 64 GB on 2026-04-16 — full pipeline ran in ~40 min at N_EXAMPLES=10. `bash smoke.sh` starts with a preflight check (scripts/preflight.py) that fails in ~10s if HF_TOKEN is wrong, a gated model isn't accessible, or a dep is missing, so env problems surface before trace generation.

Expected outputs:
- `data/sft/smoke/raw_traces.jsonl` — 3 BOHDI traces
- `data/sft/smoke/{train,val}.jsonl` — split of the graded traces
- `checkpoints/best/` — LoRA adapter saved by SFTTrainer
- `eval/smoke/lora.json` — eval summary with `mean`, `std`, `brier`, `ece`

If any stage errors, fix it before moving to the full pipeline. Do not submit slurm jobs with a broken smoke test.

## 4. Full pipeline (cluster)

On a slurm cluster, from the repo root:

```bash
export HF_TOKEN=hf_...
# optional: override if BOHDI_DIR should differ from $SLURM_SUBMIT_DIR
# export BOHDI_DIR=/path/to/bohdi-lora

bash run_all.sh
```

This submits four jobs as a dependency chain (`--dependency=afterok`):

| Job | Script | Time | Produces |
|---|---|---|---|
| 1 | `slurm/generate_traces.sh` | ~48 h | `data/sft/raw_traces.jsonl` |
| 2 | `slurm/filter_traces.sh`  | ~24 h | `data/sft/{train,val}.jsonl` |
| 3 | `slurm/train_lora.sh`     | ~12 h | `checkpoints/best/` |
| 4 | `slurm/eval_lora.sh`      | ~12 h | `eval/{base,lora}_{no_wrapper,bodhi}.json` |

Monitor:
```bash
squeue -u $USER
tail -f logs/generate_traces_<jobid>.out
```

### Running one stage at a time

Each stage can be submitted alone; see the commands inside the matching `slurm/*.sh`. The most common reason to do this is to resume from an interrupted generate step — `generate_traces.py` supports `--resume-from <path>` and skips prompt_ids it already produced.

## 5. Expected outputs

### Stage 1 — generate
`data/sft/raw_traces.jsonl` — one JSON object per line, each with:
```json
{
  "prompt_id": "...",
  "messages": [...],
  "response": "...",
  "tags": [...],
  "source_dataset": "healthbench_hard" | "healthbench",
  "model": "google/medgemma-27b-text-it",
  "bodhi": true
}
```
Expect ~4800 lines after excluding the 200-sample eval holdout.

### Stage 2 — filter
`data/sft/{train,val}.jsonl` — same shape plus a `"grade"` field:
```json
"grade": {
  "overall_score": 0.73,
  "tag_scores": {"accuracy": 0.8, "safety": 0.5},
  "criteria_results": [...]
}
```
Note: `overall_score` can go **negative** when negative-point rubric items are "met" (i.e., the response did the bad thing). The `--min-score 0.4` default filters these out. With the default threshold, expect 2500–3000 kept traces; 10% of them go to val.

### Stage 3 — train
`checkpoints/best/` — LoRA adapter weights + tokenizer. `trainer_state.json` has the loss curve. Training is seeded (`seed: 42` in the config); reruns with the same data produce bit-identical weights on the same hardware.

### Stage 4 — eval
Five JSON files in `eval/`:
- Four per-config files, one per `{base|lora} x {no_wrapper|bodhi}` combo
- `eval/ushape.json` — post-hoc stratified analysis (see below)

Each per-config file contains:
```json
{
  "config": "...",
  "n_examples": 200,
  "mean": 0.XX, "std": 0.XX, "median": 0.XX,
  "brier": 0.XX, "ece": 0.XX,
  "results": [...]
}
```
The headline comparison is `lora_no_wrapper.mean` vs `base_bodhi.mean` — we want the LoRA model (no wrapper) to match or beat the base model with the wrapper.

### Stage 4b — U-shape stratification (`eval/ushape.json`)

Inspired by the Nature Medicine 2026 triage paper (s41591-026-04297-7). Post-hoc aggregation that re-uses the four per-config eval outputs and HealthBench metadata — no extra model inference. Produced automatically by `slurm/eval_lora.sh` and `smoke.sh`; can be re-run manually:

```bash
python scripts/eval_ushape.py \
    --eval-jsons eval/base_no_wrapper.json eval/base_bodhi.json \
                 eval/lora_no_wrapper.json eval/lora_bodhi.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/ushape.json
```

Structure:
- `thresholds` — the pos-points quartile cutoffs used to tier examples
- `configs.<name>.by_tier` — per-tier (`easy`/`medium`/`hard`) stats: `n`, `mean`, `median`, `min`, `max`, `fail_rate` (fraction scoring below `--fail-threshold`, default 0.4)
- `configs.<name>.by_theme` — same stats broken down by HealthBench theme, e.g. `emergency_referrals`, `hedging`, `context_seeking`

The figures are rendered automatically by `scripts/plot_ushape.py`, which runs at the end of `slurm/eval_lora.sh` and `smoke.sh`. Output in `eval/figures/`:

- `u_curve.png` — mean score across `easy | medium | hard`, one line per config. The "U" on `base_no_wrapper` should flatten on `lora_no_wrapper`.
- `u_fail.png` — same x-axis, failure rate on y-axis. Classic inverted-U from the Nature Medicine paper; LoRA should flatten it.
- `theme_fail.png` — grouped bar chart, themes on x-axis, fail rate per config. `emergency_referrals` and `hedging` are highlighted (BOHDI's humility claim lives there).

To regenerate plots from an existing `eval/ushape.json` without rerunning eval:
```bash
python scripts/plot_ushape.py --input eval/ushape.json --out-dir eval/figures
```

## 6. Reproducibility guarantees

**Deterministic under the same hardware + software**:
- `random`, `numpy`, `torch` seeded to `42` (configurable via `seed:` in the training YAML)
- `SFTConfig(seed=42, data_seed=42)`
- Greedy decoding (`do_sample=False`) in trace generation and eval

**Not deterministic across**:
- Different GPU models (kernel non-determinism — cuDNN/cuBLAS)
- Different CUDA / PyTorch versions (numerical rounding differences)
- Different numbers of GPUs (batch ordering changes)

If you need bit-exact reproducibility, lock: GPU model, CUDA version, PyTorch version, and the output of `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` on HF | `HF_TOKEN` unset or missing access | Accept terms on HF model page, re-export token |
| `CUDA out of memory` on train | 27B at bf16 is ~55 GB just for weights | Use a >=80 GB GPU, or add QLoRA (not currently supported) |
| `Could not auto-detect response template` | Tokenizer's chat template is non-standard | Printed suffix shows the last 50 chars — extend `find_response_template` in `scripts/train_lora.py` if needed |
| Slurm job runs in wrong dir | `SLURM_SUBMIT_DIR` not set (manual `sbatch` from unusual location) | Set `export BOHDI_DIR=/path/to/bohdi-lora` before `run_all.sh` |
| `autoawq` install fails on Mac | Expected — no CUDA wheel | Platform marker skips it; smoke test uses ungated Qwen 0.5B grader |
| `ModuleNotFoundError: liger_kernel` on import trl | Optional TRL dep | `pip install liger-kernel` (Linux/CUDA only) |

## 8. Minimal environment freeze

After `pip install -r requirements.txt`, capture the exact versions for later:
```bash
pip freeze > requirements.lock.txt
```
Commit `requirements.lock.txt` alongside your results so reviewers can reproduce the env.
