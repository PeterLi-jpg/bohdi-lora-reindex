"""Evaluate model on HealthBench Hard (base vs lora, with/without BOHDI wrapper)."""

import argparse
import json
import math
import urllib.request
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# make sure we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.filter_traces import GRADER_TEMPLATE, LocalGrader, parse_json_response, grade_trace

HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
DATA_DIR = Path("data/raw")


def load_eval_data(sample_ids_path):
    path = DATA_DIR / "healthbench_hard.jsonl"
    if not path.exists():
        print("Downloading HealthBench Hard...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    with open(sample_ids_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    eval_ids = set(data)

    filtered = [ex for ex in examples if ex["prompt_id"] in eval_ids]
    print(f"{len(filtered)} eval examples loaded")
    return filtered


def load_model(model_name, lora_path=None):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if lora_path:
        print(f"Merging LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def make_bodhi_wrapper(model, tokenizer, max_new_tokens=1024):
    """Build a reusable BODHI wrapper around the given model."""
    from bodhi import BODHI, BODHIConfig

    def chat_fn(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def gen_response(model, tokenizer, messages, use_bodhi, bodhi_wrapper=None, max_new_tokens=1024):
    if not use_bodhi:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    resp = bodhi_wrapper.complete(messages)
    return resp.content


# -- calibration metrics --

def compute_brier_score(results):
    """Brier score across all individual rubric criteria.

    Each criterion is a binary prediction: the model either meets it or not.
    We treat the overall example score (clamped to [0, 1]) as the model's
    "confidence" and each criterion outcome as the ground truth.  Lower is better.
    """
    y_true = []
    y_pred = []
    for r in results:
        conf = max(0.0, min(1.0, r["score"]))
        for crit in r["criteria_results"]:
            # positive-point criteria only (negative ones are penalties, not predictions)
            if crit["points"] > 0:
                y_true.append(1.0 if crit["criteria_met"] else 0.0)
                y_pred.append(conf)
    if not y_true:
        return None
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(results, n_bins=10):
    """Expected Calibration Error with equal-width bins.

    Same setup as Brier: example score (clamped to [0, 1]) is the predicted
    confidence, individual criterion outcomes are the labels.
    """
    y_true = []
    y_pred = []
    for r in results:
        conf = max(0.0, min(1.0, r["score"]))
        for crit in r["criteria_results"]:
            if crit["points"] > 0:
                y_true.append(1.0 if crit["criteria_met"] else 0.0)
                y_pred.append(conf)
    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        if lo == 0.0:
            mask = mask | (y_pred == 0.0)
        n = mask.sum()
        if n == 0:
            continue
        avg_conf = y_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n / len(y_true)) * abs(avg_acc - avg_conf)
    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--grader-model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    examples = load_eval_data(args.sample_ids)
    if args.max_examples:
        examples = examples[:args.max_examples]

    model, tokenizer = load_model(args.model, args.lora_path)
    grader = LocalGrader(args.grader_model)

    bodhi_wrapper = make_bodhi_wrapper(model, tokenizer) if args.use_bodhi else None

    tag = f"{'lora' if args.lora_path else 'base'}_{'bodhi' if args.use_bodhi else 'no_wrapper'}"
    print(f"\nEval: {tag}  ({len(examples)} examples)\n")

    all_results = []
    scores = []
    for ex in tqdm(examples, desc=tag):
        resp = gen_response(model, tokenizer, ex["prompt"], args.use_bodhi, bodhi_wrapper)
        grade = grade_trace(grader, ex["prompt"], resp, ex["rubrics"])
        all_results.append({
            "prompt_id": ex["prompt_id"], "response": resp,
            "score": grade["overall_score"], "tag_scores": grade["tag_scores"],
            "criteria_results": grade["criteria_results"],
        })
        scores.append(grade["overall_score"])

    brier = compute_brier_score(all_results)
    ece = compute_ece(all_results)

    summary = {
        "config": tag, "model": args.model,
        "lora_path": args.lora_path, "use_bodhi": args.use_bodhi,
        "n_examples": len(examples),
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "brier": brier, "ece": ece,
        "results": all_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    brier_str = f"{brier:.4f}" if brier is not None else "n/a"
    ece_str = f"{ece:.4f}" if ece is not None else "n/a"
    mean_str = f"{summary['mean']:.4f}" if summary['mean'] is not None else "n/a"
    std_str = f"{summary['std']:.4f}" if summary['std'] is not None else "n/a"
    print(f"\n{tag}: score={mean_str} +/- {std_str}  brier={brier_str}  ece={ece_str}  -> {out}")


if __name__ == "__main__":
    main()
