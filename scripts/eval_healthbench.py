"""Evaluate a model on HealthBench Hard with independent rubric graders."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.healthbench_grading import (
    DEFAULT_EVAL_GRADER_MODEL,
    LocalGrader,
    grade_trace,
)

HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
DATA_DIR = Path("data/raw")


def load_eval_data(sample_ids_path):
    path = DATA_DIR / "healthbench_hard.jsonl"
    if not path.exists():
        print("Downloading HealthBench Hard...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)

    examples = []
    with open(path) as handle:
        for line in handle:
            examples.append(json.loads(line))

    with open(sample_ids_path) as handle:
        data = json.load(handle)
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
    from bodhi import BODHI, BODHIConfig

    def chat_fn(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def summarize_grader_run(graded_results):
    normalized_scores = [item["normalized_score"] for item in graded_results]
    positive_scores = [item["positive_score"] for item in graded_results]
    legacy_scores = [item["legacy_score"] for item in graded_results]
    total_parse_failures = sum(item["parse_failures"] for item in graded_results)
    total_criteria = sum(item["total_criteria"] for item in graded_results)
    parse_failure_rate = (total_parse_failures / total_criteria) if total_criteria else 0.0

    return {
        "mean_normalized_score": float(np.mean(normalized_scores)) if normalized_scores else None,
        "std_normalized_score": float(np.std(normalized_scores)) if normalized_scores else None,
        "median_normalized_score": float(np.median(normalized_scores)) if normalized_scores else None,
        "mean_positive_score": float(np.mean(positive_scores)) if positive_scores else None,
        "mean_legacy_score": float(np.mean(legacy_scores)) if legacy_scores else None,
        "total_parse_failures": total_parse_failures,
        "total_criteria": total_criteria,
        "parse_failure_rate": parse_failure_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--grader-model", default=DEFAULT_EVAL_GRADER_MODEL)
    parser.add_argument("--secondary-grader-model", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--max-parse-failure-rate",
        type=float,
        default=0.05,
        help="fail if any grader exceeds this parse failure rate",
    )
    args = parser.parse_args()

    examples = load_eval_data(args.sample_ids)
    if args.max_examples:
        examples = examples[:args.max_examples]

    model, tokenizer = load_model(args.model, args.lora_path)
    bodhi_wrapper = make_bodhi_wrapper(model, tokenizer) if args.use_bodhi else None

    tag = f"{'lora' if args.lora_path else 'base'}_{'bodhi' if args.use_bodhi else 'no_wrapper'}"
    print(f"\nEval: {tag}  ({len(examples)} examples)\n")

    responses = []
    for ex in tqdm(examples, desc=f"{tag}: generate"):
        responses.append({
            "prompt_id": ex["prompt_id"],
            "prompt": ex["prompt"],
            "rubrics": ex["rubrics"],
            "response": gen_response(model, tokenizer, ex["prompt"], args.use_bodhi, bodhi_wrapper),
        })

    grader_models = [args.grader_model] + list(args.secondary_grader_model)
    grader_runs = []
    for idx, grader_model in enumerate(grader_models):
        grader = LocalGrader(grader_model)
        graded_results = []
        for item in tqdm(responses, desc=f"{tag}: grade[{idx + 1}/{len(grader_models)}]"):
            grade = grade_trace(grader, item["prompt"], item["response"], item["rubrics"])
            graded_results.append({
                "prompt_id": item["prompt_id"],
                "normalized_score": grade["normalized_score"],
                "positive_score": grade["positive_score"],
                "legacy_score": grade["legacy_score"],
                "score_details": grade["score_details"],
                "tag_scores": grade["tag_scores"],
                "criteria_results": grade["criteria_results"],
                "parse_failures": grade["parse_failures"],
                "total_criteria": grade["total_criteria"],
            })

        summary = summarize_grader_run(graded_results)
        if summary["parse_failure_rate"] > args.max_parse_failure_rate:
            raise RuntimeError(
                f"grader {grader_model} exceeded parse failure threshold: "
                f"{summary['parse_failure_rate']:.2%} > {args.max_parse_failure_rate:.2%}"
            )

        grader_runs.append({
            "role": "primary" if idx == 0 else "secondary",
            "grader_model": grader_model,
            "summary": summary,
            "results": graded_results,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": tag,
        "model": args.model,
        "lora_path": args.lora_path,
        "use_bodhi": args.use_bodhi,
        "n_examples": len(examples),
        "score_metric": "normalized_rubric_score",
        "evaluation_notes": [
            "Normalized rubric score accounts for both positive credit and negative penalties.",
            "Calibration metrics are intentionally omitted until the pipeline has a model-derived confidence signal.",
            "HealthBench-family evaluation does not by itself establish broader transfer beyond this benchmark family.",
        ],
        "responses": [
            {
                "prompt_id": item["prompt_id"],
                "response": item["response"],
            }
            for item in responses
        ],
        "grader_runs": grader_runs,
    }
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)

    for run in grader_runs:
        summary = run["summary"]
        print(
            f"{tag} [{run['role']}] {run['grader_model']}: "
            f"normalized={summary['mean_normalized_score']:.4f} +/- {summary['std_normalized_score']:.4f} "
            f"positive={summary['mean_positive_score']:.4f} "
            f"parse_failures={summary['total_parse_failures']}/{summary['total_criteria']}"
        )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
