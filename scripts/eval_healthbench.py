"""Evaluate model on HealthBench Hard (base vs lora, with/without BOHDI wrapper)."""

import argparse
import json
import urllib.request
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# make sure we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.filter_traces import GRADER_TEMPLATE, LocalGrader, parse_json_response

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


def gen_response(model, tokenizer, messages, use_bodhi, max_new_tokens=1024):
    if not use_bodhi:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    from bodhi import BODHI, BODHIConfig
    def chat_fn(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    bodhi = BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))
    resp = bodhi.complete(messages)
    return resp.content


def grade_example(grader, prompt_messages, response_text, rubric_items):
    convo = prompt_messages + [{"role": "assistant", "content": response_text}]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)

    results = []
    for rubric in rubric_items:
        crit = f"[{rubric['points']}] {rubric['criterion']}"
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", crit)

        parsed = {}
        for _ in range(3):
            raw = grader.grade(prompt)
            parsed = parse_json_response(raw)
            if "criteria_met" in parsed and isinstance(parsed["criteria_met"], bool):
                break
        else:
            parsed = {"criteria_met": False, "explanation": "parse failed"}

        results.append({
            "criterion": rubric["criterion"], "points": rubric["points"],
            "tags": rubric.get("tags", []), "criteria_met": parsed["criteria_met"],
        })

    total_pos = sum(r["points"] for r in results if r["points"] > 0)
    earned = sum(r["points"] for r in results if r["criteria_met"])
    score = earned / total_pos if total_pos > 0 else 0.0

    tag_items = defaultdict(list)
    for r in results:
        for t in r.get("tags", []):
            tag_items[t].append(r)
    tag_scores = {}
    for t, items in tag_items.items():
        pos = sum(r["points"] for r in items if r["points"] > 0)
        if pos > 0:
            tag_scores[t] = sum(r["points"] for r in items if r["criteria_met"]) / pos

    return {"overall_score": score, "tag_scores": tag_scores, "criteria_results": results}


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

    tag = f"{'lora' if args.lora_path else 'base'}_{'bodhi' if args.use_bodhi else 'no_wrapper'}"
    print(f"\nEval: {tag}  ({len(examples)} examples)\n")

    all_results = []
    scores = []
    for ex in tqdm(examples, desc=tag):
        resp = gen_response(model, tokenizer, ex["prompt"], args.use_bodhi)
        grade = grade_example(grader, ex["prompt"], resp, ex["rubrics"])
        all_results.append({
            "prompt_id": ex["prompt_id"], "response": resp,
            "score": grade["overall_score"], "tag_scores": grade["tag_scores"],
            "criteria_results": grade["criteria_results"],
        })
        scores.append(grade["overall_score"])

    summary = {
        "config": tag, "model": args.model,
        "lora_path": args.lora_path, "use_bodhi": args.use_bodhi,
        "n_examples": len(examples),
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "results": all_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{tag}: {summary['mean']:.4f} +/- {summary['std']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
