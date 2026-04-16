"""Generate BOHDI wrapper traces over HealthBench for SFT training data."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("data/raw")

DATASET_URLS = {
    "healthbench": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def ensure_downloaded(name):
    """Download dataset to data/raw/ if not already there."""
    import urllib.request
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"Downloading {name}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URLS[name], path)
    return path


def load_dataset(name):
    path = ensure_downloaded(name)
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ex["_source"] = name
            examples.append(ex)
    print(f"  {name}: {len(examples)} examples")
    return examples


def load_multiple_datasets(names):
    all_ex = []
    seen = set()
    for name in names:
        for ex in load_dataset(name):
            if ex["prompt_id"] not in seen:
                seen.add(ex["prompt_id"])
                all_ex.append(ex)
    print(f"Total unique: {len(all_ex)}")
    return all_ex


def load_exclude_ids(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    return set(data)


class LocalModel:
    def __init__(self, model_name, device="auto"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

    def generate(self, messages, max_new_tokens=1024):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_response(model, messages, use_bodhi):
    if not use_bodhi:
        return model.generate(messages)

    from bodhi import BODHI, BODHIConfig
    chat_fn = lambda msgs: model.generate(msgs)
    bodhi = BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))
    resp = bodhi.complete(messages)
    return resp.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--datasets", nargs="+", default=["healthbench_hard", "healthbench"],
                        choices=list(DATASET_URLS.keys()))
    parser.add_argument("--exclude-ids", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    examples = load_multiple_datasets(args.datasets)

    if args.exclude_ids:
        exclude = load_exclude_ids(args.exclude_ids)
        before = len(examples)
        examples = [ex for ex in examples if ex["prompt_id"] not in exclude]
        print(f"Excluded {before - len(examples)} eval examples, {len(examples)} left")

    if args.max_examples:
        examples = examples[:args.max_examples]

    done_ids = set()
    if args.resume_from and Path(args.resume_from).exists():
        with open(args.resume_from) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupt lines from interrupted runs
        examples = [ex for ex in examples if ex["prompt_id"] not in done_ids]
        print(f"Resuming, skipping {len(done_ids)} already done")

    print(f"\nGenerating {len(examples)} traces, bodhi={args.use_bodhi}\n")
    model = LocalModel(args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done_ids else "w"
    ok, fail = 0, 0

    with open(out_path, mode) as f:
        for ex in tqdm(examples):
            try:
                resp = generate_response(model, ex["prompt"], args.use_bodhi)
                trace = {
                    "prompt_id": ex["prompt_id"],
                    "messages": ex["prompt"],
                    "response": resp,
                    "tags": ex.get("example_tags", []),
                    "source_dataset": ex.get("_source", "unknown"),
                    "model": args.model,
                    "bodhi": args.use_bodhi,
                }
                f.write(json.dumps(trace) + "\n")
                f.flush()
                ok += 1
            except Exception as e:
                print(f"  Error on {ex['prompt_id']}: {e}")
                fail += 1

    print(f"\nDone: {ok} ok, {fail} failed -> {out_path}")


if __name__ == "__main__":
    main()
