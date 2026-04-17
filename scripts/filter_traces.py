"""Grade BOHDI traces with the HealthBench rubric grader and filter by score."""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm
from scripts.healthbench_grading import (
    DEFAULT_FILTER_GRADER_MODEL,
    LocalGrader,
    grade_trace,
    load_rubrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--healthbench-data", required=True, nargs="+")
    parser.add_argument("--grader-model", default=DEFAULT_FILTER_GRADER_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graded-output", default=None, help="save all graded traces for debugging")
    parser.add_argument(
        "--max-parse-failure-rate",
        type=float,
        default=0.05,
        help="fail if grader parse failures exceed this fraction of all rubric items",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rubrics_by_id = load_rubrics(args.healthbench_data)

    traces = []
    with open(args.input) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"Loaded {len(traces)} raw traces")

    grader = LocalGrader(args.grader_model)

    graded = []
    total_parse_failures = 0
    total_criteria = 0
    for trace in tqdm(traces, desc="Grading"):
        rubrics = rubrics_by_id.get(trace["prompt_id"])
        if rubrics is None:
            print(f"  no rubrics for {trace['prompt_id']}, skipping")
            continue
        result = grade_trace(grader, trace["messages"], trace["response"], rubrics)
        trace["grade"] = result
        trace["selection_score"] = result["normalized_score"]
        graded.append(trace)
        total_parse_failures += result["parse_failures"]
        total_criteria += result["total_criteria"]

    print(f"Graded {len(graded)}/{len(traces)}")
    parse_failure_rate = (total_parse_failures / total_criteria) if total_criteria else 0.0
    print(
        f"Parse failures: {total_parse_failures}/{total_criteria} "
        f"({parse_failure_rate:.2%})"
    )
    if parse_failure_rate > args.max_parse_failure_rate:
        raise RuntimeError(
            "grader parse failure rate exceeded the configured threshold: "
            f"{parse_failure_rate:.2%} > {args.max_parse_failure_rate:.2%}"
        )

    if args.graded_output:
        p = Path(args.graded_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for item in graded:
                f.write(json.dumps(item) + "\n")
        print(f"All graded traces -> {p}")

    kept = [t for t in graded if t["grade"]["normalized_score"] >= args.min_score]
    print(f"Kept {len(kept)}/{len(graded)} (threshold={args.min_score})")

    scores = [t["grade"]["normalized_score"] for t in graded]
    if scores:
        print(
            f"Normalized scores: min={min(scores):.3f} max={max(scores):.3f} "
            f"mean={sum(scores)/len(scores):.3f} median={sorted(scores)[len(scores)//2]:.3f}"
        )

    random.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio)) if len(kept) > 1 else 0
    val, train = kept[:n_val], kept[n_val:]

    for name, data in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"{name}: {len(data)} -> {path}")

    print(f"\n--- summary ---")
    print(
        f"raw={len(traces)} graded={len(graded)} kept={len(kept)} "
        f"train={len(train)} val={len(val)} "
        f"parse_failures={total_parse_failures}/{total_criteria}"
    )


if __name__ == "__main__":
    main()
