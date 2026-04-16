"""U-shape evaluation: post-hoc stratification of eval scores by difficulty.

Inspired by the Nature Medicine 2026 "ChatGPT triage" paper
(s41591-026-04297-7) which showed LLM failure rates follow an inverted-U
across clinical acuity — worst at the extremes (nonurgent and emergency),
best in the middle.

BOHDI's central claim is epistemic humility. If LoRA-BOHDI works, it should
*flatten* this U: lift the tails, especially the high-acuity failures where
overconfident wrong answers are most harmful.

This script does NO model inference. It re-aggregates existing eval
outputs from scripts/eval_healthbench.py against HealthBench metadata,
producing per-tier and per-theme stats for plotting.

Two difficulty axes:
  1. Rubric complexity tertiles (pos-point sum: easy / medium / hard)
  2. Theme (emergency_referrals / hedging / context_seeking / ...)

Reported per tier: n, mean score, fail rate (score < --fail-threshold),
Brier score when available.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_healthbench_meta(paths):
    """Map prompt_id -> {themes, pos_points, n_rubrics}."""
    meta = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                ex = json.loads(line)
                themes = [t.split(":", 1)[1] for t in ex.get("example_tags", [])
                          if t.startswith("theme:")]
                pos = sum(r["points"] for r in ex["rubrics"] if r["points"] > 0)
                meta[ex["prompt_id"]] = {
                    "themes": themes,
                    "pos_points": pos,
                    "n_rubrics": len(ex["rubrics"]),
                }
    return meta


def compute_tertile_cutoffs(meta, restrict_to=None):
    """Return (q1, q2) cutoffs for pos_points tertiles.

    If restrict_to is given (an iterable of prompt_ids), compute cutoffs on
    just that subset — useful when the eval only covers a holdout.
    """
    if restrict_to is not None:
        restrict_to = set(restrict_to)
        pts = [m["pos_points"] for pid, m in meta.items() if pid in restrict_to]
    else:
        pts = [m["pos_points"] for m in meta.values()]
    if len(pts) < 3:
        raise ValueError(f"need at least 3 examples to compute tertiles, got {len(pts)}")
    qs = statistics.quantiles(pts, n=3)
    return qs[0], qs[1]


def tier_of(pos_points, q1, q2):
    if pos_points <= q1:
        return "easy"
    if pos_points <= q2:
        return "medium"
    return "hard"


def summarize(scores, fail_threshold):
    """Standard stats for a list of per-example scores."""
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": float(sum(scores) / len(scores)),
        "median": float(statistics.median(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "fail_rate": sum(1 for s in scores if s < fail_threshold) / len(scores),
    }


def aggregate_by_tier(results, meta, q1, q2, fail_threshold):
    tier_scores = defaultdict(list)
    missing = 0
    for r in results:
        pid = r["prompt_id"]
        if pid not in meta:
            missing += 1
            continue
        tier_scores[tier_of(meta[pid]["pos_points"], q1, q2)].append(r["score"])
    out = {t: summarize(s, fail_threshold) for t, s in tier_scores.items()}
    # ensure all three tiers present even if empty
    for t in ("easy", "medium", "hard"):
        out.setdefault(t, {"n": 0})
    out["_missing_prompt_ids"] = missing
    return out


def aggregate_by_theme(results, meta, fail_threshold):
    theme_scores = defaultdict(list)
    for r in results:
        pid = r["prompt_id"]
        if pid not in meta:
            continue
        for theme in meta[pid]["themes"]:
            theme_scores[theme].append(r["score"])
    return {t: summarize(s, fail_threshold) for t, s in theme_scores.items()}


def print_table(summary, fail_threshold):
    configs = list(summary["configs"].keys())
    if not configs:
        return

    print("\n=== U-shape by rubric complexity ===")
    print(f"(tier cutoffs: pos_points q1={summary['thresholds']['q1']:.1f}, "
          f"q2={summary['thresholds']['q2']:.1f})")
    print(f"{'config':<28} {'easy mean':>12} {'med mean':>12} {'hard mean':>12} "
          f"{'easy fail':>12} {'hard fail':>12}")
    for name in configs:
        t = summary["configs"][name]["by_tier"]
        vals = []
        for k in ("easy", "medium", "hard"):
            vals.append(f"{t[k]['mean']:.3f}" if t[k].get("mean") is not None else "-")
        fail_easy = f"{t['easy']['fail_rate']:.2f}" if t["easy"].get("fail_rate") is not None else "-"
        fail_hard = f"{t['hard']['fail_rate']:.2f}" if t["hard"].get("fail_rate") is not None else "-"
        print(f"{name:<28} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} "
              f"{fail_easy:>12} {fail_hard:>12}")

    print(f"\n=== By theme (mean score; fail rate = score < {fail_threshold}) ===")
    all_themes = set()
    for c in summary["configs"].values():
        all_themes.update(c["by_theme"].keys())
    header = f"{'theme':<24} " + " ".join(f"{n:<20}" for n in configs)
    print(header)
    for theme in sorted(all_themes):
        row = [f"{theme:<24}"]
        for n in configs:
            s = summary["configs"][n]["by_theme"].get(theme, {})
            if s.get("n"):
                row.append(f"{s['mean']:.3f}/{s['fail_rate']:.2f}(n={s['n']:<3})".ljust(20))
            else:
                row.append("-".ljust(20))
        print(" ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsons", nargs="+", required=True,
                        help="eval outputs from scripts/eval_healthbench.py")
    parser.add_argument("--healthbench", nargs="+", required=True,
                        help="HealthBench JSONL files with metadata")
    parser.add_argument("--output", required=True, help="where to write aggregated JSON")
    parser.add_argument("--fail-threshold", type=float, default=0.4,
                        help="score below this counts as a failure (default 0.4, "
                             "matches filter_traces default)")
    parser.add_argument("--tertile-on-holdout-only", action="store_true",
                        help="compute tier cutoffs on eval subset rather than full dataset "
                             "(default: use full HealthBench)")
    args = parser.parse_args()

    meta = load_healthbench_meta(args.healthbench)
    print(f"Loaded metadata for {len(meta)} HealthBench prompts")

    # gather prompt_ids referenced by eval files, for the restrict-to option
    all_eval_pids = set()
    evals = []
    for ep in args.eval_jsons:
        with open(ep) as f:
            ev = json.load(f)
        evals.append((ep, ev))
        for r in ev.get("results", []):
            all_eval_pids.add(r["prompt_id"])

    restrict = all_eval_pids if args.tertile_on_holdout_only else None
    q1, q2 = compute_tertile_cutoffs(meta, restrict_to=restrict)
    print(f"Tier cutoffs (pos_points): q1={q1:.1f} q2={q2:.1f}")

    summary = {
        "thresholds": {"q1": q1, "q2": q2, "fail_below": args.fail_threshold},
        "configs": {},
    }
    for ep, ev in evals:
        name = ev.get("config") or Path(ep).stem
        results = ev.get("results", [])
        summary["configs"][name] = {
            "source": ep,
            "n_examples": len(results),
            "overall_mean": ev.get("mean"),
            "overall_brier": ev.get("brier"),
            "overall_ece": ev.get("ece"),
            "by_tier": aggregate_by_tier(results, meta, q1, q2, args.fail_threshold),
            "by_theme": aggregate_by_theme(results, meta, args.fail_threshold),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print_table(summary, args.fail_threshold)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
