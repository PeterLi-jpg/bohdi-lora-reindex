"""Render the U-shape eval into paper-ready figures.

Consumes eval/ushape.json produced by scripts/eval_ushape.py.
Produces:
  - u_curve.png      — mean score across easy/medium/hard tiers, one line per config
  - u_fail.png       — failure rate across the same tiers (the classic "U")
  - theme_fail.png   — per-theme failure rate (emergency_referrals, hedging, ...)

No model inference happens here. Purely post-hoc plotting.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — works on slurm/GCP
import matplotlib.pyplot as plt


TIER_ORDER = ["easy", "medium", "hard"]

# Stable config order for consistent colors across figures
CONFIG_ORDER = [
    "base_no_wrapper",
    "base_bodhi",
    "lora_no_wrapper",
    "lora_bodhi",
]
CONFIG_COLORS = {
    "base_no_wrapper": "#c0392b",   # red — worst expected
    "base_bodhi":      "#e67e22",   # orange
    "lora_no_wrapper": "#2980b9",   # blue — headline config
    "lora_bodhi":      "#27ae60",   # green — best expected
}
CONFIG_LABELS = {
    "base_no_wrapper": "Base (no wrapper)",
    "base_bodhi":      "Base + BOHDI wrapper",
    "lora_no_wrapper": "LoRA (no wrapper)",
    "lora_bodhi":      "LoRA + BOHDI wrapper",
}


def ordered_configs(summary):
    """Respect CONFIG_ORDER where possible, append any unknown names at the end."""
    present = list(summary["configs"].keys())
    ordered = [c for c in CONFIG_ORDER if c in present]
    extras = [c for c in present if c not in CONFIG_ORDER]
    return ordered + extras


def plot_u_curve(summary, out_path, metric="mean", ylabel=None, title=None):
    """One line per config, x=tier, y=metric. The 'U' (or inverted-U) appears
    as a dip or peak in the middle tier."""
    configs = ordered_configs(summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        tier_stats = summary["configs"][cfg]["by_tier"]
        ys = []
        for t in TIER_ORDER:
            s = tier_stats.get(t, {})
            ys.append(s.get(metric))
        # skip configs missing data entirely
        if all(y is None for y in ys):
            continue
        ys_plot = [y if y is not None else float("nan") for y in ys]
        ax.plot(TIER_ORDER, ys_plot,
                marker="o", linewidth=2.5, markersize=9,
                color=CONFIG_COLORS.get(cfg, None),
                label=CONFIG_LABELS.get(cfg, cfg))
    ax.set_xlabel("Difficulty tier (rubric positive-point tertiles)", fontsize=11)
    ax.set_ylabel(ylabel or metric, fontsize=11)
    ax.set_title(title or f"{metric} across difficulty tiers", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if metric == "fail_rate":
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_theme_fails(summary, out_path,
                     priority_themes=("emergency_referrals", "hedging",
                                      "context_seeking", "complex_responses")):
    """Grouped bar chart of fail_rate per theme, grouped by config.

    Highlight the priority themes — emergency_referrals and hedging are the
    BOHDI-relevant ones.
    """
    configs = ordered_configs(summary)

    # collect all themes referenced by any config
    all_themes = set()
    for cfg in configs:
        all_themes.update(summary["configs"][cfg]["by_theme"].keys())
    # Put priority themes first, then the rest alphabetized
    themes = [t for t in priority_themes if t in all_themes]
    themes += sorted(t for t in all_themes if t not in themes)
    if not themes:
        print(f"  (no theme data; skipping {out_path})")
        return

    n_configs = len(configs)
    n_themes = len(themes)
    width = 0.8 / max(n_configs, 1)

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_themes), 5.5))
    xs = range(n_themes)
    for i, cfg in enumerate(configs):
        ys = []
        for theme in themes:
            s = summary["configs"][cfg]["by_theme"].get(theme, {})
            ys.append(s.get("fail_rate") if s.get("n") else 0.0)
        offsets = [x + (i - (n_configs - 1) / 2) * width for x in xs]
        ax.bar(offsets, ys, width,
               color=CONFIG_COLORS.get(cfg, None),
               label=CONFIG_LABELS.get(cfg, cfg))

    ax.set_xticks(list(xs))
    ax.set_xticklabels(themes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Failure rate (score < threshold)", fontsize=11)
    thr = summary["thresholds"].get("fail_below")
    ax.set_title(
        "Failure rate by HealthBench theme"
        + (f" (threshold = {thr})" if thr is not None else ""),
        fontsize=12,
    )
    # shade the priority themes lightly so the eye lands there
    for j, theme in enumerate(themes):
        if theme in priority_themes[:2]:  # emergency_referrals + hedging
            ax.axvspan(j - 0.45, j + 0.45, color="yellow", alpha=0.08, zorder=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval/ushape.json",
                        help="path to ushape.json produced by eval_ushape.py")
    parser.add_argument("--out-dir", default="eval/figures")
    args = parser.parse_args()

    summary = json.loads(Path(args.input).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_u_curve(
        summary, out_dir / "u_curve.png",
        metric="mean",
        ylabel="Mean score (higher = better)",
        title="U-shape: mean score across difficulty tiers",
    )
    plot_u_curve(
        summary, out_dir / "u_fail.png",
        metric="fail_rate",
        ylabel=f"Fail rate (score < {summary['thresholds'].get('fail_below', 0.4)})",
        title="U-shape: failure rate across difficulty tiers",
    )
    plot_theme_fails(summary, out_dir / "theme_fail.png")

    print(f"\nWrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
