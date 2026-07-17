"""
Plot baseline accuracy, latency, and throughput histograms.
Usage: python scripts/plot_baseline.py
Saves: results/baseline/histogram.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "baseline"
OUT         = RESULTS_DIR / "histogram.png"

LEVEL_LABELS = {
    1: "L1\n(Easiest)",
    2: "L2\n(Easy)",
    3: "L3\n(Medium)",
    4: "L4\n(Hard)",
    5: "L5\n(Hardest)",
}
SUBJECT_SHORT = {
    "Algebra":                "Algebra",
    "Counting & Probability": "Counting &\nProbability",
    "Geometry":               "Geometry",
    "Intermediate Algebra":   "Intermediate\nAlgebra",
    "Number Theory":          "Number\nTheory",
    "Prealgebra":             "Prealgebra",
    "Precalculus":            "Precalculus",
}

COLOR_17 = "#5B8DB8"
COLOR_8  = "#E07B54"


def load(model_key):
    path = RESULTS_DIR / f"results_{model_key}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def aggregate(records, key):
    """Group records by key, return sorted list of (group, stats_dict)."""
    groups = defaultdict(lambda: {"correct": 0, "n": 0, "lat": [], "tps": []})
    for r in records:
        g = r[key]
        groups[g]["correct"] += r["correct"]
        groups[g]["n"]       += 1
        groups[g]["lat"].append(r["latency_s"])
        groups[g]["tps"].append(r["tokens_per_s"])
    result = {}
    for g, v in groups.items():
        result[g] = {
            "accuracy": v["correct"] / v["n"],
            "avg_lat":  np.mean(v["lat"]),
            "avg_tps":  np.mean(v["tps"]),
            "n":        v["n"],
        }
    return result


r17 = load("1.7b")
r8  = load("8b")

subjects = list(SUBJECT_SHORT.keys())
levels   = [1, 2, 3, 4, 5]

def int_keys(d):
    return {int(k): v for k, v in d.items()}

s17 = aggregate(r17, "subject")
s8  = aggregate(r8,  "subject")
l17 = int_keys(aggregate(r17, "level"))
l8  = int_keys(aggregate(r8,  "level"))

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle(
    "Baseline Evaluation — Qwen 1.7B vs Qwen 8B  (n=500 problems each)\n"
    "Difficulty levels: L1=Easiest → L5=Hardest",
    fontsize=13, fontweight="bold", y=0.98
)

legend_patches = [
    mpatches.Patch(color=COLOR_17, label="Qwen 1.7B"),
    mpatches.Patch(color=COLOR_8,  label="Qwen 8B"),
]
fig.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.9)

w = 0.35


def bar_pair(ax, groups, key17, key8, ylabel, title, fmt=None, ylim=None):
    x = np.arange(len(groups))
    vals17 = [key17[g] for g in groups]
    vals8  = [key8[g]  for g in groups]
    b1 = ax.bar(x - w/2, vals17, w, color=COLOR_17)
    b2 = ax.bar(x + w/2, vals8,  w, color=COLOR_8)
    ax.set_xticks(x)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    # value labels
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        label = fmt(h) if fmt else f"{h:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, h + ax.get_ylim()[1]*0.01,
                label, ha="center", va="bottom", fontsize=7.5)
    return ax


# ── Row 1: Accuracy ─────────────────────────────────────────
bar_pair(
    axes[0, 0],
    subjects,
    {s: s17[s]["accuracy"] for s in subjects},
    {s: s8[s]["accuracy"]  for s in subjects},
    ylabel="Accuracy",
    title="Accuracy by Subject",
    fmt=lambda v: f"{v:.0%}",
    ylim=(0, 1.12),
)
axes[0, 0].set_xticklabels([SUBJECT_SHORT[s] for s in subjects], fontsize=8)
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0, 0].axhline(np.mean([r["correct"] for r in r17]) / 1,  # overall
                   color=COLOR_17, linestyle=":", linewidth=1, alpha=0.7)

# compute overall acc
oa17 = sum(r["correct"] for r in r17) / len(r17)
oa8  = sum(r["correct"] for r in r8)  / len(r8)
axes[0, 0].axhline(oa17, color=COLOR_17, linestyle=":", linewidth=1, alpha=0.7)
axes[0, 0].axhline(oa8,  color=COLOR_8,  linestyle=":", linewidth=1, alpha=0.7)

bar_pair(
    axes[0, 1],
    levels,
    {l: l17[l]["accuracy"] for l in levels},
    {l: l8[l]["accuracy"]  for l in levels},
    ylabel="Accuracy",
    title="Accuracy by Difficulty Level",
    fmt=lambda v: f"{v:.0%}",
    ylim=(0, 1.12),
)
axes[0, 1].set_xticklabels([LEVEL_LABELS[l] for l in levels], fontsize=9)
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0, 1].axhline(oa17, color=COLOR_17, linestyle=":", linewidth=1, alpha=0.7,
                   label=f"1.7B avg {oa17:.0%}")
axes[0, 1].axhline(oa8,  color=COLOR_8,  linestyle=":", linewidth=1, alpha=0.7,
                   label=f"8B avg {oa8:.0%}")
axes[0, 1].legend(fontsize=8, loc="upper right")

# ── Row 2: Latency ──────────────────────────────────────────
bar_pair(
    axes[1, 0],
    subjects,
    {s: s17[s]["avg_lat"] for s in subjects},
    {s: s8[s]["avg_lat"]  for s in subjects},
    ylabel="Avg Latency (seconds)",
    title="Avg Latency by Subject",
    fmt=lambda v: f"{v:.1f}s",
)
axes[1, 0].set_xticklabels([SUBJECT_SHORT[s] for s in subjects], fontsize=8)

bar_pair(
    axes[1, 1],
    levels,
    {l: l17[l]["avg_lat"] for l in levels},
    {l: l8[l]["avg_lat"]  for l in levels},
    ylabel="Avg Latency (seconds)",
    title="Avg Latency by Difficulty Level",
    fmt=lambda v: f"{v:.1f}s",
)
axes[1, 1].set_xticklabels([LEVEL_LABELS[l] for l in levels], fontsize=9)

# ── Row 3: Throughput ────────────────────────────────────────
bar_pair(
    axes[2, 0],
    subjects,
    {s: s17[s]["avg_tps"] for s in subjects},
    {s: s8[s]["avg_tps"]  for s in subjects},
    ylabel="Avg Tokens / Second",
    title="Throughput (tok/s) by Subject",
    fmt=lambda v: f"{v:.0f}",
)
axes[2, 0].set_xticklabels([SUBJECT_SHORT[s] for s in subjects], fontsize=8)

bar_pair(
    axes[2, 1],
    levels,
    {l: l17[l]["avg_tps"] for l in levels},
    {l: l8[l]["avg_tps"]  for l in levels},
    ylabel="Avg Tokens / Second",
    title="Throughput (tok/s) by Difficulty Level",
    fmt=lambda v: f"{v:.0f}",
)
axes[2, 1].set_xticklabels([LEVEL_LABELS[l] for l in levels], fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
