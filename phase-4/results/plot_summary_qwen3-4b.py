import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"       # categorical slots 1/2 — vanilla / injection
GOOD, CRIT = "#006300", "#d03b3b"          # delta annotation colors

rows = [json.loads(l) for l in open("results_qwen3-4b.jsonl")]
summary = json.load(open("summary_qwen3-4b.json"))


def agg(rows, key):
    counts = defaultdict(lambda: [0, 0, 0])  # n, vanilla_correct, injection_correct
    for r in rows:
        k = key(r)
        c = counts[k]
        c[0] += 1
        c[1] += r["vanilla_correct"]
        c[2] += r["injection_correct"]
    return counts


def accs(counts, order):
    van = [counts[k][1] / counts[k][0] for k in order]
    inj = [counts[k][2] / counts[k][0] for k in order]
    n = [counts[k][0] for k in order]
    return van, inj, n


def bar_panel(ax, labels, van, inj, n, title, delta_fmt="{:+.2f}"):
    x = np.arange(len(labels))
    w = 0.35
    bv = ax.bar(x - w / 2, van, w, label="Vanilla", color=BLUE, alpha=0.9)
    bi = ax.bar(x + w / 2, inj, w, label="Injection", color=ORANGE, alpha=0.9)

    for rv, ri, cnt in zip(bv, bi, n):
        delta = ri.get_height() - rv.get_height()
        color = GOOD if delta >= 0 else CRIT
        ax.annotate(
            delta_fmt.format(delta),
            xy=(ri.get_x() + ri.get_width() / 2, max(ri.get_height(), rv.get_height()) + 0.015),
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )
        ax.annotate(f"n={cnt}", xy=(ri.get_x() - w / 2, 0.02), ha="center", va="bottom",
                     fontsize=7, color="#898781")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"Phase 4 — Qwen3-4B Memory Injection Results (n={summary['n']}: "
    f"{summary['n_math']} MATH-500 + {summary['n_aime']} AIME)",
    fontsize=14, fontweight="bold",
)

# ── Panel 1: Overall / by source ─────────────────────────────────────────────
src_order = ["MATH-500", "AIME", "Overall"]
src_counts = agg(rows, lambda r: r["source"])
van_s = [src_counts["MATH-500"][1] / src_counts["MATH-500"][0],
         src_counts["AIME"][1] / src_counts["AIME"][0],
         summary["overall"]["vanilla"]]
inj_s = [src_counts["MATH-500"][2] / src_counts["MATH-500"][0],
         src_counts["AIME"][2] / src_counts["AIME"][0],
         summary["overall"]["injection"]]
n_s = [src_counts["MATH-500"][0], src_counts["AIME"][0], summary["n"]]
bar_panel(axes[0, 0], src_order, van_s, inj_s, n_s, "Overall & by Source")

# ── Panel 2: by subject (MATH-500) ───────────────────────────────────────────
subj_counts = agg([r for r in rows if r["source"] == "MATH-500"], lambda r: r["subject"])
subjects = sorted(subj_counts.keys())
van_subj, inj_subj, n_subj = accs(subj_counts, subjects)
subj_labels = [s.replace(" & ", "\n& ").replace(" ", "\n", 1) if len(s) > 12 else s for s in subjects]
bar_panel(axes[0, 1], subj_labels, van_subj, inj_subj, n_subj, "MATH-500 — by Subject")

# ── Panel 3: by level (MATH-500) ─────────────────────────────────────────────
level_counts = agg([r for r in rows if r["source"] == "MATH-500"], lambda r: r["level"])
levels = sorted(level_counts.keys(), key=int)
van_lvl, inj_lvl, n_lvl = accs(level_counts, levels)
bar_panel(axes[1, 0], [f"Level {l}" for l in levels], van_lvl, inj_lvl, n_lvl, "MATH-500 — by Difficulty Level")

# ── Panel 4: by year (AIME) ──────────────────────────────────────────────────
year_counts = agg([r for r in rows if r["source"] == "AIME"], lambda r: r["year"])
years = sorted(year_counts.keys(), key=str)
van_yr, inj_yr, n_yr = accs(year_counts, years)
bar_panel(axes[1, 1], [str(y) for y in years], van_yr, inj_yr, n_yr, "AIME — by Year")

o = summary["overall"]
fig.text(
    0.5, 0.01,
    f"Overall: Vanilla={o['vanilla']:.1%} ({o['vanilla_correct']}/{summary['n']})  |  "
    f"Injection={o['injection']:.1%} ({o['injection_correct']}/{summary['n']})  |  "
    f"Delta={o['delta']:+.1%}  |  Improvements={o['improvements']}  Regressions={o['regressions']}",
    ha="center", fontsize=9, color="#52514e",
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("summary_histogram_qwen3-4b.png", dpi=150, bbox_inches="tight")
print("Saved: summary_histogram_qwen3-4b.png")
