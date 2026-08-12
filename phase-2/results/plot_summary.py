import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("summary_remote.json") as f:
    s = json.load(f)

vanilla_subj  = s["vanilla"]["by_subject"]
inject_subj   = s["injection"]["by_subject"]
vanilla_level = s["vanilla"]["by_level"]
inject_level  = s["injection"]["by_level"]

subjects = list(vanilla_subj.keys())
levels   = sorted(vanilla_level.keys(), key=int)

van_s = [vanilla_subj[k]["accuracy"] for k in subjects]
inj_s = [inject_subj[k]["accuracy"]  for k in subjects]
van_l = [vanilla_level[k]["accuracy"] for k in levels]
inj_l = [inject_level[k]["accuracy"]  for k in levels]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase 2 — Memory Injection Results (n=100)", fontsize=14, fontweight="bold")

# ── By Subject ──────────────────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(subjects))
w = 0.35
bars_v = ax.bar(x - w/2, van_s, w, label="Vanilla",    color="#4C72B0", alpha=0.85)
bars_i = ax.bar(x + w/2, inj_s, w, label="Injection",  color="#DD8452", alpha=0.85)

for bv, bi in zip(bars_v, bars_i):
    delta = bi.get_height() - bv.get_height()
    color = "#2ca02c" if delta >= 0 else "#d62728"
    ax.annotate(f"{delta:+.2f}",
                xy=(bi.get_x() + bi.get_width() / 2, max(bi.get_height(), bv.get_height()) + 0.01),
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([s.replace(" & ", "\n& ").replace(" ", "\n", 1) if len(s) > 12 else s
                    for s in subjects], fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by Subject")
ax.axhline(0.7,  color="#4C72B0", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(0.69, color="#DD8452", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend()

# ── By Difficulty Level ──────────────────────────────────────────────────────
ax = axes[1]
x = np.arange(len(levels))
bars_v = ax.bar(x - w/2, van_l, w, label="Vanilla",   color="#4C72B0", alpha=0.85)
bars_i = ax.bar(x + w/2, inj_l, w, label="Injection", color="#DD8452", alpha=0.85)

for bv, bi, lvl in zip(bars_v, bars_i, levels):
    n = vanilla_level[lvl]["n"]
    delta = bi.get_height() - bv.get_height()
    color = "#2ca02c" if delta >= 0 else "#d62728"
    ax.annotate(f"{delta:+.3f}",
                xy=(bi.get_x() + bi.get_width() / 2, max(bi.get_height(), bv.get_height()) + 0.01),
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
    ax.annotate(f"n={n}", xy=(x[levels.index(lvl)], 0.02),
                ha="center", va="bottom", fontsize=7, color="grey")

ax.set_xticks(x)
ax.set_xticklabels([f"Level {l}" for l in levels])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by Difficulty Level")
ax.axhline(0.7,  color="#4C72B0", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(0.69, color="#DD8452", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend()

# ── Phase 1 baseline annotation ──────────────────────────────────────────────
p1 = s["phase_1_baseline"]
fig.text(0.5, 0.01,
         f"Phase 1 baselines — Student 1.7B: {p1['student_1_7b']:.1%}  |  Teacher 8B: {p1['teacher_8b']:.1%}  |  Gap: {p1['gap']:.0%}pp",
         ha="center", fontsize=9, color="grey")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("summary_histogram.png", dpi=150, bbox_inches="tight")
print("Saved: summary_histogram.png")
plt.show()
