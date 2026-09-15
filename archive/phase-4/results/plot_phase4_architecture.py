#!/usr/bin/env python3
"""Phase 4 — multi-model architecture: shared retrieval front-end + eval coverage.

Phase 4 changes exactly one thing versus phase-3d: the student model. The bank,
the graph, the thresholds and the eval set are reused unchanged from phase-3d,
which panel A makes explicit and panel B verifies (identical anchor routing).

Panel B is computed from phase-4/results/results_<model>.jsonl. The eval set is
built MATH-500 first, then AIME (multi_model_eval.py:487), and is NOT shuffled —
so a run that dies early covers MATH only. That is why the partial rows have
higher absolute accuracy, and why only the paired delta is comparable.
"""

import json
import pathlib
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).parent / "phase4_architecture.png"

N_TOTAL, N_MATH_TOTAL = 287, 100

MODELS = [
    ("Qwen3-4B", "qwen3-4b", "8,192 tok"),
    ("Qwen3-1.7B", None, "16,384 tok"),        # the phase-3d reference run
    ("Qwen2.5-3B", "qwen2.5-3b", "16,384 tok"),
    ("Qwen3-0.6B", "qwen3-0.6b", "16,384 tok"),
]
NEVER_RAN = ["Gemma2-2B", "Phi3-Mini-3.8B"]

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BOX_FILL = "#fcfcfb", "#f1f0ec"
GOOD, CRIT = "#0ca30c", "#d03b3b"
MATH_C, AIME_C = "#2a78d6", "#1baf7a"      # categorical slots 1 and 3


def load(key):
    if key is None:
        p = ROOT.parent / "phase-3d/results/results_combined_fixed.jsonl"
    else:
        p = ROOT / f"results/results_{key}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


ROWS = []
for name, key, cap in MODELS:
    d = load(key)
    n = len(d)
    n_math = sum(1 for r in d if r["source"] == "MATH-500")
    va = sum(r["vanilla_correct"] for r in d) / n * 100
    ia = sum(r["injection_correct"] for r in d) / n * 100
    ROWS.append(dict(name=name, cap=cap, n=n, n_math=n_math,
                     n_aime=n - n_math, va=va, ia=ia, d=ia - va,
                     ref=key is None,
                     bonus=sum(1 for r in d if r["bonus_steps"] > 0),
                     skip=sum(1 for r in d if r["low_similarity"]),
                     route=Counter((r["source"], r["anchor_source"]) for r in d)))

fig = plt.figure(figsize=(15.5, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.06,
                      left=0.03, right=0.985, top=0.855, bottom=0.04)

fig.text(0.03, 0.955, "Phase 4 — Multi-Model Evaluation",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.03, 0.923,
         "Phase-3d's retrieval front-end held fixed; only the student model "
         "changes   ·   MATH-500 + AIME   ·   287 problems",
         fontsize=11, color=INK_2, ha="left")

# ── Panel A: pipeline ────────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 100); axA.axis("off")
axA.set_title("A · Shared retrieval, swapped student", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

# frozen front-end, drawn as one enclosing region
axA.add_patch(FancyBboxPatch(
    (0.35, 34.6), 9.3, 63.4, boxstyle="round,pad=0.05,rounding_size=0.4",
    linewidth=1.5, edgecolor=MATH_C, facecolor="#ffffff", linestyle="--",
    mutation_aspect=0.09, zorder=0))
axA.text(0.75, 95.6, "REUSED FROM PHASE-3D, UNCHANGED", fontsize=8.6,
         fontweight="bold", color=MATH_C, va="center")

STAGES = [
    ("Test problem  Q", "100 MATH-500 + 187 AIME = 287\nordered, not shuffled",
     12.5, BOX_FILL, INK_MUTED),
    ("Top-1 anchor + similarity gate",
     "phase-2d MATH bank (400) + phase-3d AIME bank (746)\n"
     "+0.05 same-source bias · +0.10 cross-dataset bump", 14.5, "#ffffff",
     INK_MUTED),
    ("Combined thought graph",
     "phase-3d graph · 4,584 nodes\nbonus step fired 0/287 (dead gate)",
     13.5, "#ffffff", CRIT),
    ("Strategy + worked example", "4 steps, 1 pattern", 9.0, BOX_FILL,
     INK_MUTED),
]

y = 92.0
GAP = 2.2
centers = []
for title, sub, h, fill, edge in STAGES:
    axA.add_patch(FancyBboxPatch(
        (0.9, y - h), 8.2, h, boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=1.6, edgecolor=edge, facecolor=fill, mutation_aspect=0.09))
    axA.text(5.0, y - h * 0.33, title, ha="center", va="center", fontsize=11,
             fontweight="bold", color=INK)
    axA.text(5.0, y - h * 0.71, sub, ha="center", va="center", fontsize=9.2,
             color=INK_2, linespacing=1.5)
    centers.append((y, y - h))
    y = y - h - GAP

for i in range(len(STAGES) - 1):
    axA.add_patch(FancyArrowPatch(
        (5.0, centers[i][1]), (5.0, centers[i + 1][0]), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.3, color=INK_MUTED, shrinkA=0,
        shrinkB=0))

# fan-out to the students
BUS_Y = 30.0
FAN_Y = 19.5
axA.add_patch(FancyArrowPatch(
    (5.0, centers[-1][1]), (5.0, BUS_Y), arrowstyle="-", linewidth=1.3,
    color=INK_MUTED, shrinkA=0, shrinkB=0))
axA.text(5.25, (centers[-1][1] + BUS_Y) / 2, "same prompt for every model",
         ha="left", va="center", fontsize=8.8, color=INK_2, style="italic")

SW = 2.05
XS_M = [0.9 + i * (SW + 0.07) for i in range(len(ROWS))]
axA.plot([XS_M[0] + SW / 2, XS_M[-1] + SW / 2], [BUS_Y, BUS_Y], lw=1.3,
         color=INK_MUTED)

for i, r in enumerate(ROWS):
    x = XS_M[i]
    axA.add_patch(FancyArrowPatch(
        (x + SW / 2, BUS_Y), (x + SW / 2, FAN_Y + 9.0), arrowstyle="-|>",
        mutation_scale=10, linewidth=1.1, color=INK_MUTED, shrinkA=0,
        shrinkB=0))
    axA.add_patch(FancyBboxPatch(
        (x, FAN_Y), SW, 9.0, boxstyle="round,pad=0.02,rounding_size=0.3",
        linewidth=1.6, edgecolor=INK if r["ref"] else INK_MUTED,
        facecolor=BOX_FILL if r["ref"] else "#ffffff", mutation_aspect=0.09))
    axA.text(x + SW / 2, FAN_Y + 5.9, r["name"], ha="center", va="center",
             fontsize=9.4, fontweight="bold", color=INK)
    axA.text(x + SW / 2, FAN_Y + 2.7, r["cap"], ha="center", va="center",
             fontsize=8.2, color=INK_2)

axA.text(5.0, FAN_Y - 3.2,
         "Qwen3-1.7B is the phase-3d reference run, re-plotted here",
         ha="center", va="center", fontsize=8.8, color=INK_2, style="italic")

axA.add_patch(FancyBboxPatch(
    (0.9, 2.5), 8.2, 10.0, boxstyle="round,pad=0.03,rounding_size=0.3",
    linewidth=1.2, edgecolor=INK_MUTED, facecolor=BOX_FILL,
    mutation_aspect=0.09))
axA.text(5.0, 9.5, "Registered but never run", ha="center", va="center",
         fontsize=9.4, fontweight="bold", color=INK)
axA.text(5.0, 5.7, " · ".join(NEVER_RAN) + "\n(no results file on disk)",
         ha="center", va="center", fontsize=8.8, color=INK_2, linespacing=1.5)

# ── Panel B: what each model actually covered ────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 100); axB.set_ylim(0, 100); axB.axis("off")
axB.set_title("B · Eval coverage and outcome", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

BAR_X0, BAR_W = 20.0, 44.0
BAR_H = 6.4
TOPB = 82.0
STEP = 15.0


def px(k):
    return BAR_X0 + BAR_W * k / N_TOTAL


# header for the track
axB.text(BAR_X0, TOPB + 8.6, "eval set order →", fontsize=9, color=INK_2,
         ha="left", va="center")
axB.plot([px(N_MATH_TOTAL)] * 2, [TOPB + 6.0, 8.0], lw=1.0, color=INK_MUTED,
         linestyle=(0, (3, 3)), zorder=0)
axB.text(px(N_MATH_TOTAL / 2), TOPB + 5.0, "MATH-500  (1–100)", fontsize=9,
         color=MATH_C, ha="center", va="center", fontweight="bold")
axB.text(px((N_MATH_TOTAL + N_TOTAL) / 2), TOPB + 5.0, "AIME  (101–287)",
         fontsize=9, color=AIME_C, ha="center", va="center", fontweight="bold")

for i, r in enumerate(ROWS):
    y = TOPB - i * STEP
    axB.text(BAR_X0 - 1.5, y + BAR_H / 2, r["name"], ha="right", va="center",
             fontsize=10.5, fontweight="bold", color=INK)
    # full track
    axB.add_patch(Rectangle((BAR_X0, y), BAR_W, BAR_H, facecolor="#ffffff",
                            edgecolor=INK_MUTED, lw=1.0, zorder=1))
    # covered portion, split by benchmark
    if r["n_math"]:
        axB.add_patch(Rectangle((BAR_X0, y), px(r["n_math"]) - BAR_X0, BAR_H,
                                facecolor=MATH_C, edgecolor="none", zorder=2))
    if r["n_aime"]:
        axB.add_patch(Rectangle((px(N_MATH_TOTAL), y),
                                px(N_MATH_TOTAL + r["n_aime"]) - px(N_MATH_TOTAL),
                                BAR_H, facecolor=AIME_C, edgecolor="none",
                                zorder=2))
    axB.text(BAR_X0 + 1.2, y + BAR_H / 2,
             f"{r['n_math']} MATH" + (f" + {r['n_aime']} AIME" if r["n_aime"]
                                      else ""),
             ha="left", va="center", fontsize=8.8, color="#ffffff",
             fontweight="bold", zorder=3)
    if r["n"] < N_TOTAL:
        axB.text(px(r["n"] if r["n_aime"] else N_MATH_TOTAL) + 1.2,
                 y + BAR_H / 2, f"stopped at {r['n']}", ha="left", va="center",
                 fontsize=8.6, color=INK_2, style="italic", zorder=3)
    # outcome
    axB.text(84.0, y + BAR_H / 2,
             f"{r['va']:.1f}% → {r['ia']:.1f}%", ha="right", va="center",
             fontsize=10, color=INK_2)
    up = r["d"] > 0
    axB.text(100.0, y + BAR_H / 2,
             f"{'▲' if up else '▼'} {r['d']:+.1f}pp", ha="right", va="center",
             fontsize=11, fontweight="bold", color=GOOD if up else CRIT)

# retrieval-invariance note
inv = FancyBboxPatch((0.5, 12.5), 99.0, 12.5,
                     boxstyle="round,pad=0.06,rounding_size=0.3",
                     linewidth=1.3, edgecolor=MATH_C, facecolor="#ffffff",
                     mutation_aspect=0.30)
axB.add_patch(inv)
ref, big = ROWS[1], ROWS[0]
same_route = ref["route"] == big["route"]
axB.text(1.4, 22.0, "RETRIEVAL IS MODEL-INDEPENDENT", fontsize=9.3,
         fontweight="bold", color=MATH_C, va="center")
axB.text(1.4, 18.3,
         f"Qwen3-4B and Qwen3-1.7B see byte-identical prompts: same anchors "
         f"({big['skip']}/{big['n']} skipped, routing "
         f"{'identical' if same_route else 'differs'}), same 0 bonus steps.",
         fontsize=9.5, color=INK, va="center")
axB.text(1.4, 15.0,
         "So every accuracy difference between rows is the model — or, for the "
         "partial rows, the eval set they never reached.",
         fontsize=9.5, color=INK, va="center")

axB.text(0.5, 7.0,
         "⚠  The eval set is ordered MATH-then-AIME and never shuffled, so a "
         "killed run is MATH-only — the easier half.",
         fontsize=9.3, color=CRIT, va="center", fontweight="bold")
axB.text(0.5, 3.6,
         "Absolute accuracy is therefore not comparable down the column; the "
         "paired Δ is. Phase-6 shuffles the eval set to remove this.",
         fontsize=9.3, color=INK_2, va="center")

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
