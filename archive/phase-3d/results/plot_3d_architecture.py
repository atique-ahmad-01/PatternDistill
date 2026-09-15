#!/usr/bin/env python3
"""Phase 3D — combined MATH-500 + AIME architecture: pipeline + anchor routing.

Panel B is drawn from the run itself: phase-3d/results/results_combined_fixed.jsonl
gives the source → anchor-source routing counts, and bonus_steps == 0 on all 287
problems. Graph stats come from phase-3d/graph/thought_graph.json.
"""

import json
import pathlib
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).parent / "phase3d_architecture.png"

RES = [json.loads(l) for l
       in (ROOT / "results/results_combined_fixed.jsonl").read_text().splitlines()
       if l.strip()]
GMETA = json.loads((ROOT / "graph/thought_graph.json").read_text())["metadata"]

N = len(RES)
INJ = [r for r in RES if not r["low_similarity"]]
SKIP = [r for r in RES if r["low_similarity"]]
ROUTE = Counter((r["source"], r["anchor_source"]) for r in INJ)
SKIP_BY_SRC = Counter(r["source"] for r in SKIP)
N_BONUS = sum(1 for r in RES if r["bonus_steps"] > 0)
N_MATH = sum(1 for r in RES if r["source"] == "MATH-500")
N_AIME = sum(1 for r in RES if r["source"] == "AIME")

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BOX_FILL = "#fcfcfb", "#f1f0ec"
CRIT = "#d03b3b"
SAME, CROSS = "#2a78d6", "#eb6834"      # categorical slots 1 and 2

fig = plt.figure(figsize=(15.5, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.06,
                      left=0.03, right=0.985, top=0.855, bottom=0.04)

fig.text(0.03, 0.955, "Phase 3D — Combined MATH-500 + AIME  (ATGI v2)",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.03, 0.923,
         "One thought graph over both datasets, cross-dataset anchors allowed "
         "   ·   Qwen3-1.7B   ·   50.9% → 54.4%  (+3.5pp;  MATH +4.0, "
         "AIME +3.2)",
         fontsize=11, color=INK_2, ha="left")

# ── Panel A: pipeline ────────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 100); axA.axis("off")
axA.set_title("A · Retrieval pipeline", fontsize=13, fontweight="bold",
              color=INK, loc="left", pad=8)

STAGES = [
    ("Test problem  Q", f"{N_MATH} MATH-500  +  {N_AIME} AIME  =  {N}",
     9.0, BOX_FILL, INK_MUTED),
    ("Full-problem embedding", "nomic-embed · 768-d", 9.0, "#ffffff",
     INK_MUTED),
    ("Top-1 anchor over the combined bank",
     "argmax over 400 MATH + 746 AIME = 1,146 vectors\n"
     "+0.05 same-source bias on the argmax · raw sim for the gate",
     14.0, "#ffffff", SAME),
    ("Similarity gate",
     "per-subject (MATH) / 0.50 (AIME) · +0.10 if cross-dataset\n"
     f"{len(SKIP)}/{N} fall below → vanilla prompt instead",
     14.0, "#ffffff", SAME),
    ("Enter the combined graph",
     f"{GMETA['num_nodes']:,} nodes "
     f"({GMETA['num_math500_nodes']:,} MATH + {GMETA['num_aime_nodes']:,} AIME)\n"
     "take the anchor's own 4 step-nodes", 13.0, "#ffffff", SAME),
    ("Novelty-gated bonus step  (≤ 1)",
     f"fired on {N_BONUS}/{N} problems — dead gate, see panel B",
     10.0, "#ffffff", CRIT),
    ("Strategy + worked example", "4 steps, 1 pattern", 9.0, BOX_FILL,
     INK_MUTED),
    ("Qwen3-1.7B", "max 16,384 new tokens", 9.0, BOX_FILL, INK_MUTED),
]

y = 99.0
GAP = 1.6
centers = []
for title, sub, h, fill, edge in STAGES:
    axA.add_patch(FancyBboxPatch(
        (0.6, y - h), 8.8, h, boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=1.6, edgecolor=edge, facecolor=fill, mutation_aspect=0.09))
    if sub:
        axA.text(5.0, y - h * 0.33, title, ha="center", va="center",
                 fontsize=11, fontweight="bold", color=INK)
        axA.text(5.0, y - h * 0.70, sub, ha="center", va="center",
                 fontsize=9.2, color=INK_2, linespacing=1.5)
    else:
        axA.text(5.0, y - h * 0.5, title, ha="center", va="center",
                 fontsize=11, fontweight="bold", color=INK)
    centers.append((y, y - h))
    y = y - h - GAP

for i in range(len(STAGES) - 1):
    axA.add_patch(FancyArrowPatch(
        (5.0, centers[i][1]), (5.0, centers[i + 1][0]), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.3, color=INK_MUTED, shrinkA=0,
        shrinkB=0))

# ── Panel B: where the anchors actually came from ────────────────────────────
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 10); axB.set_ylim(0, 100); axB.axis("off")
axB.set_title("B · Where the anchors came from", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

SRC_X, BANK_X, BW, BH = 0.55, 6.15, 3.1, 12.0
SRC_Y = {"MATH-500": 74.0, "AIME": 52.0}
BANK_Y = {"MATH-500": 74.0, "AIME": 52.0}

for src, yy in SRC_Y.items():
    n_src = N_MATH if src == "MATH-500" else N_AIME
    axB.add_patch(FancyBboxPatch(
        (SRC_X, yy), BW, BH, boxstyle="round,pad=0.05,rounding_size=0.3",
        linewidth=1.6, edgecolor=INK_MUTED, facecolor=BOX_FILL,
        mutation_aspect=0.30))
    axB.text(SRC_X + BW / 2, yy + BH * 0.62, f"{src} test", ha="center",
             va="center", fontsize=11, fontweight="bold", color=INK)
    axB.text(SRC_X + BW / 2, yy + BH * 0.27, f"n = {n_src}", ha="center",
             va="center", fontsize=9.5, color=INK_2)

for bank, yy in BANK_Y.items():
    n_bank = 400 if bank == "MATH-500" else 746
    axB.add_patch(FancyBboxPatch(
        (BANK_X, yy), BW, BH, boxstyle="round,pad=0.05,rounding_size=0.3",
        linewidth=1.6, edgecolor=INK_MUTED, facecolor="#ffffff",
        mutation_aspect=0.30))
    axB.text(BANK_X + BW / 2, yy + BH * 0.62, f"{bank} bank", ha="center",
             va="center", fontsize=11, fontweight="bold", color=INK)
    axB.text(BANK_X + BW / 2, yy + BH * 0.27, f"{n_bank} patterns",
             ha="center", va="center", fontsize=9.5, color=INK_2)

MAXN = max(ROUTE.values())


def flow(src, bank, rad, lx, ly):
    n = ROUTE[(src, bank)]
    same = src == bank
    color = SAME if same else CROSS
    lw = 1.4 + 5.0 * (n / MAXN) ** 0.5
    x0 = SRC_X + BW
    y0 = SRC_Y[src] + BH / 2
    x1 = BANK_X
    y1 = BANK_Y[bank] + BH / 2
    axB.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, alpha=0.9,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3, zorder=2))
    axB.text(lx, ly, str(n), ha="center", va="center", fontsize=11,
             fontweight="bold", color=color, zorder=6,
             bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                       edgecolor="none"))


flow("MATH-500", "MATH-500", 0.0, 4.90, 82.6)
flow("AIME", "AIME", 0.0, 4.90, 55.4)
flow("MATH-500", "AIME", -0.22, 4.25, 73.6)
flow("AIME", "MATH-500", 0.22, 4.25, 64.4)

# skipped
sk = SKIP_BY_SRC["MATH-500"]
axB.add_patch(FancyArrowPatch(
    (SRC_X + BW / 2, SRC_Y["MATH-500"]), (SRC_X + BW / 2, 40.0),
    arrowstyle="-|>", mutation_scale=13, linewidth=2.0, color=INK_MUTED,
    linestyle="--", shrinkA=2, shrinkB=2))
axB.add_patch(FancyBboxPatch(
    (SRC_X - 0.15, 30.5), BW + 0.3, 9.0,
    boxstyle="round,pad=0.05,rounding_size=0.3", linewidth=1.3,
    edgecolor=INK_MUTED, facecolor=BOX_FILL, mutation_aspect=0.30))
axB.text(SRC_X + BW / 2, 36.4, f"below threshold → skip", ha="center",
         va="center", fontsize=9.8, fontweight="bold", color=INK)
axB.text(SRC_X + BW / 2, 32.9, f"{sk} problems — all MATH-500\n"
                               f"0 AIME problems skipped",
         ha="center", va="center", fontsize=9, color=INK_2, linespacing=1.5)

# legend for the flows
axB.plot([0.55, 1.25], [24.0, 24.0], lw=3.2, color=SAME)
n_same = sum(v for (a, b), v in ROUTE.items() if a == b)
n_cross = sum(v for (a, b), v in ROUTE.items() if a != b)
axB.text(1.38, 24.0, f"same-dataset anchor  ({n_same})", fontsize=9.5,
         color=INK_2, va="center")
axB.plot([5.35, 6.05], [24.0, 24.0], lw=2.2, color=CROSS)
axB.text(6.18, 24.0, f"cross-dataset anchor  ({n_cross})", fontsize=9.5,
         color=INK_2, va="center")

# the dead gate
gate = FancyBboxPatch((0.4, 3.0), 9.2, 16.0,
                      boxstyle="round,pad=0.06,rounding_size=0.3",
                      linewidth=1.4, edgecolor=CRIT, facecolor="#ffffff",
                      mutation_aspect=0.30)
axB.add_patch(gate)
axB.text(0.85, 16.0, "THE GATE", fontsize=9.3, fontweight="bold", color=CRIT,
         va="center")
axB.text(0.85, 12.3,
         f"The combined graph holds {GMETA['cross_domain_pairs']:,} "
         f"cross-domain edge pairs (MATH ↔ AIME) — the whole point of "
         f"merging the datasets.",
         fontsize=9.6, color=INK, va="center")
axB.text(0.85, 8.9,
         f"None were reachable: an edge needs sim ≥ τ = "
         f"{GMETA['tau_semantic']:.2f}, the novelty gate rejects sim ≥ 0.85, "
         f"and 0.90 ≥ 0.85 always.",
         fontsize=9.6, color=INK, va="center")
axB.text(0.85, 5.5,
         f"Bonus steps fired {N_BONUS}/{N} — cross-dataset transfer was "
         f"never actually tested.",
         fontsize=9.6, fontweight="bold", color=CRIT, va="center")

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
