#!/usr/bin/env python3
"""Phase 3 v1 — RoT-style architecture: retrieval pipeline + traversal schematic.

The v1 graph artifacts were overwritten by v2 (phase-3/graph/ is
v2-context-enriched), so panel B is a schematic of the documented v1
traversal, not a plot of the real 1,600-node graph. Numbers in the stats
box come from phase-3/Readme.md:229.
"""

import pathlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.patheffects as pe

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

OUT = pathlib.Path(__file__).parent / "phase3_v1_architecture.png"

# ── ink + identity tokens ────────────────────────────────────────────────────
INK        = "#0b0b0b"   # text-primary
INK_2      = "#52514e"   # text-secondary
INK_MUTED  = "#8a8880"   # recessive rules / sequential edges
SURFACE    = "#fcfcfb"
BOX_FILL   = "#f1f0ec"
# categorical slots 1,2,3,7 — validated all-pairs, light mode
PAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

fig = plt.figure(figsize=(15.5, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.06,
                      left=0.03, right=0.985, top=0.855, bottom=0.04)

fig.text(0.03, 0.955, "Phase 3 v1 — Pure Graph Traversal (RoT-style)",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.03, 0.923,
         "Step-level entry + greedy reward-guided traversal   ·   MATH-500, "
         "Qwen3-1.7B   ·   70.0% → 73.0%  (+3.0pp, 6 gains / 3 losses)",
         fontsize=11, color=INK_2, ha="left")

# ── Panel A: the pipeline ────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 100); axA.axis("off")
axA.set_title("A · Retrieval pipeline", fontsize=13, fontweight="bold",
              color=INK, loc="left", pad=8)

STAGES = [
    ("Test problem  Q", "", 7.5, BOX_FILL, INK_MUTED),
    ("Step-level embedding",
     "step_text only · nomic-embed · 768-d", 11.5, "#ffffff", INK_MUTED),
    ("Subject-filtered initial node",
     r"R = 0.8·sim + 0.2·is_step0", 11.5, "#ffffff", PAT_COLORS[0]),
    ("Greedy reward-guided traversal",
     "R = 0.6·sim + 0.4·flow\nsequential + semantic edges · max 6 steps",
     15.0, "#ffffff", PAT_COLORS[0]),
    ("Assembled template",
     "5–6 steps drawn from multiple patterns", 11.5, "#ffffff", PAT_COLORS[1]),
    ("+ worked example",
     "top-1 from the phase-2d bank", 9.5, BOX_FILL, INK_MUTED),
    ("Qwen3-1.7B", "", 7.5, BOX_FILL, INK_MUTED),
]

y = 98.0
GAP = 3.6
centers = []
for title, sub, h, fill, edge in STAGES:
    box = FancyBboxPatch((0.6, y - h), 8.8, h,
                         boxstyle="round,pad=0.02,rounding_size=0.35",
                         linewidth=1.6, edgecolor=edge, facecolor=fill,
                         mutation_aspect=0.09)
    axA.add_patch(box)
    if sub:
        axA.text(5.0, y - h * 0.36, title, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=INK)
        axA.text(5.0, y - h * 0.72, sub, ha="center", va="center",
                 fontsize=9.5, color=INK_2, linespacing=1.5)
    else:
        axA.text(5.0, y - h * 0.5, title, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=INK)
    centers.append((y, y - h))
    y = y - h - GAP

for i in range(len(STAGES) - 1):
    axA.add_patch(FancyArrowPatch(
        (5.0, centers[i][1]), (5.0, centers[i + 1][0]),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
        color=INK_MUTED, shrinkA=0, shrinkB=0))

# the two v1-specific failure notes
axA.text(9.55, (centers[2][0] + centers[3][1]) / 2,
         "no problem-level\nanchor — v1 enters\nthe graph directly",
         ha="left", va="center", fontsize=9, color=PAT_COLORS[0],
         linespacing=1.5, style="italic")
axA.text(9.55, (centers[4][0] + centers[4][1]) / 2,
         "always padded\nto 6 steps",
         ha="left", va="center", fontsize=9, color=PAT_COLORS[1],
         linespacing=1.5, style="italic")

# ── Panel B: traversal over the thought graph ────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0.1, 6.4); axB.set_ylim(0.05, 5.55); axB.axis("off")
axB.set_title("B · What the traversal produced", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

N_PAT, N_STEP = 4, 5
XS = [0.95, 1.95, 2.95, 3.95, 4.95]
YS = [4.55, 3.60, 2.65, 1.70]
R = 0.20

def pos(p, s):
    return XS[s], YS[p]

# recessive sequential edges within each pattern
for p in range(N_PAT):
    for s in range(N_STEP - 1):
        x0, y0 = pos(p, s); x1, y1 = pos(p, s + 1)
        axB.add_patch(FancyArrowPatch(
            (x0 + R, y0), (x1 - R, y1), arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=INK_MUTED, alpha=0.55))

# a sample of semantic (cross-pattern) edges
SEM = [((1, 1), (0, 2)), ((0, 3), (2, 4)), ((2, 4), (3, 4)),   # path hops
       ((1, 1), (3, 1)), ((3, 2), (0, 2)), ((0, 0), (2, 0)),
       ((2, 1), (3, 0)), ((1, 3), (3, 3)), ((1, 2), (0, 3)),
       ((2, 2), (3, 4))]
for (p0, s0), (p1, s1) in SEM:
    x0, y0 = pos(p0, s0); x1, y1 = pos(p1, s1)
    axB.plot([x0, x1], [y0, y1], linestyle=(0, (3, 3)), linewidth=0.9,
             color=INK_MUTED, alpha=0.5, zorder=1)

# nodes
for p in range(N_PAT):
    axB.text(XS[0] - 0.62, YS[p], f"P{p+1}", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=INK)
    for s in range(N_STEP):
        x, y = pos(p, s)
        axB.add_patch(Circle((x, y), R, facecolor="#ffffff",
                             edgecolor=PAT_COLORS[p], linewidth=1.8, zorder=3))
        axB.text(x, y, f"s{s+1}", ha="center", va="center", fontsize=8.2,
                 color=INK_2, zorder=4)

# the greedy path: 6 steps, 4 different patterns
PATH = [(1, 0), (1, 1), (0, 2), (0, 3), (2, 4), (3, 4)]
for i in range(len(PATH) - 1):
    x0, y0 = pos(*PATH[i]); x1, y1 = pos(*PATH[i + 1])
    axB.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
        linewidth=2.6, color=INK, shrinkA=R * 72, shrinkB=R * 72 + 3,
        zorder=5, path_effects=[pe.withStroke(linewidth=5.5,
                                              foreground=SURFACE)]))
for i, (p, s) in enumerate(PATH):
    x, y = pos(p, s)
    axB.add_patch(Circle((x, y), R, facecolor=PAT_COLORS[p],
                         edgecolor=INK, linewidth=1.8, zorder=6))
    axB.text(x, y, f"s{s+1}", ha="center", va="center", fontsize=8.2,
             color="#ffffff", fontweight="bold", zorder=7)
    # badge sits above the node, unless the incoming arrow is vertical
    if i > 0 and PATH[i - 1][1] == s:
        bx, by = x + R + 0.19, y
    else:
        bx, by = x, y + R + 0.17
    axB.text(bx, by, str(i + 1), ha="center", va="center",
             fontsize=9.5, fontweight="bold", color=INK, zorder=7)

# entry marker
ex, ey = pos(*PATH[0])
axB.annotate("entry node\n(subject-filtered, is_step0 bonus)",
             xy=(ex - R * 0.8, ey + R * 0.6), xytext=(0.62, 5.22),
             ha="left", va="center", fontsize=8.8, color=INK_2,
             linespacing=1.5,
             arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color=INK_2,
                             connectionstyle="arc3,rad=0.2",
                             shrinkA=2, shrinkB=3))

# legend — identity is never color-alone; every node carries its step label
axB.plot([0.95, 1.28], [1.30, 1.30], linewidth=1.0, color=INK_MUTED,
         alpha=0.6)
axB.text(1.36, 1.30, "sequential edge (within a pattern)", fontsize=9,
         color=INK_2, va="center")
axB.plot([3.35, 3.68], [1.30, 1.30], linestyle=(0, (3, 3)), linewidth=0.9,
         color=INK_MUTED, alpha=0.6)
axB.text(3.76, 1.30, "semantic edge (τ = 0.90)", fontsize=9, color=INK_2,
         va="center")
axB.add_patch(FancyArrowPatch((0.95, 1.02), (1.28, 1.02), arrowstyle="-|>",
                              mutation_scale=13, linewidth=2.6, color=INK))
axB.text(1.36, 1.02, "greedy traversal path  (numbered 1–6)", fontsize=9,
         color=INK_2, va="center")

# the finding
axB.text(0.95, 0.70, "6 steps · 4 different patterns", fontsize=10.5,
         fontweight="bold", color=PAT_COLORS[1], va="center")
axB.text(0.95, 0.50,
         "“Frankenstein template” — all 3 losses\ncame from cross-pattern "
         "assemblies",
         fontsize=8.8, color=INK_2, va="top", linespacing=1.5)

# graph stats
stats = FancyBboxPatch((3.35, 0.30), 2.95, 0.62,
                       boxstyle="round,pad=0.03,rounding_size=0.05",
                       linewidth=1.2, edgecolor=INK_MUTED,
                       facecolor=BOX_FILL, mutation_aspect=1.0)
axB.add_patch(stats)
axB.text(3.50, 0.79, "v1 graph  (τ = 0.90)", fontsize=9.5,
         fontweight="bold", color=INK, va="center")
axB.text(3.50, 0.58, "1,600 nodes · 1,200 sequential · 28,312 semantic",
         fontsize=8.8, color=INK_2, va="center")
axB.text(3.50, 0.42, "avg degree 18.4 · max degree 157",
         fontsize=8.8, color=INK_2, va="center")

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
