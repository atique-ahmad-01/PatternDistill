#!/usr/bin/env python3
"""Phase 3 v2 — anchor-first architecture: pipeline + what the graph contributed.

Panel B is drawn from the actual run: phase-3/results/results.json records
bonus_steps == 0 and n_unique_patterns == 1 for all 100 problems. The
threshold arithmetic that forces that outcome is read off
phase-3/inject_and_eval_atgi.py (NOVELTY_THRESHOLD) and
phase-3/graph/thought_graph.json (tau_semantic).
"""

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "phase3_v2_architecture.png"

RES = json.loads((REPO / "phase-3/results/results.json").read_text())
GMETA = json.loads(
    (REPO / "phase-3/graph/thought_graph.json").read_text())["metadata"]
TAU = GMETA["tau_semantic"]
NOVELTY = 0.85          # phase-3/inject_and_eval_atgi.py:73

N = len(RES)
N_INJ = sum(1 for r in RES if not r["low_similarity"])
N_BONUS = sum(1 for r in RES if r["bonus_steps"] > 0)

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BOX_FILL = "#fcfcfb", "#f1f0ec"
CRIT = "#d03b3b"
PAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
ANCHOR = 0              # row index of the anchor pattern

fig = plt.figure(figsize=(15.5, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.06,
                      left=0.03, right=0.985, top=0.855, bottom=0.04)

fig.text(0.03, 0.955, "Phase 3 v2 — Anchor-First + Novelty Gate",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.03, 0.923,
         "Top-1 problem-level anchor, graph entered at the anchor's own steps "
         "   ·   MATH-500, Qwen3-1.7B   ·   70.0% → 76.0%  (+6.0pp, 10 gains "
         "/ 4 losses)",
         fontsize=11, color=INK_2, ha="left")

# ── Panel A: the pipeline ────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 100); axA.axis("off")
axA.set_title("A · Retrieval pipeline", fontsize=13, fontweight="bold",
              color=INK, loc="left", pad=8)

STAGES = [
    ("Test problem  Q", "", 7.5, BOX_FILL, INK_MUTED),
    ("Full-problem embedding",
     "whole problem text · nomic-embed · 768-d", 11.5, "#ffffff", INK_MUTED),
    ("Top-1 anchor  (phase-2d bank)",
     f"argmax over 400 problem vectors · mean sim 0.65 (v1: ~0.55)\n"
     f"per-subject threshold · {N - N_INJ}/{N} fall below → skip",
     15.0, "#ffffff", PAT_COLORS[0]),
    ("Enter graph at the anchor's steps",
     "all 4 step-nodes of that one pattern", 11.5, "#ffffff", PAT_COLORS[0]),
    ("Novelty-gated bonus step  (≤ 1)",
     f"fired on {N_BONUS}/{N} problems — dead gate, see panel B",
     11.5, "#ffffff", CRIT),
    ("Strategy + worked example",
     f"4 steps, 1 pattern · example on {N_INJ}/{N_INJ}", 11.5, BOX_FILL,
     INK_MUTED),
    ("Qwen3-1.7B", "", 7.5, BOX_FILL, INK_MUTED),
]

y = 98.0
GAP = 3.6
centers = []
for title, sub, h, fill, edge in STAGES:
    axA.add_patch(FancyBboxPatch(
        (0.6, y - h), 8.8, h, boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=1.6, edgecolor=edge, facecolor=fill, mutation_aspect=0.09))
    if sub:
        axA.text(5.0, y - h * 0.34, title, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=INK)
        axA.text(5.0, y - h * 0.71, sub, ha="center", va="center",
                 fontsize=9.5, color=INK_2, linespacing=1.5)
    else:
        axA.text(5.0, y - h * 0.5, title, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=INK)
    centers.append((y, y - h))
    y = y - h - GAP

for i in range(len(STAGES) - 1):
    axA.add_patch(FancyArrowPatch(
        (5.0, centers[i][1]), (5.0, centers[i + 1][0]), arrowstyle="-|>",
        mutation_scale=13, linewidth=1.4, color=INK_MUTED, shrinkA=0,
        shrinkB=0))

# ── Panel B: what the graph contributed ──────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0.1, 6.4); axB.set_ylim(0.05, 5.55); axB.axis("off")
axB.set_title("B · What the graph contributed", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

N_PAT, N_STEP = 4, 4
XS = [1.15, 2.25, 3.35, 4.45]
YS = [4.55, 3.60, 2.65, 1.70]
R = 0.21


def pos(p, s):
    return XS[s], YS[p]


for p in range(N_PAT):
    for s in range(N_STEP - 1):
        x0, y0 = pos(p, s); x1, y1 = pos(p, s + 1)
        axB.add_patch(FancyArrowPatch(
            (x0 + R, y0), (x1 - R, y1), arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=INK_MUTED, alpha=0.55))

# semantic neighbours of the anchor's steps — the rejected candidates
CANDS = [((ANCHOR, 1), (1, 1)), ((ANCHOR, 2), (1, 2)),
         ((ANCHOR, 3), (1, 3)), ((ANCHOR, 0), (2, 1))]
for (p0, s0), (p1, s1) in CANDS:
    x0, y0 = pos(p0, s0); x1, y1 = pos(p1, s1)
    axB.plot([x0, x1], [y0, y1], linestyle=(0, (3, 3)), linewidth=1.3,
             color=CRIT, alpha=0.65, zorder=1)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    axB.text(mx, my, "✗", ha="center", va="center", fontsize=12,
             fontweight="bold", color=CRIT, zorder=5,
             path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])

for p in range(N_PAT):
    label = "anchor" if p == ANCHOR else f"P{p+1}"
    axB.text(XS[0] - 0.72, YS[p], label, ha="center", va="center",
             fontsize=10.5, fontweight="bold",
             color=INK if p == ANCHOR else INK_2)
    for s in range(N_STEP):
        x, y = pos(p, s)
        on = (p == ANCHOR)
        axB.add_patch(Circle((x, y), R,
                             facecolor=PAT_COLORS[p] if on else "#ffffff",
                             edgecolor=INK if on else PAT_COLORS[p],
                             linewidth=2.2 if on else 1.8, zorder=3))
        axB.text(x, y, f"s{s+1}", ha="center", va="center", fontsize=8.4,
                 color="#ffffff" if on else INK_2,
                 fontweight="bold" if on else "normal", zorder=4)

# bracket over the anchor row
bx0, bx1 = XS[0] - R, XS[-1] + R
by = YS[ANCHOR] + R + 0.20
axB.plot([bx0, bx0, bx1, bx1], [by - 0.09, by, by, by - 0.09], lw=1.4,
         color=INK)
axB.text((bx0 + bx1) / 2, by + 0.14,
         "all 4 steps injected as one coherent unit  ·  1 pattern, always",
         ha="center", va="bottom", fontsize=9.5, color=INK)

# the arithmetic that kills the gate
gate = FancyBboxPatch((5.02, 2.32), 1.30, 1.62,
                      boxstyle="round,pad=0.05,rounding_size=0.06",
                      linewidth=1.3, edgecolor=CRIT, facecolor="#ffffff",
                      mutation_aspect=1.0)
axB.add_patch(gate)
axB.text(5.67, 3.79, "THE GATE", ha="center", va="center", fontsize=8.6,
         fontweight="bold", color=CRIT)
axB.text(5.67, 3.46, f"edge exists\n⟺ sim ≥ {TAU:.2f}", ha="center",
         va="center", fontsize=9, color=INK, linespacing=1.5)
axB.text(5.67, 3.02, f"gate rejects\n⟺ sim ≥ {NOVELTY:.2f}", ha="center",
         va="center", fontsize=9, color=INK, linespacing=1.5)
axB.plot([5.14, 6.20], [2.76, 2.76], lw=1.0, color=INK_MUTED)
axB.text(5.67, 2.55, f"{TAU:.2f} ≥ {NOVELTY:.2f}\nfor every candidate",
         ha="center", va="center", fontsize=9, fontweight="bold", color=CRIT,
         linespacing=1.5)

# legend
axB.plot([1.15, 1.48], [1.12, 1.12], linewidth=1.0, color=INK_MUTED,
         alpha=0.6)
axB.text(1.56, 1.12, "sequential edge", fontsize=9, color=INK_2, va="center")
axB.plot([3.05, 3.38], [1.12, 1.12], linestyle=(0, (3, 3)), linewidth=1.3,
         color=CRIT, alpha=0.65)
axB.text(3.46, 1.12, "semantic edge → candidate rejected by the gate",
         fontsize=9, color=INK_2, va="center")

axB.text(1.15, 0.80, f"bonus steps fired: {N_BONUS} / {N}", fontsize=10.5,
         fontweight="bold", color=CRIT, va="center")
axB.text(1.15, 0.58,
         "Every injection is the anchor pattern verbatim plus its worked\n"
         "example. The graph only re-labels and renumbers those steps —\n"
         "on MATH-500, v2's +6.0pp owes nothing to graph traversal.",
         fontsize=8.8, color=INK_2, va="top", linespacing=1.5)

# graph stats
stats = FancyBboxPatch((4.35, 0.22), 1.95, 0.62,
                       boxstyle="round,pad=0.03,rounding_size=0.05",
                       linewidth=1.2, edgecolor=INK_MUTED,
                       facecolor=BOX_FILL, mutation_aspect=1.0)
axB.add_patch(stats)
axB.text(4.47, 0.71, f"v2 graph  (τ = {TAU:.2f})", fontsize=9.5,
         fontweight="bold", color=INK, va="center")
axB.text(4.47, 0.52,
         f"{GMETA['num_nodes']:,} nodes · {GMETA['num_semantic']:,} semantic",
         fontsize=8.8, color=INK_2, va="center")
axB.text(4.47, 0.36,
         f"avg degree {GMETA['avg_degree']:.1f} · max {GMETA['max_degree']}",
         fontsize=8.8, color=INK_2, va="center")

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
