#!/usr/bin/env python3
"""RB-DERIV — teacher distillation + student evaluation architecture.

PLANNED pipeline: no run has happened yet, so this figure carries no accuracy
or token numbers. Every count shown is derived from the shipped data
(the single RB_DERIV copy) or from the split policy in teacher-prompt.yml.

The benchmark data lives in rot-multi-solution-approach-graph/data/RB_DERIV/ —
the one canonical copy; this figure reaches across to read it.
"""

import json
import pathlib
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

# The dataset now lives in exactly one place: the active RoT project. This
# archived figure reaches across to that single copy rather than keeping its
# own, so there is nothing to drift.
REPO = pathlib.Path(__file__).resolve().parents[3]
DATA = REPO / "rot-multi-solution-approach-graph" / "data" / "RB_DERIV"
OUT = pathlib.Path(__file__).parent / "rbderiv_architecture.png"

Q = json.load(open(DATA / "RB_DERIV_questions.json"))["questions"]
O = json.load(open(DATA / "RB_DERIV_objects.json"))["objects"]

N_Q = len(Q)
FAM = Counter(x["family"] for x in Q)
N_FAM = len(FAM)
PER_FAM = FAM.most_common(1)[0][1]
TEST_PER_FAM = 3
TRAIN_PER_FAM = PER_FAM - TEST_PER_FAM
N_TRAIN = N_FAM * TRAIN_PER_FAM
N_TEST = N_FAM * TEST_PER_FAM
N_FF = sum(1 for x in Q if x["false_friend_changes_answer"])
N_SAMPLES = 5
N_SOLVE_CALLS = N_TRAIN * N_SAMPLES

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BOX_FILL = "#fcfcfb", "#f1f0ec"
CRIT = "#d03b3b"
TEACH, STUDENT = "#2a78d6", "#eb6834"
ORACLE = "#6b8f3a"

fig = plt.figure(figsize=(16.0, 9.6), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.05,
                      left=0.03, right=0.985, top=0.855, bottom=0.035)

fig.text(0.03, 0.955, "RB-DERIV — teacher-distilled memory, student reuse",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.03, 0.922,
         f"Qwen3-8B teacher  →  {N_FAM} distilled technique objects  →  "
         f"Qwen3-1.7B student   ·   {N_Q} questions, {N_FAM} families × "
         f"{PER_FAM}   ·   per-family {TRAIN_PER_FAM} train / {TEST_PER_FAM} "
         "test   ·   planned pipeline, no results yet",
         fontsize=10.5, color=INK_2, ha="left")

# ── Panel A: pipeline ────────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(-7, 100); axA.axis("off")
axA.set_title("A · Pipeline", fontsize=13, fontweight="bold", color=INK,
              loc="left", pad=8)

STAGES = [
    ("RB-DERIV", f"{N_Q} questions · {N_FAM} families × {PER_FAM} · "
     "exact integer golds", 8.0, BOX_FILL, INK_MUTED),
    ("Per-family split",
     f"{TRAIN_PER_FAM} train / {TEST_PER_FAM} test in every family  →  "
     f"{N_TRAIN} train, {N_TEST} test\nsplit within family, never globally — "
     "all 16 techniques must appear in test", 10.0, "#ffffff", INK_MUTED),
    ("TEACHER · solve      Qwen3-8B",
     f"{N_SAMPLES} independent samples per train problem @ T=0.8  →  "
     f"{N_SOLVE_CALLS} calls\nSTRATEGY + steps + ANSWER · gold is NOT in the "
     "prompt", 11.0, "#ffffff", TEACH),
    ("Exact-match filter against gold",
     "keep the sample only if ANSWER == gold_answer\n"
     "GO / NO-GO:  a family with < 3 verified solutions is teacher-limited —\n"
     "do not distill it, the object would encode the teacher's error",
     12.5, "#ffffff", CRIT),
    ("TEACHER · distill      Qwen3-8B",
     "one call per family over its verified solutions  →  "
     f"{N_FAM} calls\nNAME / USE_WHEN / METHOD / CHECK\n"
     "9 different instances per family force abstraction",
     12.0, "#ffffff", TEACH),
    ("Gates  (≤ 2 retries)",
     "abstraction · no-leak · USE_WHEN shape · CHECK decisive",
     8.0, "#ffffff", TEACH),
    ("MEMORY", f"{N_FAM} distilled technique objects, one per family",
     7.5, BOX_FILL, INK_MUTED),
    ("STUDENT · solve      Qwen3-1.7B",
     "arm-conditioned prompt, memory injected · single-integer contract\n"
     "shared system prompt byte-identical across arms", 10.5, "#ffffff",
     STUDENT),
    ("Score",
     f"exact integer match on the {N_TEST} held-out questions\n"
     "accuracy · decode tokens · CHECK verdict",
     9.0, BOX_FILL, INK_MUTED),
]

y, GAP = 98.0, 1.25
centers = []
for title, sub, h, fill, edge in STAGES:
    axA.add_patch(FancyBboxPatch(
        (0.6, y - h), 8.8, h, boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=1.7, edgecolor=edge, facecolor=fill, mutation_aspect=0.09))
    axA.text(5.0, y - h * 0.30, title, ha="center", va="center",
             fontsize=11, fontweight="bold", color=INK)
    axA.text(5.0, y - h * 0.68, sub, ha="center", va="center",
             fontsize=8.9, color=INK_2, linespacing=1.45)
    centers.append((y, y - h))
    y = y - h - GAP

for i in range(len(STAGES) - 1):
    axA.add_patch(FancyArrowPatch(
        (5.0, centers[i][1]), (5.0, centers[i + 1][0]), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.3, color=INK_MUTED,
        shrinkA=0, shrinkB=0))

# ── Panel B: what the student is shown ───────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 10); axB.set_ylim(-7, 100); axB.axis("off")
axB.set_title("B · What the student is shown, per arm", fontsize=13,
              fontweight="bold", color=INK, loc="left", pad=8)

ARMS = [
    ("base", "nothing — the student must derive the technique itself",
     "reference floor", INK_MUTED, BOX_FILL),
    ("plain", "METHOD only, as a bare sentence. no gate, no verification",
     "isolates STRUCTURE from CONTENT", INK_MUTED, "#ffffff"),
    ("schema", "NAME · USE_WHEN · METHOD · CHECK   — distilled by the teacher",
     "the claim", STUDENT, "#ffffff"),
    ("schema_oracle",
     "same four fields — the 16 human-authored objects shipped with RB-DERIV",
     "quality ceiling: how much of the gap did distillation close?",
     ORACLE, "#ffffff"),
    ("schema_wrong",
     "METHOD corrupted, CHECK honest — the check would catch it if run",
     f"rescue scored only on the {N_FF}/{N_Q} instances the corruption "
     "actually changes", CRIT, "#ffffff"),
    ("schema_random",
     "a correct object from a DIFFERENT family — USE_WHEN will not match",
     "placebo: required before any stratified saving claim", INK_MUTED,
     "#ffffff"),
]

yb, GAPB, HB = 95.0, 2.2, 13.0
for name, shown, why, accent, fill in ARMS:
    axB.add_patch(FancyBboxPatch(
        (0.9, yb - HB), 8.7, HB,
        boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=1.7, edgecolor=accent, facecolor=fill,
        mutation_aspect=0.105))
    axB.add_patch(plt.Rectangle((0.9, yb - HB), 0.13, HB, color=accent,
                                zorder=3))
    axB.text(1.35, yb - HB * 0.26, name, ha="left", va="center", fontsize=11,
             fontweight="bold", color=INK, family="monospace")
    axB.text(1.35, yb - HB * 0.56, shown, ha="left", va="center",
             fontsize=8.9, color=INK_2)
    axB.text(1.35, yb - HB * 0.83, why, ha="left", va="center", fontsize=8.6,
             color=accent, style="italic")
    yb = yb - HB - GAPB

axB.text(0.9, yb - 1.2,
         "All five schema-form templates are byte-identical; only the injected\n"
         "content differs, so a token or accuracy gap between arms cannot be\n"
         "attributed to prompt wording.",
         ha="left", va="top", fontsize=8.8, color=INK_2, linespacing=1.6)

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
print(f"train={N_TRAIN} test={N_TEST} solve_calls={N_SOLVE_CALLS} ff={N_FF}")
