#!/usr/bin/env python3
"""Phase 3 v1 (RoT-style traversal) — accuracy and latency vs vanilla.

v1's per-problem results file was overwritten when v2 was run in the same
directory, so these numbers are parsed back out of the v1 run log:
    phase-3/logs/inject_and_eval_atgi_20260725_223938.log
(28,312 semantic edges = the v1 graph; the 19:20 log is an earlier v1 run of
the same config that injected on only 32/100 and is not the one the README
cites.)

The v1 script logged latency but NOT token counts, so there is no token
column here. Latency is integer-second resolution, as logged.
"""

import pathlib
import re
from collections import OrderedDict, Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

REPO = pathlib.Path(__file__).resolve().parents[2]
LOG = REPO / "phase-3/logs/inject_and_eval_atgi_20260725_223938.log"
OUT = pathlib.Path(__file__).parent / "phase3_v1_results_table.png"

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BAND = "#fcfcfb", "#f4f3ef"
GOOD, CRIT = "#0ca30c", "#d03b3b"

LINE = re.compile(
    r"\[(\d+)/100\] id=(\d+) (.+?) L(\d) \| sim=([\d.]+) thr=([\d.]+) \| "
    r"trav=(\d+)steps (\d+)pat(.) \| van=(.)\((\d+)s\) atgi=(.)\((\d+)s\)")

recs = OrderedDict()
for line in LOG.read_text().splitlines():
    m = LINE.search(line)
    if not m:
        continue
    _, pid, subj, lvl, sim, thr, steps, npat, flag, vok, vlat, aok, alat = \
        m.groups()
    recs[int(pid)] = dict(pid=int(pid), subject=subj.strip(), level=int(lvl),
                          steps=int(steps), npat=int(npat),
                          van=vok == "✓", vlat=int(vlat),
                          atgi=aok == "✓", alat=int(alat))
R = list(recs.values())


def agg(group):
    n = len(group)
    va = sum(x["van"] for x in group) / n * 100
    ia = sum(x["atgi"] for x in group) / n * 100
    vl = sum(x["vlat"] for x in group) / n
    al = sum(x["alat"] for x in group) / n
    return dict(n=n, va=va, ia=ia, d_acc=ia - va, vl=vl, al=al,
                d_lat=(al - vl) / vl * 100)


SUBJECTS = ["Algebra", "Counting & Probability", "Geometry",
            "Intermediate Algebra", "Number Theory", "Prealgebra",
            "Precalculus"]

TABLE = [("section", "Overall", None),
         ("row", "All problems", agg(R)),
         ("row", "Injected (traversal fired)",
          agg([x for x in R if x["steps"] > 0])),
         ("row", "Skipped (below threshold)",
          agg([x for x in R if x["steps"] == 0])),
         ("section", "By subject", None)]
TABLE += [("row", s, agg([x for x in R if x["subject"] == s]))
          for s in SUBJECTS]
TABLE += [("section", "By difficulty", None)]
TABLE += [("row", f"Level {L}", agg([x for x in R if x["level"] == L]))
          for L in sorted({x["level"] for x in R})]

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.8, 13.2), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")

fig.text(0.028, 0.962, "Phase 3 v1 — RoT-style traversal vs vanilla",
         fontsize=18, fontweight="bold", color=INK)
fig.text(0.028, 0.936,
         "MATH-500 test (100 problems) · Qwen3-1.7B · RTX 5070 Ti",
         fontsize=10.5, color=INK_2)

C = dict(label=3.0, n=38.0, va=48.0, ia=56.5, dacc=65.0,
         vl=76.0, il=84.5, dlat=93.5)
GROUPS = [("Accuracy  %", 41.5, 65.5), ("Mean latency  s", 69.5, 94.0)]

TOP = 88.0
RH = 3.35

for label, x0, x1 in GROUPS:
    ax.add_patch(Rectangle((x0, TOP + 4.6), x1 - x0, 0.28, color=INK_MUTED,
                           alpha=0.55, lw=0))
    ax.text((x0 + x1) / 2, TOP + 5.5, label, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK)

for key, lbl in [("n", "n"), ("va", "vanilla"), ("ia", "v1"),
                 ("dacc", "Δ pp"), ("vl", "vanilla"), ("il", "v1"),
                 ("dlat", "Δ %")]:
    ax.text(C[key], TOP + 1.8, lbl, ha="right", va="center", fontsize=10,
            fontweight="bold", color=INK_2)
ax.plot([2.4, 94.0], [TOP + 0.4, TOP + 0.4], lw=1.3, color=INK_MUTED)


def delta(x, y, value, unit, good_is_negative, bold=True):
    if abs(value) < 0.05:
        ax.text(x, y, f"=  {value:+.1f}{unit}", ha="right", va="center",
                fontsize=10.5, color=INK_2,
                fontweight="bold" if bold else "normal")
        return
    up = value > 0
    beneficial = (not up) if good_is_negative else up
    ax.text(x, y, f"{'▲' if up else '▼'} {value:+.1f}{unit}", ha="right",
            va="center", fontsize=10.5, color=GOOD if beneficial else CRIT,
            fontweight="bold" if bold else "normal")


y = TOP - 1.4
band = 0
for kind, label, a in TABLE:
    if kind == "section":
        y -= 2.0
        ax.text(C["label"], y, label.upper(), ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=INK_MUTED)
        y -= RH * 0.75
        band = 0
        continue
    if band % 2 == 0:
        ax.add_patch(FancyBboxPatch((2.4, y - RH / 2 + 0.35), 91.6, RH - 0.7,
                                    boxstyle="round,pad=0,rounding_size=0.3",
                                    facecolor=BAND, edgecolor="none",
                                    mutation_aspect=0.28, zorder=0))
    band += 1
    head = label.startswith("All")
    ax.text(C["label"], y, label, ha="left", va="center", fontsize=10.5,
            fontweight="bold" if head else "normal", color=INK)
    ax.text(C["n"], y, str(a["n"]), ha="right", va="center", fontsize=10.5,
            color=INK_2)
    ax.text(C["va"], y, f"{a['va']:.1f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["ia"], y, f"{a['ia']:.1f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(C["dacc"], y, a["d_acc"], "", good_is_negative=False)
    ax.text(C["vl"], y, f"{a['vl']:.1f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["il"], y, f"{a['al']:.1f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(C["dlat"], y, a["d_lat"], "%", good_is_negative=True)
    y -= RH

ax.plot([2.4, 94.0], [y + RH - 1.6] * 2, lw=1.3, color=INK_MUTED)

# ── side notes ───────────────────────────────────────────────────────────────
inj = [x for x in R if x["steps"] > 0]
capped = sum(1 for x in inj if x["steps"] == 6)
multi = sum(1 for x in inj if x["npat"] > 1)
losses = [x for x in R if x["van"] and not x["atgi"]]
wins = [x for x in R if x["atgi"] and not x["van"]]

box_y = y + RH - 5.4
ax.add_patch(FancyBboxPatch((2.4, box_y - 11.4), 91.6, 10.6,
                            boxstyle="round,pad=0.15,rounding_size=0.4",
                            facecolor=BAND, edgecolor=INK_MUTED, lw=1.1,
                            mutation_aspect=0.30))
ax.text(4.2, box_y - 1.6, "TRAVERSAL", fontsize=9.5, fontweight="bold",
        color=INK_MUTED, va="center")
ax.text(4.2, box_y - 4.2,
        f"{len(inj)}/100 injected · mean {sum(x['steps'] for x in inj)/len(inj):.2f} "
        f"steps from {sum(x['npat'] for x in inj)/len(inj):.1f} patterns · "
        f"{capped}/{len(inj)} hit the 6-step cap · {multi}/{len(inj)} drew on "
        f"2+ patterns",
        fontsize=10, color=INK, va="center")
ax.text(4.2, box_y - 7.0,
        f"{len(wins)} gains / {len(losses)} losses.  All {len(losses)} losses "
        f"were multi-pattern assemblies — but so were {multi} of {len(inj)} "
        f"injections, and multi-pattern",
        fontsize=10, color=INK, va="center")
ax.text(4.2, box_y - 9.3,
        "injections were net positive (+3.4pp). The “Frankenstein template” "
        "story is directional, not established.",
        fontsize=10, color=INK, va="center")

FOOT = box_y - 13.6
ax.text(2.4, FOOT,
        "Mean tokens: not recorded — the v1 script logged latency only. "
        "Latency is integer-second resolution, as logged.",
        ha="left", va="top", fontsize=9.3, color=INK_2)
ax.text(2.4, FOOT - 2.4,
        "Recovered from logs/inject_and_eval_atgi_20260725_223938.log; v1's "
        "results.json was overwritten by the v2 run. Pre-gate-v2 pattern bank.",
        ha="left", va="top", fontsize=9.3, color=INK_2)
ax.text(2.4, FOOT - 4.8,
        "▲ / ▼ mark direction; green = the desirable direction (higher "
        "accuracy, lower latency).",
        ha="left", va="top", fontsize=9.3, color=INK_2)

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
