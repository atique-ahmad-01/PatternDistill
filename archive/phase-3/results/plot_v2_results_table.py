#!/usr/bin/env python3
"""Phase 3 v2 (anchor-first) — accuracy, tokens and latency vs vanilla.

Unlike v1, v2's per-problem results survive, so every column here is read
straight from phase-3/results/results.json — no log parsing, and tokens are
available.
"""

import json
import pathlib
import statistics as st

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "phase-3/results/results.json"
OUT = pathlib.Path(__file__).parent / "phase3_v2_results_table.png"

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BAND = "#fcfcfb", "#f4f3ef"
GOOD, CRIT = "#0ca30c", "#d03b3b"

R = json.loads(SRC.read_text())


def agg(g):
    n = len(g)
    return dict(
        n=n,
        va=sum(x["vanilla_correct"] for x in g) / n * 100,
        ia=sum(x["injection_correct"] for x in g) / n * 100,
        vt=st.mean(x["vanilla_tokens"] for x in g),
        it=st.mean(x["injection_tokens"] for x in g),
        vl=st.mean(x["vanilla_latency_s"] for x in g),
        il=st.mean(x["injection_latency_s"] for x in g))


def deltas(a):
    a["d_acc"] = a["ia"] - a["va"]
    a["d_tok"] = (a["it"] - a["vt"]) / a["vt"] * 100
    a["d_lat"] = (a["il"] - a["vl"]) / a["vl"] * 100
    return a


SUBJECTS = ["Algebra", "Counting & Probability", "Geometry",
            "Intermediate Algebra", "Number Theory", "Prealgebra",
            "Precalculus"]

TABLE = [("section", "Overall", None),
         ("row", "All problems", deltas(agg(R))),
         ("row", "Injected (above threshold)",
          deltas(agg([x for x in R if not x["low_similarity"]]))),
         ("row", "Skipped (below threshold)",
          deltas(agg([x for x in R if x["low_similarity"]]))),
         ("section", "By subject", None)]
TABLE += [("row", s, deltas(agg([x for x in R if x["subject"] == s])))
          for s in SUBJECTS]
TABLE += [("section", "By difficulty", None)]
TABLE += [("row", f"Level {L}",
           deltas(agg([x for x in R if x["level"] == L])))
          for L in sorted({x["level"] for x in R}, key=int)]

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.2, 13.2), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 112); ax.set_ylim(0, 100)
ax.axis("off")

fig.text(0.024, 0.962, "Phase 3 v2 — Anchor-first injection vs vanilla",
         fontsize=18, fontweight="bold", color=INK)
fig.text(0.024, 0.936,
         "MATH-500 test (100 problems) · Qwen3-1.7B · RTX 5070 Ti",
         fontsize=10.5, color=INK_2)

C = dict(label=3.0, n=34.0, va=43.0, ia=51.0, dacc=59.5,
         vt=70.0, it=78.0, dtok=86.5,
         vl=97.0, il=104.0)
DLAT_X = 111.6
GROUPS = [("Accuracy  %", 36.5, 60.0), ("Mean tokens", 63.0, 87.0),
          ("Mean latency  s", 90.0, 111.6)]

TOP = 88.0
RH = 3.35

for label, x0, x1 in GROUPS:
    ax.add_patch(Rectangle((x0, TOP + 4.6), x1 - x0, 0.28, color=INK_MUTED,
                           alpha=0.55, lw=0))
    ax.text((x0 + x1) / 2, TOP + 5.5, label, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK)

for key, lbl in [("n", "n"), ("va", "vanilla"), ("ia", "v2"),
                 ("dacc", "Δ pp"), ("vt", "vanilla"), ("it", "v2"),
                 ("dtok", "Δ %"), ("vl", "vanilla"), ("il", "v2")]:
    ax.text(C[key], TOP + 1.8, lbl, ha="right", va="center", fontsize=10,
            fontweight="bold", color=INK_2)
ax.text(DLAT_X, TOP + 1.8, "Δ %", ha="right", va="center", fontsize=10,
        fontweight="bold", color=INK_2)
ax.plot([2.4, 111.6], [TOP + 0.4, TOP + 0.4], lw=1.3, color=INK_MUTED)


def delta(x, y, value, unit, good_is_negative):
    if abs(value) < 0.05:
        ax.text(x, y, f"=  {value:+.1f}{unit}", ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=INK_2)
        return
    up = value > 0
    beneficial = (not up) if good_is_negative else up
    ax.text(x, y, f"{'▲' if up else '▼'} {value:+.1f}{unit}", ha="right",
            va="center", fontsize=10.5, fontweight="bold",
            color=GOOD if beneficial else CRIT)


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
        ax.add_patch(FancyBboxPatch((2.4, y - RH / 2 + 0.35), 109.2, RH - 0.7,
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
    ax.text(C["vt"], y, f"{a['vt']:,.0f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["it"], y, f"{a['it']:,.0f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(C["dtok"], y, a["d_tok"], "%", good_is_negative=True)
    ax.text(C["vl"], y, f"{a['vl']:.1f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["il"], y, f"{a['il']:.1f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(DLAT_X, y, a["d_lat"], "%", good_is_negative=True)
    y -= RH

ax.plot([2.4, 111.6], [y + RH - 1.6] * 2, lw=1.3, color=INK_MUTED)

# ── notes ────────────────────────────────────────────────────────────────────
n_inj = sum(1 for x in R if not x["low_similarity"])
n_bonus = sum(1 for x in R if x["bonus_steps"] > 0)
wins = sum(1 for x in R if x["injection_correct"] and not x["vanilla_correct"])
losses = sum(1 for x in R if x["vanilla_correct"] and not x["injection_correct"])

box_y = y + RH - 5.4
ax.add_patch(FancyBboxPatch((2.4, box_y - 11.4), 109.2, 10.6,
                            boxstyle="round,pad=0.15,rounding_size=0.4",
                            facecolor=BAND, edgecolor=INK_MUTED, lw=1.1,
                            mutation_aspect=0.30))
ax.text(4.2, box_y - 1.6, "RETRIEVAL", fontsize=9.5, fontweight="bold",
        color=INK_MUTED, va="center")
ax.text(4.2, box_y - 4.2,
        f"{n_inj}/100 injected · every injection is exactly 4 steps from "
        f"1 pattern plus its worked example · bonus steps fired {n_bonus}/100",
        fontsize=10, color=INK, va="center")
ax.text(4.2, box_y - 7.0,
        f"{wins} gains / {losses} losses. The skipped rows are identical by "
        f"construction — below threshold, v2 sends the vanilla prompt, so "
        f"those 19 problems",
        fontsize=10, color=INK, va="center")
ax.text(4.2, box_y - 9.3,
        "contribute +0.0pp and 0% by definition; the real effect lives in the "
        "81 injected rows (+7.4pp, −6.6% tokens).",
        fontsize=10, color=INK, va="center")

FOOT = box_y - 13.6
ax.text(2.4, FOOT,
        "Source: phase-3/results/results.json (the v2 run — v1's results were "
        "overwritten by it). Pre-gate-v2 pattern bank.",
        ha="left", va="top", fontsize=9.3, color=INK_2)
ax.text(2.4, FOOT - 2.4,
        "▲ / ▼ mark direction; green = the desirable direction (higher "
        "accuracy, fewer tokens, lower latency).",
        ha="left", va="top", fontsize=9.3, color=INK_2)

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
