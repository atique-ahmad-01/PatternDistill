#!/usr/bin/env python3
"""Cross-phase results table — accuracy, tokens, latency, injection vs vanilla.

Every row is a paired within-run comparison (same problems, same model, same
decode settings, vanilla vs injected), so the DELTA column is comparable
across rows. The absolute vanilla column is NOT: the partial runs sample a
different MATH/AIME mix (see the composition column).

Sources are each phase's own results file; nothing is re-run here.
"""

import json
import pathlib
import statistics as st

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "results_table_tok_lat_acc.png"

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, BAND = "#fcfcfb", "#f4f3ef"
GOOD, CRIT = "#0ca30c", "#d03b3b"

# tag, method, model, benchmark mix, n-note, path, vanilla-key, injected-key
RUNS = [
    ("3 v2", "Anchor-first + graph", "Qwen3-1.7B", "MATH-500", "",
     "phase-3/results/results.json", "vanilla_correct", "injection_correct"),
    ("3b", "Think injection (TI)", "Qwen3-1.7B", "MATH-500", "",
     "phase-3b/results/results.json", "vanilla_correct", "injection_correct"),
    ("3c", "ATGI v2 on code", "Qwen3-1.7B", "MBPP", "",
     "phase-3c/results/results.json", "vanilla_pass", "injection_pass"),
    ("3d", "ATGI v2 combined", "Qwen3-1.7B", "100 MATH / 187 AIME", "",
     "phase-3d/results/results_combined_fixed.jsonl",
     "vanilla_correct", "injection_correct"),
    ("3e", "ATBS (24K, no example)", "Qwen3-1.7B", "100 MATH / 40 AIME", "†",
     "phase-3e/results/results_clean.jsonl", "vanilla_correct", "atbs_correct"),
    ("4", "ATGI v2", "Qwen3-4B", "100 MATH / 187 AIME", "",
     "phase-4/results/results_qwen3-4b.jsonl",
     "vanilla_correct", "injection_correct"),
    ("4", "ATGI v2", "Qwen2.5-3B", "100 MATH / 13 AIME", "†",
     "phase-4/results/results_qwen2.5-3b.jsonl",
     "vanilla_correct", "injection_correct"),
    ("4", "ATGI v2", "Qwen3-0.6B", "78 MATH / 0 AIME", "†",
     "phase-4/results/results_qwen3-0.6b.jsonl",
     "vanilla_correct", "injection_correct"),
]


def load(path):
    p = REPO / path
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads(p.read_text())


rows = []
for tag, method, model, mix, note, path, vk, ik in RUNS:
    d = load(path)
    n = len(d)
    tk = "atbs_tokens" if ik.startswith("atbs") else "injection_tokens"
    lk = "atbs_latency_s" if ik.startswith("atbs") else "injection_latency_s"
    va = sum(bool(r[vk]) for r in d) / n * 100
    ia = sum(bool(r[ik]) for r in d) / n * 100
    vt = st.mean(r["vanilla_tokens"] for r in d)
    it = st.mean(r[tk] for r in d)
    vl = st.mean(r["vanilla_latency_s"] for r in d)
    il = st.mean(r[lk] for r in d)
    rows.append(dict(tag=tag, method=method, model=model, mix=mix, note=note,
                     n=n, va=va, ia=ia, d_acc=ia - va,
                     vt=vt, it=it, d_tok=(it - vt) / vt * 100,
                     vl=vl, il=il, d_lat=(il - vl) / vl * 100))

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17.6, 7.2), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 112); ax.set_ylim(0, 100)
ax.axis("off")

fig.text(0.022, 0.945, "PatternDistill — injection vs vanilla",
         fontsize=18, fontweight="bold", color=INK)
fig.text(0.022, 0.902,
         "Means over each run's own problems; Δ is paired within-run.",
         fontsize=10.5, color=INK_2)

# column x positions (data coords, 0–100)
C = dict(tag=2.6, method=6.6, model=21.0, mix=30.5, n=44.5,
         va=51.0, ia=58.5, dacc=66.0,
         vt=74.5, it=82.0, dtok=89.5,
         vl=98.0, il=104.5)
# metric groups: (label, x_start, x_end)
GROUPS = [("Accuracy  %", 47.5, 66.5), ("Mean tokens", 68.5, 90.0),
          ("Mean latency  s", 92.0, 111.6)]

TOP = 79.0
RH = 7.4

# group headers
for label, x0, x1 in GROUPS:
    ax.add_patch(Rectangle((x0, TOP + 8.4), x1 - x0, 0.35, color=INK_MUTED,
                           alpha=0.55, lw=0))
    ax.text((x0 + x1) / 2, TOP + 9.6, label, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK)

HEAD = [("tag", "Phase"), ("method", "Method"), ("model", "Model"),
        ("mix", "Eval set"), ("n", "n"),
        ("va", "vanilla"), ("ia", "injected"), ("dacc", "Δ pp"),
        ("vt", "vanilla"), ("it", "injected"), ("dtok", "Δ %"),
        ("vl", "vanilla"), ("il", "injected")]
for key, label in HEAD:
    align = "left" if key in ("tag", "method", "model", "mix") else "right"
    ax.text(C[key], TOP + 3.0, label, ha=align, va="center", fontsize=10,
            fontweight="bold", color=INK_2)
ax.plot([2.0, 111.6], [TOP + 0.9, TOP + 0.9], lw=1.3, color=INK_MUTED)

DLAT_X = 111.6


def delta(ax, x, y, value, unit, good_is_negative):
    """Signed delta with an arrow glyph — never colour alone."""
    up = value > 0
    beneficial = (not up) if good_is_negative else up
    color = GOOD if beneficial else CRIT
    if abs(value) < 0.05:
        color, glyph = INK_2, "="
    else:
        glyph = "▲" if up else "▼"
    ax.text(x, y, f"{glyph} {value:+.1f}{unit}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=color)


for i, r in enumerate(rows):
    y = TOP - 2.2 - i * RH
    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch((2.0, y - RH / 2 + 0.6), 109.6, RH - 1.2,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor=BAND, edgecolor="none",
                                    mutation_aspect=0.35, zorder=0))
    ax.text(C["tag"], y, r["tag"], ha="left", va="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.text(C["method"], y, r["method"], ha="left", va="center", fontsize=10.5,
            color=INK)
    ax.text(C["model"], y, r["model"], ha="left", va="center", fontsize=10.5,
            color=INK_2)
    ax.text(C["mix"], y, r["mix"] + r["note"], ha="left", va="center",
            fontsize=10, color=INK_2)
    ax.text(C["n"], y, str(r["n"]), ha="right", va="center", fontsize=10.5,
            color=INK_2)

    ax.text(C["va"], y, f"{r['va']:.1f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["ia"], y, f"{r['ia']:.1f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(ax, C["dacc"], y, r["d_acc"], "", good_is_negative=False)

    ax.text(C["vt"], y, f"{r['vt']:,.0f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["it"], y, f"{r['it']:,.0f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(ax, C["dtok"], y, r["d_tok"], "%", good_is_negative=True)

    ax.text(C["vl"], y, f"{r['vl']:.1f}", ha="right", va="center",
            fontsize=10.5, color=INK_2)
    ax.text(C["il"], y, f"{r['il']:.1f}", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    delta(ax, DLAT_X, y, r["d_lat"], "%", good_is_negative=True)

ax.text(DLAT_X, TOP + 3.0, "Δ %", ha="right", va="center", fontsize=10,
        fontweight="bold", color=INK_2)

ax.plot([2.0, 111.6], [TOP - 2.2 - (len(rows) - 0.5) * RH] * 2, lw=1.3,
        color=INK_MUTED)

FOOT = TOP - 2.2 - (len(rows) - 0.5) * RH - 3.6
ax.text(2.0, FOOT,
        "▲ / ▼ mark the direction; green = the desirable direction (higher "
        "accuracy, fewer tokens, lower latency).   Δ pp is percentage points; "
        "Δ % is relative to vanilla.",
        ha="left", va="top", fontsize=9.3, color=INK_2)
ax.text(2.0, FOOT - 3.4,
        "† partial run — killed early on the shared GPU box, so the eval set is "
        "MATH-heavy and its absolute accuracy is not comparable to a full "
        "287-problem row. The paired Δ still is.",
        ha="left", va="top", fontsize=9.3, color=INK_2)
ax.text(2.0, FOOT - 6.8,
        "All rows predate the gate-v2 pattern bank (phase-6). 3c is MBPP "
        "pass@1, not math accuracy. 3e reports ATBS, not ATGI.",
        ha="left", va="top", fontsize=9.3, color=INK_2)

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"wrote {OUT}")
