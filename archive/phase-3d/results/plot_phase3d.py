#!/usr/bin/env python3
"""Phase 3D — Combined visualization: pass/fail overlap, latency, token usage."""

import json, pathlib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

DATA = pathlib.Path(__file__).parent / "results_combined_fixed.jsonl"
rows = [json.loads(l) for l in DATA.read_text().splitlines()]

# ---------- categorise ----------
both_pass, both_fail, atgi_win, atgi_loss = [], [], [], []
for r in rows:
    vc, ic = r["vanilla_correct"], r["injection_correct"]
    if vc and ic:
        both_pass.append(r)
    elif not vc and not ic:
        both_fail.append(r)
    elif not vc and ic:
        atgi_win.append(r)
    else:
        atgi_loss.append(r)

# ---------- figure ----------
fig = plt.figure(figsize=(18, 10), facecolor="white")
fig.suptitle("Phase 3D — ATGI vs Vanilla  (287 problems: 100 MATH-500 + 187 AIME)",
             fontsize=15, fontweight="bold", y=0.97)

# ── Panel 1: Stacked bar — pass / fail breakdown ──
ax1 = fig.add_subplot(2, 3, 1)
cats = ["Vanilla", "ATGI"]
pass_v = len(both_pass) + len(atgi_loss)   # vanilla correct
pass_a = len(both_pass) + len(atgi_win)    # atgi correct
fail_v = len(both_fail) + len(atgi_win)
fail_a = len(both_fail) + len(atgi_loss)

bars_pass = ax1.bar(cats, [pass_v, pass_a], color="#4CAF50", label="Correct")
bars_fail = ax1.bar(cats, [fail_v, fail_a], bottom=[pass_v, pass_a], color="#E57373", label="Wrong")
for bar, val in zip(bars_pass, [pass_v, pass_a]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             f"{val}\n({val/287*100:.1f}%)", ha="center", va="center", fontweight="bold", color="white", fontsize=12)
for bar, bot, val in zip(bars_fail, [pass_v, pass_a], [fail_v, fail_a]):
    ax1.text(bar.get_x() + bar.get_width()/2, bot + val/2,
             f"{val}", ha="center", va="center", color="white", fontsize=12)
ax1.set_ylabel("Problems")
ax1.set_title("Pass / Fail Counts")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim(0, 310)

# ── Panel 2: Overlap matrix (Venn-style table) ──
ax2 = fig.add_subplot(2, 3, 2)
ax2.axis("off")
matrix = [
    ["", "ATGI ✓", "ATGI ✗"],
    ["Vanilla ✓", f"{len(both_pass)}", f"{len(atgi_loss)}"],
    ["Vanilla ✗", f"{len(atgi_win)}", f"{len(both_fail)}"],
]
colors = [
    ["#EEEEEE", "#BBDEFB", "#BBDEFB"],
    ["#BBDEFB", "#C8E6C9", "#FFCDD2"],
    ["#BBDEFB", "#C8E6C9", "#FFCDD2"],
]
table = ax2.table(cellText=matrix, cellColours=colors, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.2, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == 0:
        cell.set_text_props(fontweight="bold")
ax2.set_title("Overlap Matrix", pad=20)

# ── Panel 3: Wins / Losses by dataset ──
ax3 = fig.add_subplot(2, 3, 3)
# per-dataset wins/losses
math_wins = sum(1 for r in atgi_win if r["source"] == "MATH-500")
math_losses = sum(1 for r in atgi_loss if r["source"] == "MATH-500")
aime_wins = sum(1 for r in atgi_win if r["source"] == "AIME")
aime_losses = sum(1 for r in atgi_loss if r["source"] == "AIME")

x = np.arange(2)
w = 0.35
b1 = ax3.bar(x - w/2, [math_wins, aime_wins], w, color="#4CAF50", label="ATGI Wins")
b2 = ax3.bar(x + w/2, [-math_losses, -aime_losses], w, color="#E57373", label="ATGI Losses")
ax3.set_xticks(x)
ax3.set_xticklabels(["MATH-500", "AIME"])
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("Count (losses shown negative)")
ax3.set_title("Wins vs Losses by Dataset")
for bar, val in zip(b1, [math_wins, aime_wins]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"+{val}", ha="center", fontweight="bold", color="#2E7D32")
for bar, val in zip(b2, [math_losses, aime_losses]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.0, f"-{val}", ha="center", fontweight="bold", color="#C62828")
ax3.legend(fontsize=9)

# ── Panel 4: Latency by outcome category ──
ax4 = fig.add_subplot(2, 3, 4)
cat_names = ["Both ✓", "Both ✗", "ATGI Wins", "ATGI Losses"]
cat_lists = [both_pass, both_fail, atgi_win, atgi_loss]
van_lat = [np.mean([r["vanilla_latency_s"] for r in c]) if c else 0 for c in cat_lists]
atgi_lat = [np.mean([r["injection_latency_s"] for r in c]) if c else 0 for c in cat_lists]

x = np.arange(len(cat_names))
w = 0.35
b1 = ax4.bar(x - w/2, van_lat, w, color="#90CAF9", label="Vanilla")
b2 = ax4.bar(x + w/2, atgi_lat, w, color="#FFB74D", label="ATGI")
ax4.set_xticks(x)
ax4.set_xticklabels(cat_names, fontsize=10)
ax4.set_ylabel("Avg Latency (s)")
ax4.set_title("Latency by Outcome")
ax4.legend(fontsize=9)
for bar, val in zip(list(b1) + list(b2), list(van_lat) + list(atgi_lat)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}s", ha="center", fontsize=9)

# ── Panel 5: Token usage by outcome category ──
ax5 = fig.add_subplot(2, 3, 5)
van_tok = [np.mean([r["vanilla_tokens"] for r in c]) if c else 0 for c in cat_lists]
atgi_tok = [np.mean([r["injection_tokens"] for r in c]) if c else 0 for c in cat_lists]

b1 = ax5.bar(x - w/2, van_tok, w, color="#90CAF9", label="Vanilla")
b2 = ax5.bar(x + w/2, atgi_tok, w, color="#FFB74D", label="ATGI")
ax5.set_xticks(x)
ax5.set_xticklabels(cat_names, fontsize=10)
ax5.set_ylabel("Avg Tokens")
ax5.set_title("Token Usage by Outcome")
ax5.legend(fontsize=9)
for bar, val in zip(list(b1) + list(b2), list(van_tok) + list(atgi_tok)):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f"{val:.0f}", ha="center", fontsize=9)

# ── Panel 6: Overall latency & token totals ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
total_van_lat = sum(r["vanilla_latency_s"] for r in rows)
total_atgi_lat = sum(r["injection_latency_s"] for r in rows)
total_van_tok = sum(r["vanilla_tokens"] for r in rows)
total_atgi_tok = sum(r["injection_tokens"] for r in rows)
avg_van_lat = np.mean([r["vanilla_latency_s"] for r in rows])
avg_atgi_lat = np.mean([r["injection_latency_s"] for r in rows])
van_limit = sum(1 for r in rows if r["vanilla_tokens"] >= 16384)
atgi_limit = sum(1 for r in rows if r["injection_tokens"] >= 16384)

summary = [
    ["Metric", "Vanilla", "ATGI", "Δ"],
    ["Avg Latency", f"{avg_van_lat:.1f}s", f"{avg_atgi_lat:.1f}s", f"{avg_atgi_lat - avg_van_lat:+.1f}s"],
    ["Total Tokens", f"{total_van_tok:,}", f"{total_atgi_tok:,}", f"{total_atgi_tok - total_van_tok:+,}"],
    ["Avg Tokens", f"{total_van_tok/287:,.0f}", f"{total_atgi_tok/287:,.0f}", f"{(total_atgi_tok - total_van_tok)/287:+,.0f}"],
    ["Hit 16K Limit", f"{van_limit}", f"{atgi_limit}", f"{atgi_limit - van_limit:+d}"],
    ["Correct", f"{pass_v} ({pass_v/287*100:.1f}%)", f"{pass_a} ({pass_a/287*100:.1f}%)", f"+{pass_a - pass_v}"],
]
colors_s = [["#E0E0E0"]*4] + [["#F5F5F5"]*4]*5
table2 = ax6.table(cellText=summary, cellColours=colors_s, loc="center", cellLoc="center")
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.3, 1.8)
for (r, c), cell in table2.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
    if c == 3 and r > 0:
        val = summary[r][3]
        cell.set_text_props(color="#2E7D32" if val.startswith("-") or (val.startswith("+") and r == 5) else "#C62828")
ax6.set_title("Overall Summary", pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = pathlib.Path(__file__).parent / "phase3d_dashboard.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
