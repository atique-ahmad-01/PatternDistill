import json
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1/2 — vanilla / injection
GOOD, CRIT = "#006300", "#d03b3b"      # delta annotation colors

rows = [json.loads(l) for l in open("results_qwen3-4b.jsonl")]


def agg(rows, key):
    d = defaultdict(lambda: {"n": 0, "vl": [], "il": [], "vt": [], "it": []})
    for r in rows:
        g = d[key(r)]
        g["n"] += 1
        g["vl"].append(r["vanilla_latency_s"])
        g["il"].append(r["injection_latency_s"])
        g["vt"].append(r["vanilla_tokens"])
        g["it"].append(r["injection_tokens"])
    return d


def means(groups, order, field_v, field_i):
    van = [st.mean(groups[k][field_v]) for k in order]
    inj = [st.mean(groups[k][field_i]) for k in order]
    n = [groups[k]["n"] for k in order]
    return van, inj, n


def bar_panel(ax, labels, van, inj, n, title, ylabel, fmt="{:+.0f}"):
    x = np.arange(len(labels))
    w = 0.35
    bv = ax.bar(x - w / 2, van, w, label="Vanilla", color=BLUE, alpha=0.9)
    bi = ax.bar(x + w / 2, inj, w, label="Injection", color=ORANGE, alpha=0.9)

    ymax = max(max(van), max(inj))
    for rv, ri, cnt in zip(bv, bi, n):
        delta = ri.get_height() - rv.get_height()
        color = GOOD if delta <= 0 else CRIT  # lower is better for latency/tokens
        ax.annotate(
            fmt.format(delta),
            xy=(ri.get_x() + ri.get_width() / 2, max(ri.get_height(), rv.get_height()) + ymax * 0.02),
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )
        ax.annotate(f"n={cnt}", xy=(ri.get_x() - w / 2, ymax * 0.02), ha="center", va="bottom",
                     fontsize=7, color="#898781")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Phase 4 — Qwen3-4B Latency & Token Usage (n=287: 100 MATH-500 + 187 AIME)",
             fontsize=14, fontweight="bold")

# ── by source ─────────────────────────────────────────────────────────────
src_groups = agg(rows, lambda r: r["source"])
src_groups["Overall"] = {
    "n": len(rows),
    "vl": [r["vanilla_latency_s"] for r in rows], "il": [r["injection_latency_s"] for r in rows],
    "vt": [r["vanilla_tokens"] for r in rows], "it": [r["injection_tokens"] for r in rows],
}
src_order = ["MATH-500", "AIME", "Overall"]

van_l, inj_l, n_l = means(src_groups, src_order, "vl", "il")
bar_panel(axes[0, 0], src_order, van_l, inj_l, n_l, "Latency — Overall & by Source", "Seconds", "{:+.1f}s")

van_t, inj_t, n_t = means(src_groups, src_order, "vt", "it")
bar_panel(axes[0, 1], src_order, van_t, inj_t, n_t, "Token Usage — Overall & by Source", "Tokens", "{:+.0f}")

# ── MATH-500 by difficulty level ─────────────────────────────────────────────
lvl_groups = agg([r for r in rows if r["source"] == "MATH-500"], lambda r: r["level"])
levels = sorted(lvl_groups.keys(), key=int)
lvl_labels = [f"Level {l}" for l in levels]

van_ll, inj_ll, n_ll = means(lvl_groups, levels, "vl", "il")
bar_panel(axes[1, 0], lvl_labels, van_ll, inj_ll, n_ll, "MATH-500 Latency — by Difficulty Level", "Seconds", "{:+.1f}s")

van_lt, inj_lt, n_lt = means(lvl_groups, levels, "vt", "it")
bar_panel(axes[1, 1], lvl_labels, van_lt, inj_lt, n_lt, "MATH-500 Token Usage — by Difficulty Level", "Tokens", "{:+.0f}")

o_lat_v, o_lat_i = st.mean([r["vanilla_latency_s"] for r in rows]), st.mean([r["injection_latency_s"] for r in rows])
o_tok_v, o_tok_i = st.mean([r["vanilla_tokens"] for r in rows]), st.mean([r["injection_tokens"] for r in rows])
fig.text(
    0.5, 0.01,
    f"Overall mean latency: Vanilla={o_lat_v:.1f}s  Injection={o_lat_i:.1f}s  ({o_lat_i - o_lat_v:+.1f}s)  |  "
    f"Overall mean tokens: Vanilla={o_tok_v:.0f}  Injection={o_tok_i:.0f}  ({o_tok_i - o_tok_v:+.0f})",
    ha="center", fontsize=9, color="#52514e",
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("latency_tokens_histogram_qwen3-4b.png", dpi=150, bbox_inches="tight")
print("Saved: latency_tokens_histogram_qwen3-4b.png")
