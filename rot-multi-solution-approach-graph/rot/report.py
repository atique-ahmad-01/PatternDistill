"""Stage 7 -- scoring and reporting.

Produces, from the per-arm evaluation JSONL:

  * results/<run>/summary.csv        one row per arm
  * results/<run>/summary.md         the table, rendered
  * results/<run>/by_group.csv       per-family accuracy for every arm
  * results/<run>/per_problem.csv    every problem x arm
  * results/<run>/summary.json       everything above, machine-readable

Improvements are reported against the baseline arm both ways, because the two
answer different questions and quoting only the flattering one is the most
common way these numbers mislead:

  absolute = acc_arm - acc_base              (percentage points)
  relative = (acc_arm - acc_base) / acc_base (percent of the baseline)

Token and latency figures are reported alongside accuracy, never on their own.
A large token saving with collapsing accuracy is the model giving up, not a
saving -- so the table always shows both in the same row.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import read_jsonl, results_path
from .llm import split_thinking_tokens


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def collect(cfg):
    rows = []
    for arm in cfg["evaluation"]["arms"]:
        p = results_path(cfg, f"eval_{arm}.jsonl")
        rows.extend(read_jsonl(p))
    return rows


def summarize(cfg, log, extra=None):
    rows = collect(cfg)
    if not rows:
        raise FileNotFoundError(
            "no evaluation rows found -- run `main.py evaluate` first")

    # Recompute the thinking/answer split from the stored raw_reply. Result
    # files written before the fix in llm.split_thinking_tokens booked an
    # unterminated <think> as pure ANSWER -- exactly backwards, and biased per
    # arm because truncation rates differ per arm. raw_reply is stored on every
    # row, so old runs are corrected on read and need no student re-run.
    for r in rows:
        if r.get("raw_reply") is not None and r.get("output_tokens") is not None:
            t, a = split_thinking_tokens(r["raw_reply"], r["output_tokens"])
            r["thinking_tokens"], r["answer_tokens"] = t, a

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    base_arm = cfg["evaluation"]["baseline_arm"]
    base_rows = by_arm.get(base_arm, [])
    base_acc = (sum(1 for r in base_rows if r["correct"]) / len(base_rows)
                if base_rows else 0.0)

    # Compare arms only on problems every arm actually answered. A partial run
    # would otherwise compare a 48-problem arm against a 12-problem arm.
    common = None
    for arm, rs in by_arm.items():
        ids = {r["problem_id"] for r in rs}
        common = ids if common is None else (common & ids)
    common = common or set()

    summary = []
    for arm in cfg["evaluation"]["arms"]:
        rs = [r for r in by_arm.get(arm, []) if r["problem_id"] in common]
        if not rs:
            continue
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"])
        acc = n_correct / n
        b = [r for r in base_rows if r["problem_id"] in common]
        bacc = (sum(1 for r in b if r["correct"]) / len(b)) if b else 0.0
        same_grp = [x for r in rs for x in (r.get("retrieved_same_group") or [])]

        # A truncated row's token count is CENSORED: the model hit max_tokens,
        # so its true trace is >= that number by an unknown amount. Averaging
        # the cap in drags every mean toward it, and because arms truncate at
        # different rates the drag differs per arm -- which manufactures arm
        # differences out of nothing. Token and latency means are therefore
        # taken over FINISHED rows only, and truncation is reported as its own
        # outcome rather than being folded into the token figures.
        fin = [r for r in rs if not r.get("truncated")]
        fin_by_id = {r["problem_id"]: r for r in fin}
        bfin_by_id = {r["problem_id"]: r for r in b if not r.get("truncated")}

        # Paired against the baseline on problems that FINISHED IN BOTH arms.
        # Comparing each arm's own finished subset to the baseline's would
        # compare different problem sets, and the subsets differ precisely by
        # difficulty -- the hard ones are the ones that truncate.
        pair_ids = [i for i in fin_by_id if i in bfin_by_id]
        if pair_ids and arm != base_arm:
            pb = sum(bfin_by_id[i]["output_tokens"] for i in pair_ids) / len(pair_ids)
            pa = sum(fin_by_id[i]["output_tokens"] for i in pair_ids) / len(pair_ids)
            paired_pct = round((pa - pb) / pb * 100, 2) if pb else None
        else:
            paired_pct = None

        summary.append({
            "arm": arm,
            "n": n,
            "correct": n_correct,
            "incorrect": n - n_correct,
            "unparseable": sum(1 for r in rs if r.get("unparseable")),
            "truncated": sum(1 for r in rs if r.get("truncated")),
            "accuracy": round(acc, 4),
            "abs_improvement_pp": round((acc - bacc) * 100, 2),
            "rel_improvement_pct": (round((acc - bacc) / bacc * 100, 2)
                                    if bacc > 0 else None),
            "n_finished": len(fin),
            # input tokens are the prompt: never censored, so all rows count
            "mean_input_tokens": _mean([r.get("input_tokens") for r in rs]),
            "mean_output_tokens": _mean([r.get("output_tokens") for r in fin]),
            "mean_thinking_tokens": _mean([r.get("thinking_tokens") for r in fin]),
            "mean_answer_tokens": _mean([r.get("answer_tokens") for r in fin]),
            "mean_latency_s": _mean([r.get("latency_s") for r in fin]),
            "n_paired_vs_base": len(pair_ids),
            "paired_output_tokens_vs_base_pct": paired_pct,
            "mean_retrieval_score": _mean(
                [s for r in rs for s in (r.get("retrieval_scores") or [])
                 if s is not None]),
            "same_family_retrieval_rate": (round(sum(same_grp) / len(same_grp), 4)
                                           if same_grp else None),
        })

    # Per-family. RB-DERIV requires this: digital_root and zeckendorf have only
    # 6/12 distinct golds, so guessing beats chance there and an aggregate
    # number can hide it.
    by_group = []
    groups = sorted({str(r.get("group")) for r in rows})
    for g in groups:
        entry = {"group": g}
        for arm in cfg["evaluation"]["arms"]:
            rs = [r for r in by_arm.get(arm, [])
                  if str(r.get("group")) == g and r["problem_id"] in common]
            entry[arm] = (round(sum(1 for r in rs if r["correct"]) / len(rs), 4)
                          if rs else None)
            entry[f"{arm}_n"] = len(rs)
        by_group.append(entry)

    out = {
        "run_name": cfg.get("run_name", "default"),
        "student_model": cfg["student"]["model"],
        "teacher_model": cfg["teacher"]["model"],
        "retrieval_mode": cfg["retrieval"]["mode"],
        "top_k": cfg["retrieval"]["top_k"],
        "seed": cfg["seed"],
        "baseline_arm": base_arm,
        "n_problems_compared": len(common),
        "summary": summary,
        "by_group": by_group,
    }
    if extra:
        out.update(extra)

    _write_csv(results_path(cfg, "summary.csv"), summary)
    _write_csv(results_path(cfg, "by_group.csv"), by_group)
    _write_per_problem(cfg, rows, common)
    Path(results_path(cfg, "summary.json")).write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    md = render_table(out, cfg)
    Path(results_path(cfg, "summary.md")).write_text(md)

    log.info("\n" + md)
    log.info(f"-> {results_path(cfg, 'summary.md')}")
    return out


def render_table(out, cfg) -> str:
    arms = cfg["evaluation"]["arms"]
    base_arm = out["baseline_arm"]

    lines = [
        f"# RoT results — {out['run_name']}", "",
        f"- teacher: `{out['teacher_model']}`",
        f"- student: `{out['student_model']}`",
        f"- retrieval: `{out['retrieval_mode']}`, top_k={out['top_k']}, seed={out['seed']}",
        f"- problems compared across all arms: **{out['n_problems_compared']}**", "",
        "| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s_row in out["summary"]:
        is_base = s_row["arm"] == base_arm
        name = f"**{s_row['arm']}**" if is_base else s_row["arm"]

        if is_base:
            absd = reld = "—"
        else:
            absd = f"{s_row['abs_improvement_pp']:+.1f}"
            reld = ("—" if s_row["rel_improvement_pct"] is None
                    else f"{s_row['rel_improvement_pct']:+.1f}")

        sf = ("—" if s_row["same_family_retrieval_rate"] is None
              else f"{s_row['same_family_retrieval_rate']:.0%}")

        ptok = s_row.get("paired_output_tokens_vs_base_pct")
        ptok = "—" if ptok is None else f"{ptok:+.1f}%"

        lines.append(
            f"| {name} | {s_row['n']} | {s_row['correct']} | "
            f"{s_row['accuracy']:.1%} | {absd} | {reld} | "
            f"{s_row.get('truncated', 0)} | "
            f"{s_row['mean_input_tokens']:.0f} | "
            f"{s_row.get('mean_thinking_tokens', 0):.0f} | "
            f"{s_row['mean_output_tokens']:.0f} | {ptok} | "
            f"{s_row['mean_latency_s']:.1f} | {sf} |"
        )

    lines += [
        "",
        "`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.",
        "`same-family` = share of retrieved memories drawn from the query's own "
        "technique family.",
        "`think tok` = reasoning-trace tokens, apportioned from the reply by "
        "character share. This is the quantity the experiment is about: a "
        "retrieved approach should collapse the SEARCH for a strategy, which "
        "lives here rather than in the final answer.",
        "",
        "`trunc` = replies that hit `student.max_tokens` mid-trace. These never "
        "reach an answer, so they score as WRONG and are indistinguishable from "
        "a genuine failure in the accuracy column -- read the two together.",
        "",
        "**Token and latency means cover FINISHED replies only.** A truncated "
        "reply is censored at the cap: its true trace is longer by an unknown "
        "amount, so averaging the cap in drags the mean toward it, by an amount "
        "that differs per arm because truncation rates differ per arm. "
        "`Δtok paired` is the honest comparison -- mean output tokens against "
        "the baseline over problems that finished in BOTH arms, so the same "
        "problems are on both sides.",
        "",
        "## Per-family accuracy",
        "",
        "| family | " + " | ".join(arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]

    low = set(cfg["evaluation"].get("low_diversity_groups") or [])
    for g in out["by_group"]:
        name = g["group"] + (" ⚠" if g["group"] in low else "")
        cells = []
        for arm in arms:
            v = g.get(arm)
            cells.append("—" if v is None else f"{v:.0%}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    if low:
        lines += ["", "⚠ = few distinct golds in this family; guessing scores above "
                      "chance there, so read those cells with care."]
    return "\n".join(lines)


def _write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_per_problem(cfg, rows, common):
    import csv
    cols = ["arm", "problem_id", "group", "gold_answer", "predicted_answer",
            "correct", "unparseable", "truncated", "input_tokens",
            "output_tokens", "thinking_tokens", "answer_tokens",
            "latency_s", "n_retrieved", "retrieved_ids",
            "retrieval_scores", "retrieved_same_group"]
    path = results_path(cfg, "per_problem.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda x: (x["problem_id"], x["arm"])):
            if r["problem_id"] not in common:
                continue
            w.writerow([_flat(r.get(c)) for c in cols])


def _flat(v):
    if isinstance(v, (list, tuple)):
        return ";".join("" if x is None else str(x) for x in v)
    return "" if v is None else v
