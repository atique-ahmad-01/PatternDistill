#!/usr/bin/env python3
"""T3 RAG over thinking traces (arXiv:2605.03344) on RB-DERIV -- pipeline CLI.

    python main.py verify      check dataset, split and servers, run nothing
    python main.py t3          teacher rewrites every memory-bank trace (t3.variant)
    python main.py evaluate    student arms on the test split
    python main.py report      summary tables from existing results

Common flags:
    --config PATH      default config.yaml
    --variant V        struct | semantic | reflect (overrides t3.variant;
                       results then land in results/t3_<V>/ unless --run is given)
    --run NAME         name this run; results land in results/<name>/
    --mock             use the offline fake backend (no GPU, no server)
    --arms A,B         run only these arms
    --no-resume        ignore existing output and start over
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from t3rag import dataset as D
from t3rag import embedding as E
from t3rag import memory as M
from t3rag import report as R
from t3rag import t3 as T3
from t3rag.config import data_path, load_config, read_jsonl, results_path, set_seeds, snapshot
from t3rag.llm import build_client
from t3rag.retrieval import Retriever

BASELINE_ARM = "A_baseline"


def get_logger(cfg, stage):
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("t3rag")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(Path(cfg["paths"]["logs_dir"]) / f"{stage}_{stamp}.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _apply_overrides(cfg, args):
    if args.variant:
        cfg["t3"]["variant"] = args.variant
        cfg["run_name"] = f"t3_{args.variant}"
    if args.run:
        cfg["run_name"] = args.run
    # The report labels the run with retrieval.mode.
    cfg["retrieval"]["mode"] = f"t3_{cfg['t3']['variant']}"
    if args.arms:
        cfg["evaluation"]["arms"] = [a.strip() for a in args.arms.split(",")]
    cfg["_mock"] = bool(args.mock)
    return cfg


# ── Stages ───────────────────────────────────────────────────────────────────

def cmd_verify(cfg, args, log):
    train, test, rep = D.split(cfg, log)
    log.info(f"split: {len(train)} train / {len(test)} test")

    expected = cfg["dataset"].get("sha256")
    actual = D.dataset_sha256(cfg)
    if expected and actual != expected:
        log.warning(f"DATASET CHANGED — expected {expected[:16]}, got "
                    f"{actual[:16]}. Results are not comparable with the baseline.")
    else:
        log.info(f"dataset integrity: OK ({actual[:16]})")

    entries = M.load(cfg)
    vecs = E.load_entry_vecs(cfg)
    log.info(f"memory bank: {len(entries)} entries, {len(vecs)} vectors")
    base = read_jsonl(Path(cfg["paths"]["baseline_dir"]) / f"eval_{BASELINE_ARM}.jsonl")
    missing = {p["problem_id"] for p in test} - {r["problem_id"] for r in base}
    log.info(f"baseline: {len(base)} rows"
             + (f", MISSING {len(missing)} test problems" if missing else ""))

    for role in ("teacher", "student"):
        c = build_client(cfg[role], seed=cfg["seed"], mock=cfg.get("_mock"))
        ok, msg = c.health()
        (log.info if ok else log.warning)(f"{role}: {'OK' if ok else 'UNAVAILABLE'} — {msg}")
    return {"split": rep}


def cmd_t3(cfg, args, log):
    """Teacher rewrites each memory entry's full trace into t3.variant.

    The teacher section in config has thinking off: this is a rewrite, not a
    solve, so Qwen3's non-thinking sampling settings (temperature 0.7, top_p
    0.8) apply.
    """
    entries = M.load(cfg)
    if not entries:
        raise SystemExit("memory bank is empty — data/artifacts/memory_bank.jsonl")
    traces = {r["solution_id"]: r["reasoning"]
              for r in read_jsonl(data_path(cfg, "teacher_solutions_validated.jsonl"))}
    section = dict(cfg["teacher"], max_tokens=cfg["t3"]["max_tokens"])
    client = build_client(section, seed=cfg["seed"], mock=cfg.get("_mock"))
    ok, msg = client.health()
    if not ok:
        raise SystemExit(f"teacher unavailable — {msg}\n"
                         f"Start it with scripts/serve_vllm.sh teacher, or pass --mock.")
    log.info(f"teacher: {msg}")
    _, rep = T3.transform(cfg, client, entries, traces, cfg["t3"]["variant"], log,
                          resume=not args.no_resume)
    return {"t3": rep}


def cmd_evaluate(cfg, args, log):
    from t3rag.student import run_arm
    train, test, rep = D.split(cfg, log)
    entries = M.load(cfg)
    texts = T3.load(cfg, cfg["t3"]["variant"])

    # The baseline is copied in, never run: see t3rag/student.py.
    dst = results_path(cfg, f"eval_{BASELINE_ARM}.jsonl")
    if not dst.exists():
        shutil.copy(Path(cfg["paths"]["baseline_dir"]) / f"eval_{BASELINE_ARM}.jsonl", dst)
        log.info(f"  [{BASELINE_ARM}] copied from data/baseline/")

    arms = [a for a in cfg["evaluation"]["arms"] if a != BASELINE_ARM]
    if not arms:
        return {"split": rep}
    embedder = E.load_embedder(cfg, log)
    if cfg.get("_mock"):
        # The stored vectors are nomic's 768-d; the mock encoder is 256-d.
        vecs = E.embed(cfg, embedder, [e["problem"] for e in entries])
    else:
        vecs = E.load_entry_vecs(cfg)
    retriever = Retriever(cfg, entries, vecs, embedder, log)
    client = build_client(cfg["student"], seed=cfg["seed"], mock=cfg.get("_mock"))
    if cfg.get("_mock"):
        client.set_answers({p["problem_id"]: p["answer"] for p in test}, test)
    ok, msg = client.health()
    if not ok:
        raise SystemExit(f"student unavailable — {msg}\n"
                         f"Start it with scripts/serve_vllm.sh student, or pass --mock.")
    log.info(f"student: {msg}")
    for arm in arms:
        run_arm(cfg, arm, client, test, retriever, texts, log,
                resume=not args.no_resume)
    return {"split": rep}


def cmd_report(cfg, args, log):
    return {"report": R.summarize(cfg, log)}


COMMANDS = {
    "verify": cmd_verify,
    "t3": cmd_t3,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", default=None, choices=list(T3.VARIANTS))
    ap.add_argument("--run", default=None, help="run name; results/<name>/")
    ap.add_argument("--mock", action="store_true",
                    help="offline fake backend; results are stamped mock=true")
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = _apply_overrides(load_config(args.config), args)
    set_seeds(cfg["seed"])
    log = get_logger(cfg, args.command)
    log.info(f"=== t3rag {args.command} (run={cfg.get('run_name')} "
             f"variant={cfg['t3']['variant']}{' MOCK' if cfg.get('_mock') else ''}) ===")

    meta = COMMANDS[args.command](cfg, args, log) or {}

    run_dir = Path(cfg["paths"]["results_dir"]) / cfg.get("run_name", "default")
    snap = snapshot(cfg, run_dir)
    manifest = results_path(cfg, "run_manifest.json")
    Path(manifest).write_text(json.dumps({
        "command": args.command,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "mock": bool(cfg.get("_mock")),
        "config_snapshot": str(snap),
        **{k: v for k, v in meta.items() if k != "report"},
    }, indent=2, ensure_ascii=False, default=str))
    log.info(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
