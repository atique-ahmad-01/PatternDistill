#!/usr/bin/env python3
"""Retrieval-of-Thought (RoT) — pipeline CLI.

    python main.py verify                 check dataset + servers, run nothing
    python main.py generate               teacher: N solutions per train problem
    python main.py validate               score solutions against gold
    python main.py build-memory           problem -> solution -> approach -> steps
    python main.py build-graph            nodes, semantic edges, embeddings
    python main.py evaluate               student arms A/B/C/D on the test split
    python main.py report                 summary tables from existing results
    python main.py all                    every stage in order

Common flags:
    --config PATH      default config.yaml
    --run NAME         name this run; results land in results/<name>/
    --mock             use the offline fake backend (no GPU, no server)
    --limit-train N    cap train problems (also settable in config)
    --limit-test N     cap test problems
    --arms A,B         run only these arms
    --no-resume        ignore existing output and start over
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rot import dataset as D
from rot import graph as G
from rot import memory as M
from rot import report as R
from rot import teacher as T
from rot import validate as V
from rot.config import load_config, results_path, set_seeds, snapshot
from rot.llm import build_client
from rot.retrieval import Retriever


def get_logger(cfg, stage):
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("rot")
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
    if args.run:
        cfg["run_name"] = args.run
    if args.limit_train is not None:
        cfg["dataset"]["limits"]["max_train"] = args.limit_train
    if args.limit_test is not None:
        cfg["dataset"]["limits"]["max_test"] = args.limit_test
    if args.top_k is not None:
        cfg["retrieval"]["top_k"] = args.top_k
    if args.retrieval_mode:
        cfg["retrieval"]["mode"] = args.retrieval_mode
    if args.arms:
        cfg["evaluation"]["arms"] = [a.strip() for a in args.arms.split(",")]
    cfg["_mock"] = bool(args.mock)
    return cfg


# ── Stages ───────────────────────────────────────────────────────────────────

def cmd_verify(cfg, args, log):
    train, test, rep = D.split(cfg, log, refresh=getattr(args, 'refresh_split', False))
    log.info(json.dumps({k: v for k, v in rep.items()
                         if k not in ("train_groups", "test_groups")}, indent=2))

    # Integrity check against a recorded digest. Self-contained on purpose:
    # the project must not depend on any sibling directory existing.
    expected = cfg["dataset"].get("sha256")
    actual = D.dataset_sha256(cfg)
    if expected:
        if actual == expected:
            log.info(f"dataset integrity: OK ({actual[:16]})")
        else:
            log.warning(f"DATASET CHANGED — expected {expected[:16]}, "
                        f"got {actual[:16]}. Results are not comparable with "
                        f"runs made before the change.")
    else:
        log.info(f"dataset sha256: {actual} (record it as dataset.sha256)")

    for role in ("teacher", "student"):
        c = build_client(cfg[role], seed=cfg["seed"], mock=cfg.get("_mock"))
        ok, msg = c.health()
        (log.info if ok else log.warning)(f"{role}: {'OK' if ok else 'UNAVAILABLE'} — {msg}")
    return {"split": rep}


def cmd_generate(cfg, args, log):
    train, test, rep = D.split(cfg, log, refresh=getattr(args, 'refresh_split', False))
    client = build_client(cfg["teacher"], seed=cfg["seed"], mock=cfg.get("_mock"))
    if cfg.get("_mock"):
        client.set_answers({p["problem_id"]: p["answer"] for p in train}, train,
                           error_rate=0.25)
    ok, msg = client.health()
    if not ok:
        raise SystemExit(
            f"teacher unavailable — {msg}\n"
            f"Start it with scripts/serve_vllm.sh teacher, or pass --mock.")
    log.info(f"teacher: {msg}")
    T.generate(cfg, client, train, log, resume=not args.no_resume)
    return {"split": rep}


def cmd_validate(cfg, args, log):
    _, _, rep = D.split(cfg, log, refresh=getattr(args, 'refresh_split', False))
    rows, vrep = V.validate(cfg, log)
    return {"validation": vrep}


def cmd_build_memory(cfg, args, log):
    train, _, _ = D.split(cfg, log, refresh=getattr(args, 'refresh_split', False))
    entries, mrep = M.build(cfg, log, train_ids={p["problem_id"] for p in train})
    return {"memory": mrep}


def cmd_build_graph(cfg, args, log):
    entries = M.load(cfg)
    _, _, grep = G.build(cfg, entries, log)
    return {"graph": grep}


def cmd_evaluate(cfg, args, log):
    from rot.student import run_arm
    train, test, rep = D.split(cfg, log, refresh=getattr(args, 'refresh_split', False))
    entries = M.load(cfg)
    if not entries:
        raise SystemExit("memory bank is empty — run build-memory first")
    _, _, entry_vecs = G.load(cfg)
    embedder = G.load_embedder(cfg, log)
    retriever = Retriever(cfg, entries, entry_vecs, embedder, log)

    client = build_client(cfg["student"], seed=cfg["seed"], mock=cfg.get("_mock"))
    if cfg.get("_mock"):
        # The mock student solves a problem only when memory is in the prompt,
        # so the arms differ and the reporting path gets exercised.
        client.set_answers({p["problem_id"]: p["answer"] for p in test}, test)
    ok, msg = client.health()
    if not ok:
        raise SystemExit(f"student unavailable — {msg}\n"
                         f"Start it with scripts/serve_vllm.sh student, or pass --mock.")
    log.info(f"student: {msg}")

    for arm in cfg["evaluation"]["arms"]:
        run_arm(cfg, arm, client, test, retriever, log, resume=not args.no_resume)
    return {"split": rep}


def cmd_report(cfg, args, log):
    return {"report": R.summarize(cfg, log)}


def cmd_all(cfg, args, log):
    meta = {}
    meta.update(cmd_generate(cfg, args, log))
    meta.update(cmd_validate(cfg, args, log))
    meta.update(cmd_build_memory(cfg, args, log))
    meta.update(cmd_build_graph(cfg, args, log))
    meta.update(cmd_evaluate(cfg, args, log))
    R.summarize(cfg, log, extra=meta)
    return meta


COMMANDS = {
    "verify": cmd_verify,
    "generate": cmd_generate,
    "validate": cmd_validate,
    "build-memory": cmd_build_memory,
    "build-graph": cmd_build_graph,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
    "all": cmd_all,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default=None, help="run name; results/<name>/")
    ap.add_argument("--mock", action="store_true",
                    help="offline fake backend; results are stamped mock=true")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--retrieval-mode", default=None,
                    choices=["full_solution", "approach", "subtasks",
                             "approach_subtasks"])
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--refresh-split", action="store_true",
                    help="recompute and re-freeze the split "
                         "(invalidates any data generated under the old one)")
    args = ap.parse_args()

    cfg = _apply_overrides(load_config(args.config), args)
    set_seeds(cfg["seed"])
    log = get_logger(cfg, args.command)
    log.info(f"=== rot {args.command} "
             f"(run={cfg.get('run_name')}{' MOCK' if cfg.get('_mock') else ''}) ===")

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
