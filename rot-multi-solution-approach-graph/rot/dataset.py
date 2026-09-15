"""Dataset loading, filtering, and train/test splitting.

The dataset is READ-ONLY. Nothing here writes to it, and every stage re-derives
its view from the one file, so there is a single source of truth for problems
and golds.

Source: RB_DERIV/RB_DERIV_questions.json -- 192 derivation-deep questions,
16 technique families x 12 instances, every gold an exact integer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .scoring import is_gradable


def load_raw(cfg) -> list[dict]:
    """Load records exactly as they sit on disk."""
    path = Path(cfg["dataset"]["path"])
    blob = json.loads(path.read_text())
    key = cfg["dataset"].get("records_key")
    if key:
        if key not in blob:
            raise KeyError(f"records_key {key!r} not in {path} "
                           f"(top-level keys: {list(blob)})")
        records = blob[key]
    else:
        records = blob
    if not isinstance(records, list):
        raise ValueError(f"expected a list of records, got {type(records)}")
    return records


def normalize(cfg, records) -> list[dict]:
    """Map dataset field names onto the pipeline's internal names."""
    f = cfg["dataset"]["fields"]
    keep = cfg["dataset"].get("keep_fields") or []
    out = []
    for i, r in enumerate(records):
        item = {
            "problem_id": str(r[f["id"]]),
            "problem": r[f["problem"]],
            "answer": r[f["answer"]],
            "group": r.get(f.get("group")) if f.get("group") else None,
            "_index": i,
        }
        for k in keep:
            if k in r:
                item[k] = r[k]
        out.append(item)
    return out


def dataset_sha256(cfg) -> str:
    """Full SHA-256 of the dataset file."""
    return hashlib.sha256(Path(cfg["dataset"]["path"]).read_bytes()).hexdigest()


def dataset_fingerprint(cfg) -> str:
    """SHA-256 of the dataset file, recorded with every result set.

    Makes it impossible to compare two runs against different data without
    noticing.
    """
    h = hashlib.sha256(Path(cfg["dataset"]["path"]).read_bytes()).hexdigest()
    return h[:16]


def _compute_split(cfg, log=None):
    """Compute the split from scratch. Use `split()` instead."""
    items = normalize(cfg, load_raw(cfg))
    n_total = len(items)

    if cfg["dataset"]["gradable_only"]:
        kept = [x for x in items if is_gradable(x["answer"])]
    else:
        kept = list(items)
    n_dropped = n_total - len(kept)

    sp = cfg["dataset"]["split"]
    mode = sp["mode"]
    rng = random.Random(sp["seed"])

    by_group: dict = {}
    for x in kept:
        by_group.setdefault(x["group"], []).append(x)

    train, test = [], []
    if mode == "per_family":
        # Hold out a fixed count from EVERY family. RB-DERIV's README and
        # teacher-prompt.yml both require this: a global shuffle can leave a
        # technique absent from test, and technique reuse is the whole
        # experiment.
        n_hold = sp["test_per_family"]
        for g in sorted(by_group, key=lambda v: (v is None, v)):
            group = sorted(by_group[g], key=lambda x: x["problem_id"])
            rng.shuffle(group)
            if len(group) <= n_hold:
                raise ValueError(
                    f"family {g!r} has {len(group)} instances but "
                    f"test_per_family={n_hold} leaves no training data")
            test.extend(group[:n_hold])
            train.extend(group[n_hold:])
    elif mode == "stratified":
        frac = sp["test_size"] / max(len(kept), 1)
        for g in sorted(by_group, key=lambda v: (v is None, v)):
            group = sorted(by_group[g], key=lambda x: x["problem_id"])
            rng.shuffle(group)
            n_hold = max(1, round(len(group) * frac))
            test.extend(group[:n_hold])
            train.extend(group[n_hold:])
    elif mode == "native":
        end = sp.get("native_train_end", int(len(kept) * 0.75))
        train = [x for x in kept if x["_index"] < end]
        test = [x for x in kept if x["_index"] >= end]
    else:
        raise ValueError(f"unknown split mode {mode!r}")

    train.sort(key=lambda x: x["_index"])
    test.sort(key=lambda x: x["_index"])

    n_train_full, n_test_full = len(train), len(test)

    # The failure that silently invalidates a whole run. Cheap to rule out.
    train_ids = {x["problem_id"] for x in train}
    test_ids = {x["problem_id"] for x in test}
    overlap = train_ids & test_ids
    assert not overlap, f"train/test overlap on {len(overlap)} ids: {sorted(overlap)[:5]}"

    missing = set(by_group) - {x["group"] for x in test}
    report = {
        "dataset_path": str(cfg["dataset"]["path"]),
        "dataset_sha256_16": dataset_fingerprint(cfg),
        "n_total_on_disk": n_total,
        "n_excluded_ungradable": n_dropped,
        "split_mode": mode,
        "n_groups": len(by_group),
        "n_train_available": n_train_full,
        "n_test_available": n_test_full,
        "n_train_used": len(train),
        "n_test_used": len(test),
        "groups_absent_from_test": sorted(str(g) for g in missing),
        "train_groups": dict(sorted(Counter(x["group"] for x in train).items(),
                                    key=lambda kv: str(kv[0]))),
        "test_groups": dict(sorted(Counter(x["group"] for x in test).items(),
                                   key=lambda kv: str(kv[0]))),
    }

    if log:
        log.info(f"Dataset: {n_total} questions, {len(by_group)} families "
                 f"(sha {report['dataset_sha256_16']})")
        if n_dropped:
            log.info(f"  excluded (ungradable gold): {n_dropped}")
        log.info(f"  split={mode}  train={len(train)}  test={len(test)}")
        if missing:
            log.warning(f"  families absent from test: {sorted(missing)}")

    return train, test, report


def _take_per_group(items, n):
    """Trim to n items while keeping every family represented.

    A plain items[:n] would take whole families in file order and drop the
    rest entirely, which makes a trial run unrepresentative in exactly the way
    per-family splitting exists to prevent.
    """
    if n >= len(items):
        return items
    by_group: dict = {}
    for x in items:
        by_group.setdefault(x["group"], []).append(x)
    out, i = [], 0
    groups = sorted(by_group, key=lambda v: (v is None, str(v)))
    while len(out) < n:
        progressed = False
        for g in groups:
            if i < len(by_group[g]):
                out.append(by_group[g][i])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        i += 1
    out.sort(key=lambda x: x["_index"])
    return out


# ── Frozen split ─────────────────────────────────────────────────────────────
#
# The split is computed ONCE, written to disk, and reused by every stage.
#
# Without this, each command re-derives the split from config. A seed changed
# between `generate` and `evaluate`, or a switch from per_family to stratified,
# silently moves the train/test boundary -- and a problem the teacher already
# solved can land in the test set. The memory bank would then contain a
# solution to a problem the student is being scored on, which invalidates the
# whole run while every individual stage still looks like it succeeded.
#
# Freezing makes that impossible: the boundary is decided once, and any later
# change to the split settings is a loud error rather than a quiet reshuffle.

SPLIT_KEYS = ("mode", "test_per_family", "test_size", "seed")


def _split_signature(cfg):
    sp = cfg["dataset"]["split"]
    return {
        "dataset_sha256": dataset_sha256(cfg),
        "gradable_only": cfg["dataset"]["gradable_only"],
        **{k: sp.get(k) for k in SPLIT_KEYS},
    }


def split_path(cfg):
    from .config import data_path
    return data_path(cfg, "split.json")


def split(cfg, log=None, refresh=False):
    """Return (train, test, report), using the frozen split when one exists.

    Limits (`dataset.limits`) are applied as a VIEW on top of the frozen split
    and never change the stored boundary, so a trial run with --limit-train
    cannot move which problems are held out.
    """
    import json

    path = split_path(cfg)
    sig = _split_signature(cfg)

    if path.exists() and not refresh:
        stored = json.loads(path.read_text())
        if stored.get("signature") != sig:
            diffs = [k for k in sig
                     if stored.get("signature", {}).get(k) != sig[k]]
            raise SystemExit(
                f"Split settings changed since the split was frozen: {diffs}\n"
                f"  frozen: {stored.get('signature')}\n"
                f"  now:    {sig}\n"
                f"Reusing generated data across a different split would leak "
                f"test problems into the memory bank.\n"
                f"Either restore the old settings, or start a clean run:\n"
                f"  rm -rf {path.parent} && python main.py all")
        ids = {"train": stored["train_ids"], "test": stored["test_ids"]}
        items = normalize(cfg, load_raw(cfg))
        by_id = {x["problem_id"]: x for x in items}
        train = [by_id[i] for i in ids["train"] if i in by_id]
        test = [by_id[i] for i in ids["test"] if i in by_id]
        report = stored["report"]
        report["frozen"] = True
        if log:
            log.info(f"Split: frozen — {len(train)} train / {len(test)} test")
    else:
        train, test, report = _compute_split(cfg, log)
        report["frozen"] = False
        path.write_text(json.dumps({
            "signature": sig,
            "created": __import__("datetime").datetime.now().isoformat(
                timespec="seconds"),
            "train_ids": [x["problem_id"] for x in train],
            "test_ids": [x["problem_id"] for x in test],
            "report": report,
        }, indent=2))
        if log:
            log.info(f"Split: frozen to {path}")

    # Apply limits as a view on the frozen split. Done for BOTH paths -- the
    # freshly-frozen one and the reloaded one -- so the first run and every
    # later run see exactly the same problems.
    lim = cfg["dataset"]["limits"]
    if lim.get("max_train"):
        train = _take_per_group(train, lim["max_train"])
    if lim.get("max_test"):
        test = _take_per_group(test, lim["max_test"])
    report = dict(report)
    report["n_train_used"] = len(train)
    report["n_test_used"] = len(test)
    if log and (report["n_train_used"] != report.get("n_train_available")
                or report["n_test_used"] != report.get("n_test_available")):
        log.info(f"  limited by dataset.limits to {len(train)} train / "
                 f"{len(test)} test (frozen split is "
                 f"{report.get('n_train_available')}/"
                 f"{report.get('n_test_available')})")

    overlap = {x["problem_id"] for x in train} & {x["problem_id"] for x in test}
    assert not overlap, f"frozen split is corrupt: overlap {sorted(overlap)[:5]}"
    return train, test, report
