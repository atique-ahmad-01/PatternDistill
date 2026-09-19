"""Configuration loading.

One YAML file drives the whole pipeline. Paths inside it are resolved relative
to the config file's own directory, so the CLI works from anywhere.

Every stage writes the resolved config alongside its output. A result set that
cannot name the settings that produced it is not reproducible, and that is the
single most common way a run becomes unusable three weeks later.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import yaml


class Config(dict):
    """dict with attribute access and dotted lookup."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path="config.yaml") -> Config:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    cfg = Config(raw)
    root = p.parent

    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    cfg["dataset"]["path"] = str((root / raw["dataset"]["path"]).resolve())
    for key in ("data_dir", "results_dir", "logs_dir"):
        cfg["paths"][key] = str((root / raw["paths"][key]).resolve())

    _validate(cfg)
    return cfg


def _validate(cfg):
    ds = cfg["dataset"]
    if not Path(ds["path"]).exists():
        raise FileNotFoundError(
            f"dataset not found: {ds['path']}\n"
            "Set dataset.path in config.yaml to an existing dataset file."
        )
    mode = ds["split"]["mode"]
    valid_modes = ("per_family", "stratified", "native")
    if mode not in valid_modes:
        raise ValueError(f"dataset.split.mode must be one of {valid_modes}, got {mode!r}")

    rmode = cfg["retrieval"]["mode"]
    valid = ("full_solution", "approach", "subtasks", "approach_subtasks")
    if rmode not in valid:
        raise ValueError(f"retrieval.mode must be one of {valid}, got {rmode!r}")

    inc = cfg["memory"]["include"]
    if inc not in ("correct_only", "all"):
        raise ValueError(f"memory.include must be correct_only|all, got {inc!r}")

    if not 0.0 < cfg["graph"]["tau_semantic"] <= 1.0:
        raise ValueError("graph.tau_semantic must be in (0, 1]")
    if cfg["retrieval"]["top_k"] < 1:
        raise ValueError("retrieval.top_k must be >= 1")

    rc = cfg.get("rot")
    if rc:
        # Both thresholds are RAW cosine, matching the reference implementation
        # (the paper's rescaled sim~ is not what its code uses). A value at or
        # below 0.5 is almost always graph.tau_semantic pasted across, or the
        # rescaled figure from the paper -- either makes nearly every pair an
        # edge and the traversal never terminates on similarity.
        for key in ("tau_semantic", "sim_threshold"):
            if not 0.5 < rc[key] <= 1.0:
                raise ValueError(
                    f"rot.{key} is RAW cosine and must be in (0.5, 1.0]; got "
                    f"{rc[key]}. The reference uses tau_semantic=0.85 and "
                    f"sim_threshold=0.80. Do not copy graph.tau_semantic here, "
                    f"and do not use the paper's rescaled 0.85 as a raw value.")
        for key in ("w_first_query", "w_first_s0", "w_trav_query", "w_trav_flow"):
            if not 0.0 <= rc[key] <= 1.0:
                raise ValueError(f"rot.{key} must be in [0, 1]")
        if rc["l_max"] < 1:
            raise ValueError("rot.l_max must be >= 1")
        if rc["max_neighbors"] < 0:
            raise ValueError("rot.max_neighbors must be >= 0 (0 = uncapped)")
        if rc["tag_source"] not in ("gold", "predicted", "none"):
            raise ValueError("rot.tag_source must be gold|predicted|none, "
                             f"got {rc['tag_source']!r}")
        if rc["injection"] not in ("ti_prefill", "user_prompt"):
            raise ValueError("rot.injection must be ti_prefill|user_prompt, "
                             f"got {rc['injection']!r}")


def set_seeds(seed: int):
    """Seed every source of randomness the pipeline touches."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def snapshot(cfg, out_dir):
    """Write the resolved config next to a result set."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = copy.deepcopy(dict(cfg))
    (out / "config.snapshot.yaml").write_text(
        yaml.safe_dump(clean, sort_keys=False, default_flow_style=False))
    return out / "config.snapshot.yaml"


def data_path(cfg, *parts) -> Path:
    p = Path(cfg["paths"]["data_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def results_path(cfg, *parts) -> Path:
    p = Path(cfg["paths"]["results_dir"]).joinpath(cfg.get("run_name", "default"), *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path, rows, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
