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
    for key in ("data_dir", "baseline_dir", "results_dir", "logs_dir"):
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

    if cfg["retrieval"]["top_k"] < 1:
        raise ValueError("retrieval.top_k must be >= 1")

    tc = cfg.get("t3")
    if not tc:
        raise ValueError("config needs a t3 section (t3.variant)")
    variants = ("struct", "semantic", "reflect")
    if tc["variant"] not in variants:
        raise ValueError(f"t3.variant must be one of {variants}, "
                         f"got {tc['variant']!r}")


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
