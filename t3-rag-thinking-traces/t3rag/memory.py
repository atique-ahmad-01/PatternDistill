"""The memory bank, copied from the RoT project's build-memory stage.

    problem  ->  solution  ->  approach + subtasks[]

651 gold-verified teacher solutions to the 144 TRAIN problems. The bank was
asserted train-only when it was built, and retrieval adds its own leakage
guards on top. T3 rewrites each entry's full trace; the parsed approach and
steps are kept as the fallback text for an entry whose rewrite came out empty.
"""

from __future__ import annotations

from .config import data_path, read_jsonl


def _render_solution(approach: str, subtasks: list) -> str:
    """Fallback example text when an entry's T3 rewrite is empty."""
    lines = [f"APPROACH: {approach}"]
    for i, s in enumerate(subtasks, 1):
        lines.append(f"STEP {i}: {s}")
    return "\n".join(lines)


def load(cfg):
    return read_jsonl(data_path(cfg, "memory_bank.jsonl"))
