"""
Phase 6 — Shared configuration.

Every threshold lives here in one place, because the phase-4 dead-code bug was
caused by exactly the opposite: TAU_SEMANTIC lived in the graph builder and
NOVELTY_THRESHOLD lived in the evaluator, they drifted into contradiction, and
nothing checked. Graph augmentation was silently impossible for three phases.

The invariant below is asserted at import time, so that class of bug cannot
recur silently.
"""

# ── Graph / retrieval ────────────────────────────────────────────────────────

TAU_SEMANTIC = 0.90        # similarity (rescaled to [0,1]) required for an edge
NOVELTY_THRESHOLD = 0.95   # reject an augmentation candidate at/above this

# A candidate reaches the evaluator only via an edge, so it always scores
# >= TAU_SEMANTIC. If NOVELTY_THRESHOLD <= TAU_SEMANTIC every candidate is
# rejected by construction and augmentation is dead code.
#
# Measured on the phase-3d graph (tau=0.90):
#   novelty 0.85 ->     0 candidates survive (the shipped bug)
#   novelty 0.90 ->     0 candidates survive
#   novelty 0.95 -> 2,219 candidates survive, 256/400 anchors augmented
assert NOVELTY_THRESHOLD > TAU_SEMANTIC, (
    f"NOVELTY_THRESHOLD ({NOVELTY_THRESHOLD}) must exceed TAU_SEMANTIC "
    f"({TAU_SEMANTIC}), otherwise every augmentation candidate is rejected "
    f"by construction and the graph contributes nothing."
)

MAX_BONUS_STEPS = 1        # one appended step, never a merge of equals

# Per-subject cosine cutoffs, carried over from phase-2d. These were tuned
# against contaminated patterns, so phase-6 re-measures them rather than
# treating them as settled.
SIM_THRESHOLDS = {
    "MATH-500": {
        "Intermediate Algebra": 0.60,
        "Number Theory":        0.65,
        "Precalculus":          0.65,
        "Geometry":             0.55,
        "default":              0.50,
    },
    "AIME": {"default": 0.50},
}

SAME_SOURCE_BIAS     = 0.05   # prefer an anchor from the same dataset
CROSS_THRESHOLD_BUMP = 0.10   # demand more from a cross-dataset anchor

# ── Generation budget ────────────────────────────────────────────────────────

# 24,576 (up from phase-4's 8,192) applied to BOTH arms. Phase-4 truncated 74%
# of AIME vanilla runs, and 111 of 117 AIME both-wrong cases had both arms
# truncated, which made the comparison uninterpretable. Raising only the
# injection arm would manufacture a win, so this is deliberately shared.
MAX_NEW_TOKENS = 24576
EARLY_EXIT_ON_BOXED = True

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# ── Ablation arms ────────────────────────────────────────────────────────────

# Arm B is the essential control: it isolates "we gave the model room to think"
# from "our pattern redesign worked". Without it the budget fix masks
# everything else.
ARMS = {
    "A_vanilla":        {"inject": False},
    "B_old_patterns":   {"inject": True,  "bank": "old", "graph": False, "example": False, "escape": True},
    "C_new_patterns":   {"inject": True,  "bank": "new", "graph": False, "example": False, "escape": True},
    "D_new_plus_graph": {"inject": True,  "bank": "new", "graph": True,  "example": False, "escape": True},
    "E_new_plus_ex":    {"inject": True,  "bank": "new", "graph": False, "example": True,  "escape": True},
    "F_no_escape":      {"inject": True,  "bank": "new", "graph": False, "example": False, "escape": False},
}
