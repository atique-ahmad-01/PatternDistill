"""
Phase 7 — RB-DERIV: shared configuration.

The benchmark data lives here, alongside the code. The prompt specs still live
in ../sep-paper/, so a couple of paths reach across. Resolve them here, once,
so no stage hard-codes a relative hop of its own.

Constants below are transcribed from the specs, not invented:
  ../sep-paper/teacher-prompt.yml   split, sampling, distill gates
  ../sep-paper/solver-prompt.yml    arms, parsing, scoring, confounds
Where a spec and this file disagree, the spec wins — fix it here.
"""

import pathlib

# ── Paths ────────────────────────────────────────────────────────────────────

PHASE = pathlib.Path(__file__).resolve().parent
REPO = PHASE.parent
PAPER = REPO / "sep-paper"

RB_DERIV = PHASE / "RB_DERIV"
QUESTIONS = RB_DERIV / "RB_DERIV_questions.json"
OBJECTS = RB_DERIV / "RB_DERIV_objects.json"
TEACHER_PROMPT = PAPER / "teacher-prompt.yml"
SOLVER_PROMPT = PAPER / "solver-prompt.yml"

DATA = PHASE / "data"
LOGS = PHASE / "logs"
RESULTS = PHASE / "results"

# Fail loudly at import if the sibling moved, rather than three stages later
# with an empty result set.
for _p in (QUESTIONS, OBJECTS, TEACHER_PROMPT, SOLVER_PROMPT):
    assert _p.exists(), f"missing input: {_p}"

# ── Models ───────────────────────────────────────────────────────────────────

TEACHER_MODEL = "Qwen/Qwen3-8B"      # 4-bit; 16.6GB card, see phase-2d/NOTES.md
STUDENT_MODEL = "Qwen/Qwen3-1.7B"

# ── Split (teacher-prompt.yml: "per-family 9 train / 3 test, NOT a global
# 75/25 shuffle" — all 16 techniques must appear in test) ────────────────────

TEST_PER_FAMILY = 3
SPLIT_SEED = 0

# Rotating the held-out 3 gives every question one turn as test: n=192 instead
# of n=48, and it fixes the false-friend denominator at the same time. Costs 4x
# student decode, which is cheap here (48 x 6 arms x 4 = 1,152 calls).
# Left off by default — turning it on is a methodology decision, not a flag.
SPLIT_ROTATIONS = 1   # set to 4 for full rotation

# ── Teacher: solve (teacher-prompt.yml stage 1) ──────────────────────────────

N_SAMPLES = 5
SOLVE_TEMPERATURE = 0.8
MIN_VERIFIED_PER_FAMILY = 3   # below this the family is teacher-limited

# ── Teacher: distill (teacher-prompt.yml stage 2) ────────────────────────────

DISTILL_UNIT = "family"       # 16 objects, one per family
GATE_MAX_RETRIES = 2

# ── Student arms (solver-prompt.yml) ─────────────────────────────────────────
#
# schema_oracle is in the architecture figure but has no template in
# solver-prompt.yml yet — it reuses the `schema` template with the 16
# human-authored objects shipped in RB_DERIV_objects.json.

ARMS = {
    "base":          {"inject": None,           "contract": False},
    "plain":         {"inject": "method_only",  "contract": False},
    "schema":        {"inject": "distilled",    "contract": True},
    "schema_oracle": {"inject": "shipped",      "contract": True},
    "schema_wrong":  {"inject": "false_friend", "contract": True},
    "schema_random": {"inject": "off_family",   "contract": True},
}

# ── Generation ───────────────────────────────────────────────────────────────

MAX_NEW_TOKENS = 24576        # phase-6 value; 8,192 truncated 74% of AIME runs
STUDENT_TEMPERATURE = 0.0     # answers are single integers; no sampling noise

# ── Scoring (solver-prompt.yml) ──────────────────────────────────────────────

ANSWER_RE = r"ANSWER:\s*(-?\d+)\s*$"     # LAST match, MULTILINE
CHECK_RE = r"^CHECK:\s*(PASS|FAIL)\b"    # LAST match; absent in base/plain
ON_MISSING_ANSWER = "score_incorrect"    # never retry — retries break token accounting

# Rescue is scored ONLY where the corruption actually changes the answer.
# 175/192 on the full set; on a single 9/3 split the test-set count is 46/48.
# Compute it from the split in use — do not hard-code either number.
FF_DENOMINATOR = "false_friend_changes_answer == True, within the scored split"

# Families where guessing beats chance (solver-prompt.yml:151-154). Report
# per-family; do not let these carry an aggregate claim.
LOW_DIVERSITY_FAMILIES = {"digital_root", "zeckendorf", "xor_range", "fib_huge"}
