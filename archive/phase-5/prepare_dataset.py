"""
Phase 5 — Dataset Preparation (nvidia/OpenMathReasoning, standalone)

Selects 500 problems from the `additional_problems` split of
nvidia/OpenMathReasoning (31,271 candidates with problem_type ==
"has_answer_extracted"), stratified by a difficulty proxy derived from
pass_rate_72b_tir (Qwen2.5-Math-72B-Instruct pass rate — lower = harder),
into 5 levels (mirrors MATH-500's level 1-5 taxonomy, which this dataset
otherwise lacks).

100 problems per level -> 500 total.
Within each level: 80 train / 20 test -> 400 train / 100 test overall.

Output: data/dataset.json — 500 records, ordered so that
  [0:400]   = train  (pattern generation)
  [400:500] = test   (held-out eval)
matching the TRAIN_SLICE / TEST_SLICE convention used in phase-2.

Source pool: data/openmathreasoning_pool.parquet (already pulled via
duckdb from the additional_problems parquet file on the Hub, filtered to
problem_type == "has_answer_extracted").
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
POOL = ROOT / "data" / "openmathreasoning_pool.parquet"
OUT = ROOT / "data" / "dataset.json"

SEED = 42
PER_LEVEL = 100
TRAIN_PER_LEVEL = 80


def bucket_level(pass_rate: float) -> int:
    """5 = hardest (0/32 solved by 72B), 1 = easiest."""
    if pass_rate == 0.0:
        return 5
    if pass_rate <= 0.25:
        return 4
    if pass_rate <= 0.50:
        return 3
    if pass_rate <= 0.75:
        return 2
    return 1


def main():
    df = pd.read_parquet(POOL)
    df = df[df["pass_rate_72b_tir"] != "n/a"].copy()
    df["pass_rate"] = df["pass_rate_72b_tir"].astype(float)
    df["level"] = df["pass_rate"].apply(bucket_level)

    train_rows, test_rows = [], []
    for level in range(1, 6):
        pool = df[df["level"] == level].sample(n=PER_LEVEL, random_state=SEED + level)
        train_rows.append(pool.iloc[:TRAIN_PER_LEVEL])
        test_rows.append(pool.iloc[TRAIN_PER_LEVEL:])

    train_df = pd.concat(train_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

    records = []
    for i, row in pd.concat([train_df, test_df], ignore_index=True).iterrows():
        records.append({
            "id": i,
            "problem": row["problem"],
            "subject": row["problem_source"],
            "level": int(row["level"]),
            "answer": row["expected_answer"],
            "pass_rate_72b_tir": row["pass_rate"],
            "dataset": "OpenMathReasoning",
            "source_split": "additional_problems",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} records -> {OUT}")
    print(f"  Train [0:400]: {len(train_df)}")
    print(f"  Test  [400:500]: {len(test_df)}")
    print("\nLevel distribution (train):")
    print(train_df["level"].value_counts().sort_index())
    print("\nLevel distribution (test):")
    print(test_df["level"].value_counts().sort_index())
    print("\nSubject (problem_source) distribution (all 500):")
    print(pd.concat([train_df, test_df])["problem_source"].value_counts())


if __name__ == "__main__":
    main()
