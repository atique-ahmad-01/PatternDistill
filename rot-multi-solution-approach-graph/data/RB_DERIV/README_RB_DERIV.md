# Simple details

  

This is a benchmark for testing whether an AI model can reuse a math shortcut that's handed to it, instead of trying to figure out the shortcut from scratch.

  
## The core idea

Each of the 192 questions is a math problem with huge numbers — numbers so large you can't just brute-force the answer by computing step-by-step. To solve them, you need to know a clever trick or formula. For example:

- "Apply this rule a trillion times — what's the result?" (you need a closed-form formula)
- "What's the 1,000,000,000,000th Fibonacci number mod 1000?" (you need a fast-doubling trick)
- "1000 people in a circle, eliminate every 2nd — who survives?" (you need the Josephus formula)

There are 16 families of problems (12 questions each), covering things like game theory, number theory, combinatorics, etc.

  

## The experiment

The AI is tested under 4 conditions ("arms"):

  
1. **base** — just the question, no help. The model must figure out the trick on its own.
2. **plain** — the correct shortcut/formula is given, but without structure.
3. **schema** — the shortcut is given in a structured format with: a name, when to use it, how to use it, and how to verify it worked.
4. **schema_wrong** — a deliberately wrong shortcut is given, but with an honest self-check that would catch the error if the model actually runs the check.

  

## What it measures

- **Can giving the AI a known formula save computation?** (Does it generate fewer tokens when it doesn't have to re-derive the trick?)
- **Does accuracy hold up** when using the provided formula vs. figuring it out alone?
- **Can the AI catch a wrong formula** by using the provided self-check? (The "false friend" arm tests this— if the model blindly trusts the wrong formula, it gets the wrong answer. If it actually runs the CHECK step, it can detect the error.)

## Why "derivation-deep"

The key design principle: these problems are hard because of the thinking required to find the right approach, not because of the calculation after you know the approach. Once you know the formula, plugging in the numbers is trivial (1-3 operations). This means that giving the AI the shortcut can genuinely save work — you're removing the need to discover the trick, not the need to execute it.


# RB-DERIV — a derivation-deep reuse benchmark (handoff package)


Two files are the whole deliverable:

| file | contents |
|---|---|
| `RB_DERIV_objects.json` | 16 reusable technique **objects** |
| `RB_DERIV_questions.json` | 192 **questions** with exact integer golds |

`rb_deriv_run.py` is a reference runner; it is not part of the data.

## Design rule

Every family scales the **derivation** the solver must perform — it must discover a principle it cannot
brute-force (a cycle, a period, a closed form) — while the **execution** after the technique is known is
1–3 operations.

This matters because injected knowledge can remove derivation and search but **never execution**. Families
whose cost is execution (fill a DP table, sieve, simulate a stack) cannot show a saving no matter how
expensive they are, and are excluded by construction. Two earlier benchmark attempts failed precisely by
scaling execution instead of derivation.

## Object format

```json
{
  "object_id": "josephus",
  "schema": { "NAME": ..., "USE_WHEN": ..., "METHOD": ..., "CHECK": ... },
  "false_friend": { "METHOD": ..., "CHECK": ...,
                    "changes_answer_on": [1,1,0,...], "n_changed": 12 }
}
```

- `schema` is the object injected at use time. `USE_WHEN` is a precondition the solver applies to itself;
  `CHECK` is a cheap, decisive verification of that method's own failure mode.
- `false_friend` is a plausible-but-**wrong** METHOD paired with an *honest* CHECK — one that would catch
  the error if actually performed. It is for use-time gate experiments.
- `changes_answer_on` marks the instances where the corruption genuinely changes the answer.
  **Score false-friend rescue only on those instances** (175/192 here). A "wrong" method that happens to
  agree with the truth on the sampled instances measures nothing.

## Question format

```json
{ "query_id": "q0001", "family": "lcg_closed", "question": "...",
  "gold_answer": 968, "uses_object": "lcg_closed",
  "false_friend_answer": 493, "false_friend_changes_answer": true }
```

Golds are computed exactly in Python (`rb_deriv_build.py`); every answer is a single integer.

## Families (16 × 12 instances)

`lcg_closed`, `perm_power`, `grundy_134`, `grundy_124`, `grundy_23`, `wythoff`, `power_tower`,
`xor_range`, `josephus`, `frobenius`, `fib_huge`, `digital_root`, `bitcount_range`,
`necklace_burnside`, `zeckendorf`, `crt_three`.

The three `grundy_*` families share one technique type at different periods — useful for testing
within-technique recurrence.

## Audits already applied (please preserve if you extend it)

1. **Answer diversity** — per-family distinct-gold counts are reported by the builder. `digital_root`
   (6/12) and `zeckendorf` (6/12) are intrinsically low-range; the rest are 8–12/12.
2. **No identity answers** — instances whose answer equals a trivially-echoed input were removed
   (step counts that were multiples of a cycle length made 3/5 LCG and 2/5 permutation answers equal
   their own input; a solver replying "unchanged" would have scored 60% free).
3. **Non-benign false friends** — every corruption is checked against every instance. Three families
   originally had corruptions that were arithmetically identical to the correct method and were rewritten.
4. **Discriminating checks** — a `CHECK` whose test case cannot separate the correct method from the
   corrupted one is useless; check cases were chosen to separate them.

## Reporting requirements

- Always pair `saving` with the `accuracy` delta from the same cell. A saving with collapsing accuracy is
  the solver abandoning the problem, not a saving — one earlier run showed +66.6% "saving" while accuracy
  fell from 8/8 to 1/8.
- Do **not** stratify saving by measured base decode. `saving = (base − memory)/base` sorted on `base`
  produces a strong monotone gradient at *zero* true effect (regression to the mean). Stratify on a
  generator-side difficulty measure instead, and report the same table for a random-object arm as placebo.
