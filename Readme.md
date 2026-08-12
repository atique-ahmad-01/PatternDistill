# PatternDistill

> Can we extract reasoning patterns from an 8B model and transfer them to a 1.7B model so it solves problems similarly?

## Research Question

Does **neural memory distillation** (fine-tuning patterns into weights) outperform **memory injection** (feeding patterns in-context at inference) for a small model — and by how much?

---

## Hardware

| | |
|---|---|
| **Development / script testing** | Mac (local) — write scripts, smoke-test on small subsets |
| **Full experiment runs** | `murtaza-MS-7E12` — Ubuntu, NVIDIA RTX 5070 Ti 16GB VRAM, CUDA 13.2 |

```
nvidia-smi summary:
  GPU 0 — RTX 5070 Ti   16303 MiB  ← primary, use this for all runs
  GPU 1 — RTX 5070      12227 MiB  
```

Scripts accept a `--max-problems N` flag so they can be smoke-tested locally (N=5) before full runs on the GPU machine.

---

## Connecting to the GPU Machine

Connect via Tailscale (<gpu-host>):

```bash
sudo brew services start tailscale && sudo tailscale up
ssh students@<gpu-host>
```

Sync your local code to the GPU machine:

```bash
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
  ~/Desktop/PatternDistill/ \
  students@<gpu-host>:~/PatternDistill/
```

Pull results back to your Mac after a run:

```bash
rsync -avz \
  students@<gpu-host>:~/PatternDistill/results/ \
  ~/Desktop/PatternDistill/results/
```

To make sure PyTorch uses GPU 0 (RTX 5070 Ti) and not the ERR GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Add this to the top of any training or inference run.

---

## Models

| Role | Model |
|---|---|
| Teacher | `qwen3:8b` (via Ollama) |
| Student | `qwen3:1.7b` (via Ollama) |
| Embeddings | `nomic-embed-text` (via Ollama) |

---

## Dataset

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
```

500 competition math problems across 7 subjects, difficulty levels 1–5.
Split: first 400 used to build memory bank / fine-tuning data, last 100 held out for evaluation.

---

## Full Experiment Plan

### Phase 0 — Environment Setup

- Verify Ollama is running at `localhost:11434`
- Pull models: `qwen3:8b`, `qwen3:1.7b`, `nomic-embed-text`
- Install Python dependencies (see `requirements.txt`)
- Confirm CUDA is available on GPU machine (`nvidia-smi`)

---

### Phase 1 — Vanilla Baseline

**Goal:** Establish the performance gap between 1.7B and 8B before any pattern work.

Run both models on the full MATH500 test set with no patterns, no fine-tuning — raw inference only.

**Measure per model:**

| Metric | Description |
|---|---|
| Overall accuracy | % correct answers |
| Accuracy by subject | Per each of 7 math subjects |
| Accuracy by difficulty | Per level 1–5 |
| Latency per problem | Seconds from prompt to response |
| Output tokens | Avg tokens in model response |
| Tokens/sec | Throughput |

**Script:** `scripts/baseline.py`

**Output:** `results/baseline/summary.json`

**Expected outcome:** 8B meaningfully outperforms 1.7B — this gap is what we attempt to close.

---

### Phase 2 — Teacher Pattern Generation

**Goal:** Use the 8B teacher to label 400 problems with reasoning patterns.

Run `generate_patterns.py` on MATH500[0:400]. For each problem the 8B model produces:

```
<cot>   step-by-step reasoning   </cot>
<pattern>   1-2 sentence abstract rule   </pattern>
<answer>   final answer   </answer>
```

**Script:** `method_a/scripts/generate_patterns.py`

**Output:** `method_a/data/patterns/patterns.json`

```json
[
  {
    "problem": "...",
    "subject": "Algebra",
    "level": 3,
    "gold_answer": "...",
    "cot": "...",
    "pattern": "...",
    "teacher_answer": "..."
  }
]
```

This file feeds both Method A (memory bank) and Method B (fine-tuning).

---

### Phase 3 — Method A: Memory Injection

**Hypothesis:** Injecting a retrieved reasoning pattern into the 1.7B student's prompt at inference time will improve its accuracy.

#### Phase 3A — Build Memory Bank

Run `build_memory_bank.py`. Embeds each problem using `nomic-embed-text`, stores vectors in a FAISS index for cosine similarity retrieval.

**Script:** `method_a/scripts/build_memory_bank.py`

**Output:**
- `method_a/memory/bank.json` — pattern metadata
- `method_a/memory/bank.index` — FAISS vectors

#### Phase 3B — Inference + Evaluation

Run `inject_and_eval.py` on MATH500[400:] (100 held-out problems).

For each problem:
1. Embed the query problem
2. Retrieve the most similar pattern from the bank (top-1)
3. Run 1.7B **without** pattern (vanilla condition)
4. Run 1.7B **with** pattern injected into prompt (injection condition)
5. Record correctness + latency for both

**Script:** `method_a/scripts/inject_and_eval.py`

**Output:** `results/method_a/results.json`, `results/method_a/summary.json`

---

### Phase 4 — Method B: Neural Memory Distillation

**Hypothesis:** Fine-tuning the 1.7B student on pattern-augmented examples internalizes reasoning patterns into its weights, allowing it to generalize without any retrieval at inference time.

#### Phase 4A — Fine-tune Student

Run `finetune.py`. Fine-tunes Qwen 1.7B using **LoRA/QLoRA** via `trl` SFTTrainer + `peft` on the 400 examples from Phase 2.

Training format:
```
Pattern: {extracted pattern}
Problem: {problem}
Answer: {answer}
```

**Script:** `method_b/scripts/finetune.py`

**Config:** `method_b/configs/lora_config.yaml`

**Output:** `method_b/checkpoints/` — LoRA adapter weights

#### Phase 4B — Evaluate Fine-tuned Student

Run `evaluate.py` on the same 100 held-out problems. **No pattern injected** — model must rely solely on internalized patterns from fine-tuning.

**Script:** `method_b/scripts/evaluate.py`

**Output:** `results/method_b/results.json`, `results/method_b/summary.json`

---

### Phase 5 — Final Comparison

Aggregate all results into a single table across all 4 conditions:

| Condition | Description |
|---|---|
| **1.7B vanilla** | Student, no patterns, no fine-tuning (Phase 1) |
| **Method A** | Student + memory injection at inference (Phase 3) |
| **Method B** | Student + neural distillation, no injection (Phase 4) |
| **8B vanilla** | Teacher ceiling, no modifications (Phase 1) |

Reported metrics per condition:

| Metric | Description |
|---|---|
| Overall accuracy | % correct on 100 held-out problems |
| Accuracy by subject | Per each of 7 subjects |
| Accuracy by difficulty | Per level 1–5 |
| Latency per problem | Seconds |
| Avg output tokens | Reasoning length |
| Tokens/sec | Throughput |
| Injection vs distillation gap | Delta between Method A and Method B |

**Script:** `analysis/compare.py`

---

### Phase 6 — Analysis

Answer the 3 core research questions using Phase 5 results:

**Q1.** Does Method B (distillation) outperform Method A (injection) on MATH500?

**Q2.** Does Method B generalize to problem types **not seen** during fine-tuning?

**Q3.** At what difficulty level does Method A (injection) break down?

Additional analysis:
- Pattern reuse rate — does the student correctly apply the given pattern?
- Retrieval similarity vs correctness correlation — does a closer match lead to better injection?
- Token efficiency — accuracy per token used across all methods

**Notebook:** `analysis/analysis.ipynb`


---

## Results

### Phase 1 — Vanilla Baseline ✓ Complete

Run on MATH-500 full test set (500 problems). Hardware: NVIDIA RTX 5070 Ti (GPU 0), Ollama inference via `localhost:11434`.

| Model | Accuracy | Correct / Total | Avg Latency (s) | Avg Output Tokens | Tokens/sec |
|---|---|---|---|---|---|
| Qwen 1.7B | **50.8%** | 254 / 500 | 48.6s | 4,688 | 96.5 |
| Qwen 8B | **71.8%** | 359 / 500 | 58.6s | 4,218 | 72.0 |
| **Gap (8B − 1.7B)** | **+21.0 pp** | +105 problems | — | — | — |

#### Accuracy by Subject

| Subject | Qwen 1.7B | Qwen 8B | Gap |
|---|---|---|---|
| Algebra | 73.4% | **89.5%** | +16.1 pp |
| Prealgebra | 59.8% | **78.1%** | +18.3 pp |
| Number Theory | 53.2% | **75.8%** | +22.6 pp |
| Counting & Probability | 39.5% | **68.4%** | +28.9 pp |
| Geometry | 43.9% | **46.3%** | +2.4 pp ← both models weak |
| Intermediate Algebra | 30.9% | **61.9%** | +31.0 pp |
| Precalculus | 32.1% | **57.1%** | +25.0 pp |

#### Accuracy by Difficulty Level

| Level | Description | Qwen 1.7B | Qwen 8B | Gap |
|---|---|---|---|---|
| L1 | Easiest | 74.4% | **90.7%** | +16.3 pp |
| L2 | Easy | 62.2% | **83.3%** | +21.1 pp |
| L3 | Medium | 61.9% | **79.1%** | +17.2 pp |
| L4 | Hard | 48.4% | **71.9%** | +23.5 pp |
| L5 | Hardest | 29.1% | **52.2%** | +23.1 pp |

> Visualisation: `results/baseline/histogram.png` — accuracy, latency, and throughput by subject and difficulty level.

**Key findings:**
- Geometry is a weakness for **both** models (44% vs 46%) — distillation may have limited headroom here
- The largest gaps are in Intermediate Algebra (+31pp) and Precalculus (+25pp) — prime targets for distillation
- 1.7B collapses at L5 (29%) vs 8B's 52% — hardest problems are where patterns should help most
- 1.7B is ~34% faster in throughput (96.5 vs 72.0 tok/s) despite worse accuracy — smaller model advantage

### Phase 5 — Full Comparison

| Condition | Accuracy | Δ vs 1.7B baseline | Avg Latency (s) |
|---|---|---|---|
| 1.7B vanilla | 50.8% | — | 48.6s |
| Method A (injection) | — | — | — |
| Method B (distillation) | — | — | — |
| 8B vanilla (ceiling) | 71.8% | +21.0 pp | 58.6s |

---

## Project Structure

```
PatternDistill/
├── scripts/
│   ├── baseline.py                  # Phase 1: vanilla eval both models
│   ├── analyze.py                   # CLI tool: inspect results, wrong answers
│   └── plot_baseline.py             # Generate histogram.png (accuracy/latency/throughput)
├── method_a/
│   ├── config.yaml
│   ├── requirements.txt
│   ├── data/patterns/patterns.json  # Phase 2 output
│   ├── memory/
│   │   ├── bank.json                # Phase 3A output
│   │   └── bank.index               # Phase 3A output
│   └── scripts/
│       ├── generate_patterns.py     # Phase 2
│       ├── build_memory_bank.py     # Phase 3A
│       └── inject_and_eval.py       # Phase 3B
├── method_b/
│   ├── configs/lora_config.yaml
│   ├── requirements.txt
│   ├── checkpoints/                 # Phase 4A output
│   └── scripts/
│       ├── finetune.py              # Phase 4A
│       └── evaluate.py              # Phase 4B
├── analysis/
│   ├── compare.py                   # Phase 5
│   └── analysis.ipynb               # Phase 6
├── results/
│   ├── baseline/
│   ├── method_a/
│   └── method_b/
└── Readme.md
```

---

## Running Order

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3A  →  Phase 3B  ─┐
                                →  Phase 4A  →  Phase 4B  ─┤
                                                             ↓
                                                        Phase 5
                                                             ↓
                                                        Phase 6
```

Phases 3 and 4 can run in parallel after Phase 2 completes.

---

## Development Workflow

1. Write and test each script locally with `--max-problems 5` (smoke test, CPU)
2. Once the script runs end-to-end without errors, transfer to the GPU machine
3. Run at full scale (500 problems) on the 16GB VRAM machine
4. Pull results back locally for analysis
