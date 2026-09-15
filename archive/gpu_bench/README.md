# GPU Benchmark & Monitor

Two scripts for GPU performance testing and real-time monitoring on the PatternDistill server.

## Scripts

| Script | Purpose |
|--------|---------|
| `gpu_bench.py` | One-shot benchmark: bandwidth, FP32, BF16 throughput |
| `gpu_monitor.py` | Live dashboard: GPU stats + experiment log progress |

## Setup

Activate the conda environment that has PyTorch:

```bash
conda activate ms-atique
```

---

## gpu_bench.py — Performance Benchmark

### Usage

```bash
# Test all available GPUs
python gpu_bench.py

# Test GPU 0 only (RTX 5070 Ti)
python gpu_bench.py --gpu 0

# Test GPU 1 only (RTX 5070)
python gpu_bench.py --gpu 1
```

### One-liner (no activation needed)

```bash
/home/students/miniforge3/envs/ms-atique/bin/python /home/students/gpu_bench/gpu_bench.py --gpu 0
/home/students/miniforge3/envs/ms-atique/bin/python /home/students/gpu_bench/gpu_bench.py --gpu 1
```

### What it measures

1. **Memory Bandwidth** — 1 GB device-to-device copy (GB/s)
2. **FP32 Throughput** — 4096×4096 matrix multiply (TFLOPS)
3. **BF16 Throughput** — 4096×4096 matrix multiply (TFLOPS)

### Actual results (GPU 0)

| GPU | VRAM | Bandwidth | FP32 | BF16 |
|-----|------|-----------|------|------|
| RTX 5070 Ti | 16.6 GB | 769.2 GB/s | 33.54 TFLOPS | 98.38 TFLOPS |

### Expected results (approximate)

| GPU | VRAM | BW | FP32 | BF16 |
|-----|------|----|------|------|
| RTX 5070 Ti | 16 GB | ~700 GB/s | ~45 TFLOPS | ~90 TFLOPS |
| RTX 5070    | 12 GB | ~560 GB/s | ~37 TFLOPS | ~74 TFLOPS |

---

## gpu_monitor.py — Real-time Dashboard

Live view of GPU utilization, VRAM, temperature, power, and experiment progress.

### Usage

```bash
# Auto-detects experiment log (run from PatternDistill/phase-2)
cd /home/students/PatternDistill/phase-2
/home/students/miniforge3/envs/ms-atique/bin/python /home/students/gpu_bench/gpu_monitor.py

# Point to a specific log file
/home/students/miniforge3/envs/ms-atique/bin/python /home/students/gpu_bench/gpu_monitor.py \
  --log logs/run_generate_gpu0.out

# Faster refresh (every 1 second)
/home/students/miniforge3/envs/ms-atique/bin/python /home/students/gpu_bench/gpu_monitor.py --interval 1
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | auto-detect | Path to experiment log file |
| `--interval` | 3 | Refresh rate in seconds |

### What it shows

- Per-GPU: utilization bar, VRAM usage, temperature, power draw, fan speed, clock
- Active processes: PID, VRAM used, process name
- Experiment progress: parsed from log with color-coded ✓/✗ lines and progress bar

Press `Ctrl+C` to exit.
