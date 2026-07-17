#!/usr/bin/env python3
"""
GPU Performance Benchmark
Tests compute throughput, memory bandwidth, and latency on one or both GPUs.

Usage:
    python gpu_bench.py          # test all available GPUs
    python gpu_bench.py --gpu 0  # test GPU 0 only
    python gpu_bench.py --gpu 1  # test GPU 1 only
"""

import argparse
import sys
import time

def check_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("ERROR: PyTorch not found. Activate your conda env first.")
        sys.exit(1)

def bench_gpu(torch, gpu_idx):
    print(f"\n{'='*60}")
    print(f"  GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
    print(f"{'='*60}")

    props = torch.cuda.get_device_properties(gpu_idx)
    total_mem = props.total_memory / 1e9
    print(f"  VRAM       : {total_mem:.1f} GB")
    print(f"  SM count   : {props.multi_processor_count}")
    print(f"  Compute cap: {props.major}.{props.minor}")

    dev = torch.device(f"cuda:{gpu_idx}")

    # --- Memory bandwidth ---
    print(f"\n  [1/3] Memory Bandwidth Test (1 GB copy)...")
    size = 1024 * 1024 * 256  # 256M float32 = 1 GB
    try:
        a = torch.ones(size, dtype=torch.float32, device=dev)
        b = torch.empty(size, dtype=torch.float32, device=dev)
        # Warmup
        b.copy_(a); torch.cuda.synchronize(gpu_idx)
        # Timed
        t0 = time.perf_counter()
        for _ in range(5):
            b.copy_(a)
        torch.cuda.synchronize(gpu_idx)
        elapsed = (time.perf_counter() - t0) / 5
        bw = (2 * size * 4) / elapsed / 1e9  # read + write, bytes -> GB/s
        print(f"  Bandwidth  : {bw:.1f} GB/s")
        del a, b
    except RuntimeError as e:
        print(f"  Bandwidth  : FAILED ({e})")

    # --- Compute throughput (FP32 GEMM) ---
    print(f"\n  [2/3] Compute Throughput Test (FP32 GEMM 4096x4096)...")
    N = 4096
    try:
        a = torch.randn(N, N, dtype=torch.float32, device=dev)
        b = torch.randn(N, N, dtype=torch.float32, device=dev)
        # Warmup
        torch.mm(a, b); torch.cuda.synchronize(gpu_idx)
        # Timed
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize(gpu_idx)
        elapsed = (time.perf_counter() - t0) / iters
        flops = 2 * N**3  # multiply-add = 2 FLOPs per element
        tflops = flops / elapsed / 1e12
        print(f"  Throughput : {tflops:.2f} TFLOPS (FP32)")
        del a, b
    except RuntimeError as e:
        print(f"  Throughput : FAILED ({e})")

    # --- BF16 GEMM ---
    print(f"\n  [3/3] Compute Throughput Test (BF16 GEMM 4096x4096)...")
    try:
        a = torch.randn(N, N, dtype=torch.bfloat16, device=dev)
        b = torch.randn(N, N, dtype=torch.bfloat16, device=dev)
        torch.mm(a, b); torch.cuda.synchronize(gpu_idx)
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize(gpu_idx)
        elapsed = (time.perf_counter() - t0) / iters
        tflops = flops / elapsed / 1e12
        print(f"  Throughput : {tflops:.2f} TFLOPS (BF16)")
        del a, b
    except RuntimeError as e:
        print(f"  Throughput : FAILED ({e})")

    # --- VRAM usage ---
    torch.cuda.synchronize(gpu_idx)
    used = torch.cuda.memory_allocated(gpu_idx) / 1e9
    reserved = torch.cuda.memory_reserved(gpu_idx) / 1e9
    print(f"\n  VRAM used  : {used:.2f} GB allocated / {reserved:.2f} GB reserved")
    print(f"  STATUS     : OK")


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to test (default: all available)")
    args = parser.parse_args()

    torch = check_torch()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs available.")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"\nPyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"GPUs    : {n_gpus} detected")

    if args.gpu is not None:
        if args.gpu >= n_gpus:
            print(f"ERROR: GPU {args.gpu} not found (only {n_gpus} GPU(s) available).")
            sys.exit(1)
        gpus_to_test = [args.gpu]
    else:
        gpus_to_test = list(range(n_gpus))

    for idx in gpus_to_test:
        try:
            bench_gpu(torch, idx)
        except Exception as e:
            print(f"\n  GPU {idx} FAILED: {e}")

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
