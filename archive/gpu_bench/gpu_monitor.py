#!/usr/bin/env python3
"""
Real-time GPU Monitor for PatternDistill
Shows live GPU stats + experiment progress.

Usage:
    python gpu_monitor.py                        # monitor all GPUs
    python gpu_monitor.py --log path/to/run.out  # also tail experiment log
    python gpu_monitor.py --interval 2           # refresh every 2s (default: 3)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

def color_pct(val, warn=70, crit=90):
    if val >= crit: return f"{RED}{val}%{RESET}"
    if val >= warn: return f"{YELLOW}{val}%{RESET}"
    return f"{GREEN}{val}%{RESET}"

def color_temp(val):
    if val >= 85: return f"{RED}{val}°C{RESET}"
    if val >= 75: return f"{YELLOW}{val}°C{RESET}"
    return f"{GREEN}{val}°C{RESET}"

def bar(pct, width=20):
    filled = int(width * pct / 100)
    b = "█" * filled + "░" * (width - filled)
    if pct >= 90:   return f"{RED}{b}{RESET}"
    if pct >= 70:   return f"{YELLOW}{b}{RESET}"
    return f"{GREEN}{b}{RESET}"

# ── nvidia-smi query ──────────────────────────────────────────────────────────
QUERY_FIELDS = [
    "index", "name", "temperature.gpu", "utilization.gpu",
    "memory.used", "memory.total", "power.draw", "power.limit",
    "fan.speed", "clocks.current.sm",
]

def get_gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(QUERY_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(QUERY_FIELDS):
                continue
            def safe_int(v):
                try: return int(float(v))
                except: return 0
            gpus.append({
                "index":    safe_int(parts[0]),
                "name":     parts[1],
                "temp":     safe_int(parts[2]),
                "util":     safe_int(parts[3]),
                "mem_used": safe_int(parts[4]),
                "mem_total":safe_int(parts[5]),
                "pwr_draw": safe_int(parts[6]),
                "pwr_limit":safe_int(parts[7]),
                "fan":      safe_int(parts[8]),
                "clock":    safe_int(parts[9]),
            })
        return gpus
    except Exception as e:
        return []

def get_gpu_processes():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_index,pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        procs = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4 or parts[0] == "[N/A]":
                continue
            procs.append({
                "gpu":  parts[0],
                "pid":  parts[1],
                "mem":  parts[2],
                "name": os.path.basename(parts[3]),
            })
        return procs
    except:
        return []

# ── log tail ──────────────────────────────────────────────────────────────────
def tail_log(path, n=8):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except:
        return []

def parse_progress(log_lines):
    """Extract [X/Y] progress from log lines."""
    for line in reversed(log_lines):
        if line.strip().startswith("[") and "/" in line:
            try:
                bracket = line.strip().split("]")[0].lstrip("[")
                cur, total = bracket.split("/")
                return int(cur), int(total)
            except:
                pass
    return None, None

# ── render ────────────────────────────────────────────────────────────────────
def render(gpus, procs, log_lines, log_path):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{BOLD}{CYAN}  PatternDistill — GPU Monitor{RESET}  {DIM}{now}{RESET}\n")

    if not gpus:
        print(f"  {RED}No GPUs found or nvidia-smi unavailable.{RESET}")
        return

    for g in gpus:
        mem_pct = int(g["mem_used"] / g["mem_total"] * 100) if g["mem_total"] else 0
        status = f"{RED}ERR{RESET}" if g["util"] == 0 and g["mem_used"] > 100 and g["temp"] == 0 else f"{GREEN}OK{RESET}"

        print(f"  {BOLD}GPU {g['index']} — {g['name']}{RESET}  [{status}]")
        print(f"  {'─'*56}")
        print(f"  Util    {bar(g['util'])}  {color_pct(g['util'])}")
        print(f"  VRAM    {bar(mem_pct)}  {g['mem_used']} / {g['mem_total']} MiB  ({mem_pct}%)")
        print(f"  Temp    {color_temp(g['temp'])}   Fan: {g['fan']}%   Clock: {g['clock']} MHz")
        print(f"  Power   {g['pwr_draw']}W / {g['pwr_limit']}W")

        gpu_procs = [p for p in procs if p["gpu"] == str(g["index"])]
        if gpu_procs:
            print(f"  {DIM}Processes:{RESET}")
            for p in gpu_procs:
                print(f"    PID {p['pid']}  {p['mem']} MiB  {DIM}{p['name']}{RESET}")
        print()

    # ── Experiment log ────────────────────────────────────────────────────────
    if log_path and log_lines:
        cur, total = parse_progress(log_lines)
        print(f"  {BOLD}Experiment Log{RESET}  {DIM}{log_path}{RESET}")
        print(f"  {'─'*56}")
        if cur and total:
            pct = int(cur / total * 100)
            print(f"  Progress  {bar(pct, width=30)}  {CYAN}{cur}/{total}{RESET} ({pct}%)")
        for line in log_lines:
            if not line.strip(): continue
            if "[WARN]" in line:
                print(f"  {YELLOW}{line}{RESET}")
            elif "pattern=✗" in line:
                print(f"  {RED}{line}{RESET}")
            elif "pattern=✓" in line:
                print(f"  {GREEN}{line}{RESET}")
            else:
                print(f"  {DIM}{line}{RESET}")
        print()

    print(f"  {DIM}Press Ctrl+C to exit{RESET}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time GPU Monitor for PatternDistill")
    parser.add_argument("--log",      type=str,   default=None, help="Path to experiment log file to tail")
    parser.add_argument("--interval", type=float, default=3.0,  help="Refresh interval in seconds (default: 3)")
    args = parser.parse_args()

    # Auto-detect latest log if not specified
    log_path = args.log
    if not log_path:
        candidates = [
            "logs/run_generate_gpu0.out",
            "logs/run_generate_full.out",
            "logs/run_generate_resume.out",
        ]
        for c in candidates:
            if os.path.exists(c):
                log_path = c
                break

    print(f"Starting monitor... (Ctrl+C to stop)")
    time.sleep(0.5)

    try:
        while True:
            gpus      = get_gpu_stats()
            procs     = get_gpu_processes()
            log_lines = tail_log(log_path, n=8) if log_path else []
            render(gpus, procs, log_lines, log_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")

if __name__ == "__main__":
    main()
