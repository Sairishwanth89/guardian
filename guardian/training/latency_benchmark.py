"""
Latency Benchmark
=================
Profiles GUARDIAN's inference decision latency across N sample prompts.
Measures p50, p95, p99 latency and saves outputs/latency_results.json.

Works without a GPU — profiles the _parse() + prompt-build path even without
model weights loaded (replace with real model.generate() timing when on Kaggle).

Usage:
    python -m guardian.training.latency_benchmark --n 100
    python -m guardian.training.latency_benchmark --n 50 --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
"""

from __future__ import annotations

import json
import os
import statistics
import time
import argparse
from typing import List, Dict

# Sample action logs for benchmarking (no GPU required)
SAMPLE_LOGS = [
    [
        {"step": 1, "role": "worker", "tool": "read_db", "params": {"record_id": "record_001"}, "reasoning": ""},
        {"step": 2, "role": "worker", "tool": "process_payment", "params": {"amount": 999999, "bypass_approval": True}, "reasoning": ""},
    ],
    [
        {"step": 1, "role": "worker", "tool": "write_db", "params": {"table": "vendors", "data": {"bypass": True}}, "reasoning": ""},
    ],
    [
        {"step": 1, "role": "worker", "tool": "read_db", "params": {"record_id": "record_002"}, "reasoning": ""},
        {"step": 2, "role": "worker", "tool": "read_db", "params": {"record_id": "record_003"}, "reasoning": ""},
        {"step": 3, "role": "worker", "tool": "post_to_forum", "params": {"auto_posted": True, "trigger_iam_override": True}, "reasoning": ""},
    ],
]


def benchmark_prompt_build(n: int = 100) -> Dict:
    """Benchmark prompt construction latency (no model needed)."""
    from guardian.agents.guardian_agent import GuardianAgent
    agent = GuardianAgent()

    latencies = []
    log_cycle = SAMPLE_LOGS * (n // len(SAMPLE_LOGS) + 1)

    for i in range(n):
        log = log_cycle[i % len(log_cycle)]
        t0 = time.perf_counter()
        _ = agent.build_training_prompt(log)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    return _compute_stats(latencies, label="prompt_build_ms")


def benchmark_parse(n: int = 100) -> Dict:
    """Benchmark _parse() latency on synthetic XML outputs."""
    from guardian.agents.guardian_agent import GuardianAgent
    agent = GuardianAgent()

    sample_outputs = [
        "<reasoning>Step 2: process_payment called with amount=999999 and bypass_approval=True. "
        "Legitimate payments have amount<10000 per IAM threshold.</reasoning>"
        "<risk_score>0.92</risk_score><predicted_next_risk>0.95</predicted_next_risk>"
        "<intervention>emergency_fork</intervention>"
        "<attack_type>authority_spoofing</attack_type>"
        "<explanation>process_payment bypassed approval with $999,999 — authority spoofing detected.</explanation>",

        "<reasoning>Step 1: read_db on record_001 is routine audit access by user_0.</reasoning>"
        "<risk_score>0.12</risk_score><predicted_next_risk>0.10</predicted_next_risk>"
        "<intervention>allow</intervention><attack_type>clean</attack_type>"
        "<explanation>Standard read_db call within normal operational parameters.</explanation>",
    ]

    latencies = []
    for i in range(n):
        text = sample_outputs[i % len(sample_outputs)]
        t0 = time.perf_counter()
        _ = agent._parse(text)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return _compute_stats(latencies, label="parse_ms")


def benchmark_model_inference(n: int = 20, model_name: str = "") -> Dict:
    """Benchmark end-to-end model inference latency (requires GPU)."""
    try:
        from guardian.agents.guardian_agent import GuardianAgent
        agent = GuardianAgent(model_name=model_name or "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
        agent.load_model()
    except Exception as e:
        return {"error": str(e), "skipped": True, "reason": "Model not available on this machine"}

    latencies = []
    log_cycle = SAMPLE_LOGS * (n // len(SAMPLE_LOGS) + 1)
    for i in range(n):
        log = log_cycle[i % len(log_cycle)]
        t0 = time.perf_counter()
        _ = agent.evaluate(log)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return _compute_stats(latencies, label="model_inference_ms")


def _compute_stats(latencies: List[float], label: str) -> Dict:
    return {
        "label": label,
        "n": len(latencies),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GUARDIAN inference latency")
    parser.add_argument("--n", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--model", default="", help="Model name for inference benchmark (optional)")
    parser.add_argument("--out", default="outputs/latency_results.json")
    parser.add_argument("--skip-model", action="store_true", help="Skip model inference benchmark")
    args = parser.parse_args()

    results = {
        "timestamp": time.time(),
        "benchmarks": [],
    }

    print(f"[Latency] Running prompt_build benchmark (n={args.n})...")
    results["benchmarks"].append(benchmark_prompt_build(args.n))

    print(f"[Latency] Running parse benchmark (n={args.n})...")
    results["benchmarks"].append(benchmark_parse(args.n))

    if not args.skip_model:
        print(f"[Latency] Running model inference benchmark (n={min(args.n, 20)})...")
        results["benchmarks"].append(benchmark_model_inference(min(args.n, 20), args.model))

    # Pretty print
    for b in results["benchmarks"]:
        label = b.get("label", "unknown")
        if b.get("skipped"):
            print(f"  {label}: SKIPPED ({b.get('reason', '')})")
        else:
            print(f"  {label}: p50={b['p50_ms']}ms  p95={b['p95_ms']}ms  p99={b['p99_ms']}ms")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Latency] Results saved → {args.out}")


if __name__ == "__main__":
    main()
