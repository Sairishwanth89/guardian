"""
Training Curve Plotter
======================
Reads outputs/training_log.jsonl and outputs/reward_breakdown_log.csv
to generate reward improvement graphs for the hackathon judging evidence.

Usage:
    python -m guardian.training.plot_training
    python -m guardian.training.plot_training --log outputs/training_log.jsonl --out outputs/plots/
"""

from __future__ import annotations

import json
import os
import argparse
from typing import List, Dict


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend — no display required
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[Plot] matplotlib not installed — run: pip install matplotlib")
        return None


def load_jsonl(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        print(f"[Plot] File not found: {path}")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_csv(path: str) -> List[Dict]:
    import csv
    records = []
    if not os.path.exists(path):
        return records
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            records.append(parsed)
    return records


def plot_reward_curve(records: List[Dict], out_dir: str, plt) -> None:
    """Total reward over training episodes."""
    rewards = [r.get("reward_total", r.get("total", 0.0)) for r in records]
    if not rewards:
        return

    # Smoothed moving average (window=10)
    window = min(10, len(rewards))
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(sum(rewards[start:i+1]) / (i - start + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, color="#4C72B0", linewidth=0.8, label="Per-episode reward")
    plt.plot(smoothed, color="#4C72B0", linewidth=2.0, label=f"Moving avg (n={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (normalized 0–1)")
    plt.title("GUARDIAN Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reward_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_component_breakdown(records: List[Dict], out_dir: str, plt) -> None:
    """Stacked area chart of all 16 reward components over time."""
    components = [
        "production_safety", "business_continuity", "intervention_timeliness",
        "attack_classification_f1", "explanation_quality", "minimality_bonus",
        "calibration_bonus", "risk_score_component", "reasoning_quality",
        "detection_lag_bonus", "rogue_ai_containment_bonus",
    ]
    data = {c: [r.get(c, 0.0) for r in records] for c in components}
    if not any(data.values()):
        return

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    x = list(range(len(records)))
    bottom = [0.0] * len(records)
    for i, (comp, values) in enumerate(data.items()):
        plt.bar(x, values, bottom=bottom, label=comp,
                color=colors[i % len(colors)], alpha=0.85, width=1.0)
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xlabel("Episode")
    plt.ylabel("Reward Component Value")
    plt.title("GUARDIAN — 16-Component Reward Breakdown")
    plt.legend(fontsize=7, loc="upper left", ncol=2)
    plt.grid(True, alpha=0.2, axis="y")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reward_components.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_detection_rate(records: List[Dict], out_dir: str, plt) -> None:
    """Attack detection rate over training (rolling window)."""
    detections = []
    for r in records:
        at = r.get("attack_type", "clean")
        detected = r.get("guardian_detected_type") or r.get("detected")
        if at != "clean" and at is not None:
            detections.append(1.0 if detected else 0.0)

    if len(detections) < 5:
        return

    window = min(20, len(detections))
    smoothed = []
    for i in range(len(detections)):
        start = max(0, i - window + 1)
        smoothed.append(sum(detections[start:i+1]) / (i - start + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(detections, alpha=0.2, color="#DD8452", linewidth=0.8)
    plt.plot(smoothed, color="#DD8452", linewidth=2.0, label=f"Detection rate (n={window})")
    plt.axhline(y=0.8, color="green", linestyle="--", alpha=0.7, label="Target 80%")
    plt.xlabel("Attack Episode Index")
    plt.ylabel("Detection Rate")
    plt.title("GUARDIAN Attack Detection Rate Over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "detection_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_attack_type_rewards(records: List[Dict], out_dir: str, plt) -> None:
    """Per-attack-type mean reward bar chart."""
    from collections import defaultdict
    by_attack: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        at = r.get("attack_type", "clean")
        reward = r.get("reward_total", r.get("total", 0.0))
        by_attack[str(at)].append(reward)

    if not by_attack:
        return

    labels = sorted(by_attack.keys())
    means = [sum(by_attack[l]) / len(by_attack[l]) for l in labels]
    colors = ["#4C72B0" if l != "clean" else "#55A868" for l in labels]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, means, color=colors, alpha=0.85)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean Reward")
    plt.title("GUARDIAN — Mean Reward by Attack Type")
    plt.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.01,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "attack_type_rewards.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GUARDIAN training curves")
    parser.add_argument("--log", default="outputs/training_log.jsonl")
    parser.add_argument("--csv", default="outputs/reward_breakdown_log.csv")
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()

    plt = _try_import_matplotlib()
    if plt is None:
        return

    records = load_jsonl(args.log)
    if not records:
        csv_records = load_csv(args.csv)
        records = csv_records

    if not records:
        print("[Plot] No training data found. Run some episodes first.")
        return

    print(f"[Plot] Loaded {len(records)} episode records from {args.log or args.csv}")
    plot_reward_curve(records, args.out, plt)
    plot_component_breakdown(records, args.out, plt)
    plot_detection_rate(records, args.out, plt)
    plot_attack_type_rewards(records, args.out, plt)
    print(f"\n[Plot] All plots saved to {args.out}/")


if __name__ == "__main__":
    main()
