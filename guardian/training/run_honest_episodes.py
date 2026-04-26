"""
Honest Baseline Runner
=======================
Runs GUARDIAN in honest/zero-shot mode (no GRPO training applied) to
establish a baseline reward score. This is the "before training" measurement
that proves GRPO actually improved the model.

Comparison:
  baseline (honest) → trained model → delta proves improvement

Saves results to outputs/honest_baseline.jsonl

Usage:
    python -m guardian.training.run_honest_episodes --n 50
    python -m guardian.training.run_honest_episodes --n 20 --no-model
"""

from __future__ import annotations

import json
import os
import time
import random
import argparse
from typing import Optional, List, Dict


ALL_ATTACK_TYPES = [
    None,  # clean
    "authority_spoofing",
    "prompt_injection",
    "approval_bypass",
    "data_exfiltration",
    "confused_deputy",
    "approval_laundering",
    "salami_slicing",
    "schema_drift_exploit",
    "rogue_internal_ai",
    "delayed_exfiltration",
    "social_engineering",
]


def run_honest_baseline(n_episodes: int = 50, use_model: bool = False) -> Dict:
    from guardian.environment.guardian_env import GUARDIANEnvironment
    from guardian.environment.reward_computer import RewardComputer
    from guardian.agents.guardian_agent import GuardianAgent
    from guardian.agents.worker_agent import FinanceWorker
    from guardian.training.episode_runner import EpisodeRunner

    print(f"\n[Honest] Running {n_episodes} honest baseline episodes...")

    env = GUARDIANEnvironment()
    worker = FinanceWorker()
    guardian = GuardianAgent()
    rc = RewardComputer()

    if use_model:
        print("[Honest] Loading model for honest baseline...")
        try:
            guardian.load_model()
        except Exception as e:
            print(f"[Honest] Model load failed ({e}) — using heuristic mode")

    runner = EpisodeRunner(env=env, worker=worker, guardian=guardian, reward_computer=rc)

    all_rewards: List[float] = []
    all_records: List[Dict] = []
    detection_results: List[bool] = []
    false_alarms: List[bool] = []

    os.makedirs("outputs", exist_ok=True)

    for i in range(n_episodes):
        attack_type = random.choice(ALL_ATTACK_TYPES)
        result = runner.run_episode(attack_type=attack_type)

        all_rewards.append(result.reward)
        detected = bool(result.guardian_detected_type)

        if attack_type is not None:
            detection_results.append(detected and result.production_intact)
        else:
            false_alarms.append(not result.production_intact)

        record = {
            "episode": i + 1,
            "episode_id": result.episode_id,
            "attack_type": attack_type,
            "reward": result.reward,
            "production_intact": result.production_intact,
            "detected": detected,
            "mode": "honest_baseline",
        }
        all_records.append(record)

        with open("outputs/honest_baseline.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        if (i + 1) % 10 == 0:
            mean_so_far = sum(all_rewards) / len(all_rewards)
            print(f"  Episode {i+1:3d}/{n_episodes}  mean_reward={mean_so_far:.4f}")

    # Final summary
    mean_reward = sum(all_rewards) / len(all_rewards)
    detection_rate = sum(detection_results) / len(detection_results) if detection_results else 0.0
    false_alarm_rate = sum(false_alarms) / len(false_alarms) if false_alarms else 0.0
    peak_reward = max(all_rewards)
    min_reward = min(all_rewards)

    summary = {
        "mode": "honest_baseline",
        "n_episodes": n_episodes,
        "mean_reward": round(mean_reward, 4),
        "peak_reward": round(peak_reward, 4),
        "min_reward": round(min_reward, 4),
        "detection_rate": round(detection_rate, 3),
        "false_alarm_rate": round(false_alarm_rate, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open("outputs/honest_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Honest] Baseline complete:")
    print(f"  Mean reward:      {mean_reward:.4f}")
    print(f"  Detection rate:   {detection_rate:.1%}")
    print(f"  False alarm rate: {false_alarm_rate:.1%}")
    print(f"  Peak reward:      {peak_reward:.4f}")
    print(f"\n  Save this summary. After GRPO training, run again to measure delta.")
    print(f"  Summary: outputs/honest_baseline_summary.json")
    print(f"  Log:     outputs/honest_baseline.jsonl")

    return summary


def compare_with_trained(baseline_path: str = "outputs/honest_baseline_summary.json",
                          trained_path: str = "outputs/trained_summary.json") -> None:
    """Print delta between baseline and trained model."""
    if not os.path.exists(baseline_path) or not os.path.exists(trained_path):
        print("[Compare] One or both summary files not found.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(trained_path) as f:
        trained = json.load(f)

    print("\n=== Baseline vs Trained Model Comparison ===")
    print(f"{'Metric':<25} {'Baseline':>12} {'Trained':>12} {'Delta':>10}")
    print("─" * 62)

    for key in ("mean_reward", "detection_rate", "false_alarm_rate", "peak_reward"):
        b = baseline.get(key, 0.0)
        t = trained.get(key, 0.0)
        delta = t - b
        sign = "+" if delta >= 0 else ""
        direction = "↑" if (key != "false_alarm_rate" and delta > 0) or (key == "false_alarm_rate" and delta < 0) else "↓"
        print(f"  {key:<23} {b:>12.4f} {t:>12.4f} {sign}{delta:>9.4f} {direction}")

    reward_improvement = ((trained.get("mean_reward", 0) - baseline.get("mean_reward", 0))
                          / max(baseline.get("mean_reward", 0.001), 0.001) * 100)
    print(f"\n  Reward improvement: {reward_improvement:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run honest baseline episodes")
    parser.add_argument("--n", type=int, default=50, help="Number of episodes")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs trained")
    args = parser.parse_args()

    if args.compare:
        compare_with_trained()
    else:
        run_honest_baseline(n_episodes=args.n, use_model=not args.no_model)


if __name__ == "__main__":
    main()
