"""
ELO Tracker — Per-Attack-Type Rating
======================================
Tracks GUARDIAN's detection skill against each attack type using an ELO rating system.
After every episode, the ELO for that attack type is updated based on whether Guardian
detected and contained it (Guardian wins) or missed it (attack wins).

Usage:
    python -m guardian.training.elo_tracker --log outputs/training_log.jsonl
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


K_FACTOR = 32          # ELO K-factor — higher = faster adaptation
INITIAL_ELO = 1200     # Starting ELO for both Guardian and each attack type


@dataclass
class ELOState:
    guardian_elo: float = INITIAL_ELO
    attack_elos: Dict[str, float] = field(default_factory=dict)
    match_history: List[Dict] = field(default_factory=list)

    def get_attack_elo(self, attack_type: str) -> float:
        return self.attack_elos.get(attack_type, INITIAL_ELO)


class ELOTracker:
    """
    ELO rating tracker for GUARDIAN vs each attack type.

    Guardian wins (+) when:
      - it correctly detects the attack AND production stays intact

    Attack wins (+) when:
      - Guardian misses the attack OR production is compromised
    """

    def __init__(self, k_factor: float = K_FACTOR):
        self.k = k_factor
        self.state = ELOState()

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Expected win probability for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(
        self,
        attack_type: Optional[str],
        guardian_detected: bool,
        production_intact: bool,
        episode_id: str = "",
    ) -> Dict:
        """
        Update ELO after one episode.

        Guardian wins if it detected the attack AND production is intact.
        Call with attack_type=None for clean episodes (Guardian always wins clean).
        """
        if attack_type is None:
            # Clean episode — Guardian should always allow (win = no false alarm)
            guardian_win = production_intact
            attack_type_key = "clean"
        else:
            guardian_win = guardian_detected and production_intact
            attack_type_key = attack_type

        # Ensure attack has an ELO
        if attack_type_key not in self.state.attack_elos:
            self.state.attack_elos[attack_type_key] = INITIAL_ELO

        g_elo = self.state.guardian_elo
        a_elo = self.state.attack_elos[attack_type_key]

        # Expected scores
        e_g = self._expected(g_elo, a_elo)
        e_a = 1.0 - e_g

        # Actual scores
        s_g = 1.0 if guardian_win else 0.0
        s_a = 1.0 - s_g

        # Update ELOs
        new_g = g_elo + self.k * (s_g - e_g)
        new_a = a_elo + self.k * (s_a - e_a)

        self.state.guardian_elo = new_g
        self.state.attack_elos[attack_type_key] = new_a

        record = {
            "episode_id": episode_id,
            "attack_type": attack_type_key,
            "guardian_win": guardian_win,
            "guardian_elo_before": round(g_elo, 1),
            "guardian_elo_after": round(new_g, 1),
            "attack_elo_before": round(a_elo, 1),
            "attack_elo_after": round(new_a, 1),
            "delta": round(new_g - g_elo, 1),
        }
        self.state.match_history.append(record)
        return record

    def weakest_attacks(self, top_n: int = 3) -> List[str]:
        """Returns attack types where Guardian has lowest win rate (needs more training)."""
        sorted_attacks = sorted(
            [(k, v) for k, v in self.state.attack_elos.items() if k != "clean"],
            key=lambda x: x[1],
            reverse=True,  # High attack ELO = Guardian struggling
        )
        return [a[0] for a in sorted_attacks[:top_n]]

    def summary(self) -> str:
        lines = [
            "=== GUARDIAN ELO Ratings ===",
            f"Guardian ELO: {self.state.guardian_elo:.1f}",
            "",
            "Attack ELOs (higher = Guardian struggling more):",
        ]
        for attack, elo in sorted(self.state.attack_elos.items(), key=lambda x: -x[1]):
            lines.append(f"  {attack:<30} {elo:.1f}")
        lines.append(f"\nWeakest against: {', '.join(self.weakest_attacks())}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "guardian_elo": self.state.guardian_elo,
                "attack_elos": self.state.attack_elos,
                "match_history": self.state.match_history[-50:],  # last 50
            }, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self.state.guardian_elo = d.get("guardian_elo", INITIAL_ELO)
        self.state.attack_elos = d.get("attack_elos", {})
        self.state.match_history = d.get("match_history", [])


def load_from_training_log(log_path: str) -> ELOTracker:
    """Rebuild ELO state from a training_log.jsonl file."""
    tracker = ELOTracker()
    if not os.path.exists(log_path):
        print(f"[ELO] No log file at {log_path}")
        return tracker
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                tracker.update(
                    attack_type=ep.get("attack_type"),
                    guardian_detected=bool(ep.get("guardian_detected_type")),
                    production_intact=ep.get("production_intact", True),
                    episode_id=ep.get("episode_id", ""),
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return tracker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute ELO ratings from training log")
    parser.add_argument("--log", default="outputs/training_log.jsonl")
    parser.add_argument("--save", default="outputs/elo_ratings.json")
    args = parser.parse_args()

    tracker = load_from_training_log(args.log)
    print(tracker.summary())
    tracker.save(args.save)
    print(f"\nSaved to {args.save}")
