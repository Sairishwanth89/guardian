"""
Multi-Session Tracker
=====================
Persists reward curves across separate training sessions (Kaggle notebook runs).
Each session appends a summary entry to outputs/session_history.jsonl so that
cross-session improvement is visible over time.

Usage:
    tracker = MultiSessionTracker()
    tracker.start_session(model_checkpoint="outputs/checkpoints/episode_40")
    # ... during training ...
    tracker.log_episode(episode_id, reward, attack_type, detected)
    tracker.end_session()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


SESSION_HISTORY_PATH = "outputs/session_history.jsonl"


@dataclass
class SessionSummary:
    session_id: str
    start_time: float
    end_time: float = 0.0
    model_checkpoint: str = ""
    total_episodes: int = 0
    mean_reward: float = 0.0
    peak_reward: float = 0.0
    detection_rate: float = 0.0        # fraction of attacks correctly detected
    false_alarm_rate: float = 0.0      # fraction of clean episodes incorrectly blocked
    attack_breakdown: Dict[str, float] = field(default_factory=dict)
    reward_curve: List[float] = field(default_factory=list)


class MultiSessionTracker:
    """
    Tracks learning progress across multiple training sessions.

    Persists to disk so Kaggle re-runs can see cumulative improvement.
    """

    def __init__(self, history_path: str = SESSION_HISTORY_PATH):
        self.history_path = history_path
        self._session: Optional[SessionSummary] = None
        self._episode_rewards: List[float] = []
        self._attack_results: Dict[str, List[bool]] = {}   # attack_type → [detected, ...]
        self._clean_results: List[bool] = []               # clean ep → [blocked, ...]

    def start_session(self, model_checkpoint: str = "") -> str:
        """Start a new training session. Returns session_id."""
        import uuid
        sid = f"session_{str(uuid.uuid4())[:8]}"
        self._session = SessionSummary(
            session_id=sid,
            start_time=time.time(),
            model_checkpoint=model_checkpoint,
        )
        self._episode_rewards = []
        self._attack_results = {}
        self._clean_results = []
        print(f"[MultiSession] Started session {sid}")
        return sid

    def log_episode(
        self,
        episode_id: str,
        reward: float,
        attack_type: Optional[str],
        guardian_detected: bool,
        production_intact: bool,
    ) -> None:
        """Log one episode result to the current session."""
        if self._session is None:
            return
        self._episode_rewards.append(reward)
        self._session.total_episodes += 1
        self._session.reward_curve.append(round(reward, 4))

        if attack_type is not None:
            if attack_type not in self._attack_results:
                self._attack_results[attack_type] = []
            self._attack_results[attack_type].append(guardian_detected and production_intact)
        else:
            # Clean episode: "detected" means incorrectly blocked (false alarm)
            self._clean_results.append(not production_intact)

    def end_session(self, save: bool = True) -> SessionSummary:
        """Finalize the session and append to history."""
        if self._session is None:
            raise RuntimeError("No active session — call start_session() first")

        s = self._session
        s.end_time = time.time()

        if self._episode_rewards:
            s.mean_reward = round(sum(self._episode_rewards) / len(self._episode_rewards), 4)
            s.peak_reward = round(max(self._episode_rewards), 4)

        # Per-attack detection rates
        for attack_type, results in self._attack_results.items():
            if results:
                s.attack_breakdown[attack_type] = round(sum(results) / len(results), 3)

        # Overall detection + false alarm
        all_attack_results = [r for results in self._attack_results.values() for r in results]
        if all_attack_results:
            s.detection_rate = round(sum(all_attack_results) / len(all_attack_results), 3)
        if self._clean_results:
            s.false_alarm_rate = round(sum(self._clean_results) / len(self._clean_results), 3)

        if save:
            self._save(s)

        duration_min = (s.end_time - s.start_time) / 60
        print(
            f"[MultiSession] Session {s.session_id} complete — "
            f"{s.total_episodes} episodes, mean_reward={s.mean_reward:.4f}, "
            f"detection_rate={s.detection_rate:.1%}, duration={duration_min:.1f}min"
        )
        self._session = None
        return s

    def _save(self, summary: SessionSummary) -> None:
        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        with open(self.history_path, "a") as f:
            f.write(json.dumps({
                "session_id": summary.session_id,
                "start_time": summary.start_time,
                "end_time": summary.end_time,
                "duration_min": round((summary.end_time - summary.start_time) / 60, 2),
                "model_checkpoint": summary.model_checkpoint,
                "total_episodes": summary.total_episodes,
                "mean_reward": summary.mean_reward,
                "peak_reward": summary.peak_reward,
                "detection_rate": summary.detection_rate,
                "false_alarm_rate": summary.false_alarm_rate,
                "attack_breakdown": summary.attack_breakdown,
            }) + "\n")

    def load_history(self) -> List[Dict]:
        """Load all past session summaries."""
        sessions = []
        if not os.path.exists(self.history_path):
            return sessions
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sessions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return sessions

    def print_cross_session_summary(self) -> None:
        sessions = self.load_history()
        if not sessions:
            print("[MultiSession] No prior sessions found.")
            return
        print(f"\n=== Cross-Session Learning Summary ({len(sessions)} sessions) ===")
        print(f"{'Session':>15} {'Episodes':>10} {'MeanReward':>12} {'DetectRate':>12} {'FARate':>8}")
        for s in sessions:
            print(
                f"{s['session_id']:>15} {s['total_episodes']:>10} "
                f"{s['mean_reward']:>12.4f} {s['detection_rate']:>12.1%} "
                f"{s['false_alarm_rate']:>8.1%}"
            )
        first = sessions[0]["mean_reward"]
        last = sessions[-1]["mean_reward"]
        delta = last - first
        print(f"\nReward improvement across sessions: {first:.4f} → {last:.4f} ({delta:+.4f})")
