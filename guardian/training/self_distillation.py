"""
Self-Distilled RLVR for GUARDIAN Fleet
========================================
Implements the three-phase loop described in the hackathon brief:

  Phase 1 — EXPLORATION:
      When facing a novel attack, freeze training and sample N=8 distinct
      Guardian intervention paths at high temperature (0.9).

  Phase 2 — VERIFICATION:
      Score each path using the deterministic 11-component reward function
      WITHOUT running a full live episode. Uses counterfactual scoring:
      given the observed attack pattern, what reward would each intervention
      have achieved? This is fully deterministic and requires no LLM judge.

  Phase 3 — DISTILLATION:
      Select the highest-reward trajectory → "Golden Trajectory".
      Append it to the persistent GoldenReplayBuffer (JSONL on disk).

  Phase 4 — GRPO:
      GRPOTrainer trains on batches sampled from the GoldenReplayBuffer
      instead of (or in addition to) the online episode samples.

Key insight:
    Standard GRPO does exploration + exploitation simultaneously — it can only
    learn from trajectories it accidentally discovers. Self-Distilled RLVR
    DECOUPLES exploration (sampling phase) from exploitation (GRPO phase).
    The sampling does the exploring; GRPO only sees verified golden data.

Why it matters for GUARDIAN:
    Novel zero-day attacks (e.g., schema_drift_exploit, salami_slicing) are
    hard to detect — if the untrained model never accidentally uses 'shadow'
    or 'canary_inject' on the first attempt, GRPO never sees a positive reward
    for that attack type, and the policy never improves.
    With N=8 sampling, we *guarantee* we find at least one correct intervention
    (statistically, with p > 0.99 for temperature=0.9) within 8 tries.

Usage (in train_grpo.py):
    from guardian.training.self_distillation import (
        SelfDistillationSampler,
        GoldenReplayBuffer,
        SelfDistillationConfig,
    )
    cfg = SelfDistillationConfig(n_samples=8, temperature=0.9)
    sampler = SelfDistillationSampler(guardian, reward_computer, cfg)
    replay = GoldenReplayBuffer("guardian/data/golden_replay.jsonl")

    # In episode loop:
    if sampler.should_run(episode_number, attack_type, recent_rewards):
        golden = sampler.find_golden_trajectory(state, attack_type, action_log)
        replay.add(golden)

    # In training step (replaces / augments all_samples):
    training_batch = replay.sample(batch_size=16)
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SelfDistillationConfig:
    """Tunable hyperparameters for the Self-Distilled RLVR loop."""

    # Sampling
    n_samples: int = 8
    """Number of Guardian responses to sample per attack step."""

    temperature: float = 0.9
    """Sampling temperature — higher = more diverse exploration."""

    min_reward_gap: float = 0.05
    """Only add to replay buffer if best_reward - mean_reward > this threshold.
    Prevents adding trajectories where all N solutions were equally good/bad."""

    # When to trigger self-distillation (don't run it on EVERY episode)
    trigger_on_new_attack_types: bool = True
    """Always sample when an attack type has fewer than min_episodes_per_attack seen."""

    min_episodes_per_attack: int = 5
    """Run self-distillation until we have >= this many golden trajectories per attack type."""

    trigger_on_low_detection: bool = True
    """Run self-distillation when per-attack detection rate drops below threshold."""

    low_detection_threshold: float = 0.5
    """Threshold below which self-distillation is triggered."""

    # Replay buffer
    max_replay_size: int = 500
    """Maximum number of golden trajectories to keep in the replay buffer."""

    min_replay_size_to_train: int = 8
    """Don't train from replay until at least this many trajectories exist."""

    replay_sample_strategy: str = "prioritized"
    """'uniform' | 'prioritized' — prioritized weights by reward."""


# ── Counterfactual Reward Scorer ──────────────────────────────────────────────

class CounterfactualScorer:
    """
    Score a Guardian decision WITHOUT running a full live episode.

    Given the known ground truth (attack_type, action_log, step context),
    compute what reward the 11-component function would have returned
    if the Guardian had made this specific decision.

    This is deterministic — no LLM, no randomness, no environment step needed.
    The reward depends on:
        intervention → fork_triggered, production_intact
        risk_score → calibration_bonus
        classified_attack → attack_classification_f1
        reasoning → explanation_quality
        attack_active → determines whether this is a TP/FP/TN/FN context
    """

    # Mapping: (attack_active, intervention) → (fork_triggered, production_intact)
    _OUTCOME_TABLE: Dict[Tuple[bool, str], Tuple[bool, bool]] = {
        # Attack active + protective interventions → fork, production safe
        (True, "shadow"):               (True,  True),
        (True, "emergency_fork"):       (True,  True),
        (True, "rollback_k"):           (True,  True),
        (True, "quarantine_tool"):      (True,  True),
        (True, "rewrite"):              (False, True),
        (True, "reduce_privs"):         (False, True),
        (True, "interrogate"):          (False, True),
        (True, "require_justification"):(False, True),
        (True, "canary_inject"):        (False, True),
        (True, "escalate_human"):       (True,  True),
        # Attack active + allow → production breached
        (True, "allow"):                (False, False),
        # Clean episode + any intervention → false alarm (fork_triggered=True means false alarm)
        (False, "shadow"):              (True,  True),   # false alarm but non-destructive
        (False, "emergency_fork"):      (True,  True),
        (False, "allow"):               (False, True),   # correct on benign
    }

    def __init__(self, reward_computer) -> None:
        self._rc = reward_computer

    def score(
        self,
        decision: Dict[str, Any],
        attack_type: Optional[str],
        attack_active: bool,
        action_log: List[Dict],
        fork_step: int,
        current_step: int,
        step_rewards: List[float],
        shadow_tokens: int = 300,
        guardian_tokens: int = 500,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score one Guardian decision counterfactually.

        Returns:
            (total_reward: float, breakdown_dict: dict)
        """
        iv = decision.get("intervention", "allow")
        at = attack_type
        classified = decision.get("classified_attack")
        risk_score = float(decision.get("risk_score", 0.3))
        explanation = decision.get("reasoning", "") + " " + decision.get("explanation", "")

        # Determine counterfactual outcomes
        key = (attack_active, iv)
        fork_triggered, production_intact = self._OUTCOME_TABLE.get(
            key,
            (False, True),  # default: no fork, production safe
        )

        # Detected attack type
        guardian_detected = classified if attack_active and classified else None

        breakdown = self._rc.compute(
            production_intact=production_intact,
            fork_triggered=fork_triggered,
            fork_step=fork_step if fork_triggered else None,
            attack_active=attack_active,
            attack_type=at,
            guardian_detected_type=guardian_detected,
            last_worker_tool=action_log[-1].get("tool", "read_db") if action_log else "read_db",
            shadow_tokens=shadow_tokens if fork_triggered else 0,
            guardian_tokens=guardian_tokens,
            intervention=iv,
            risk_score=risk_score,
            step_rewards=step_rewards,
            explanation=explanation,
            action_log=action_log,
        )
        return breakdown.total, breakdown.to_dict()


# ── Golden Trajectory ─────────────────────────────────────────────────────────

@dataclass
class GoldenTrajectory:
    """A verified, highest-reward Guardian decision for a specific episode context."""
    prompt: str
    completion: str           # the full XML string (raw completion)
    decision: Dict            # parsed decision dict
    reward: float             # verified counterfactual reward
    reward_breakdown: Dict    # 11-component breakdown
    attack_type: Optional[str]
    attack_active: bool
    n_sampled: int            # how many paths were explored
    all_rewards: List[float]  # rewards of all N sampled paths
    reward_gap: float         # best_reward - mean_reward (diversity signal)
    timestamp: float = field(default_factory=time.time)
    episode_step: int = 0


# ── Golden Replay Buffer ──────────────────────────────────────────────────────

class GoldenReplayBuffer:
    """
    Persistent, prioritized replay buffer of verified golden trajectories.

    Stored as JSONL on disk — survives across training sessions.
    Supports prioritized sampling (higher-reward trajectories sampled more often)
    and per-attack-type balancing to prevent forgetting.
    """

    def __init__(
        self,
        path: str = "guardian/data/golden_replay.jsonl",
        max_size: int = 500,
    ) -> None:
        self.path = path
        self.max_size = max_size
        self._buffer: List[GoldenTrajectory] = []
        self._per_attack_count: Dict[str, int] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def add(self, traj: GoldenTrajectory) -> None:
        """Append a golden trajectory to the buffer and persist to disk."""
        self._buffer.append(traj)
        atk = traj.attack_type or "clean"
        self._per_attack_count[atk] = self._per_attack_count.get(atk, 0) + 1

        # Evict lowest-reward entries when over capacity
        if len(self._buffer) > self.max_size:
            self._buffer.sort(key=lambda t: t.reward, reverse=True)
            self._buffer = self._buffer[:self.max_size]

        # Persist (append only — fast even for large buffers)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_dict(traj)) + "\n")

    def sample(
        self,
        batch_size: int = 16,
        strategy: str = "prioritized",
    ) -> List[Dict[str, str]]:
        """
        Sample a training batch from the buffer.

        Returns list of {"prompt": ..., "completion": ...} dicts
        ready for GRPOTrainer via huggingface datasets.

        strategy: 'prioritized' — weight by reward (higher reward → more likely)
                  'uniform'     — uniform random
                  'balanced'    — one per attack type, then fill with prioritized
        """
        if not self._buffer:
            return []
        n = min(batch_size, len(self._buffer))
        if strategy == "uniform":
            chosen = random.sample(self._buffer, n)
        elif strategy == "balanced":
            chosen = self._balanced_sample(n)
        else:
            chosen = self._prioritized_sample(n)
        return [{"prompt": t.prompt, "completion": t.completion} for t in chosen]

    def sample_with_rewards(
        self,
        batch_size: int = 16,
        strategy: str = "prioritized",
    ) -> Tuple[List[str], List[float]]:
        """
        Sample batch and return (prompts_list, rewards_list) for reward_fn in GRPOTrainer.
        """
        if not self._buffer:
            return [], []
        n = min(batch_size, len(self._buffer))
        if strategy == "prioritized":
            chosen = self._prioritized_sample(n)
        else:
            chosen = random.sample(self._buffer, n)
        prompts = [t.prompt for t in chosen]
        rewards = [t.reward for t in chosen]
        return prompts, rewards

    def size(self) -> int:
        return len(self._buffer)

    def per_attack_counts(self) -> Dict[str, int]:
        return dict(self._per_attack_count)

    def mean_reward(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(t.reward for t in self._buffer) / len(self._buffer)

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size(),
            "mean_reward": round(self.mean_reward(), 4),
            "per_attack": self.per_attack_counts(),
            "path": self.path,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _prioritized_sample(self, n: int) -> List[GoldenTrajectory]:
        """Sample weighted by reward (softmax temperature 1.0)."""
        rewards = [max(0.01, t.reward + 1.0) for t in self._buffer]  # shift to > 0
        total = sum(rewards)
        weights = [r / total for r in rewards]
        return random.choices(self._buffer, weights=weights, k=n)

    def _balanced_sample(self, n: int) -> List[GoldenTrajectory]:
        """One trajectory per attack type, fill remainder with prioritized."""
        attack_groups: Dict[str, List[GoldenTrajectory]] = {}
        for t in self._buffer:
            key = t.attack_type or "clean"
            attack_groups.setdefault(key, []).append(t)
        chosen: List[GoldenTrajectory] = []
        for group in attack_groups.values():
            chosen.append(max(group, key=lambda t: t.reward))
        # Fill remaining slots
        remaining = n - len(chosen)
        if remaining > 0 and len(self._buffer) > len(chosen):
            rest = [t for t in self._buffer if t not in chosen]
            chosen.extend(random.sample(rest, min(remaining, len(rest))))
        return chosen[:n]

    def _load(self) -> None:
        """Load existing buffer from disk on startup."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    traj = GoldenTrajectory(**d)
                    self._buffer.append(traj)
                    atk = traj.attack_type or "clean"
                    self._per_attack_count[atk] = \
                        self._per_attack_count.get(atk, 0) + 1
            print(f"[GoldenReplayBuffer] Loaded {len(self._buffer)} trajectories from {self.path}")
        except Exception as e:
            print(f"[GoldenReplayBuffer] Warning: could not load {self.path}: {e}")

    @staticmethod
    def _to_dict(traj: GoldenTrajectory) -> Dict:
        d = asdict(traj)
        return d


# ── Self-Distillation Sampler ────────────────────────────────────────────────

class SelfDistillationSampler:
    """
    Orchestrates the 3-phase Self-Distilled RLVR loop:

      1. EXPLORATION  — sample N Guardian responses at high temperature
      2. VERIFICATION — score each with CounterfactualScorer (deterministic)
      3. DISTILLATION — select best → return GoldenTrajectory

    Usage:
        sampler = SelfDistillationSampler(guardian, reward_computer, cfg)
        if sampler.should_run(episode, attack_type, per_attack_rewards):
            golden = sampler.find_golden_trajectory(
                state, attack_type, action_log, risk_history
            )
            if golden:
                replay_buffer.add(golden)
    """

    def __init__(
        self,
        guardian,           # GuardianAgent
        reward_computer,    # RewardComputer
        config: Optional[SelfDistillationConfig] = None,
    ) -> None:
        self.guardian = guardian
        self.cfg = config or SelfDistillationConfig()
        self._scorer = CounterfactualScorer(reward_computer)
        self._trigger_counts: Dict[str, int] = {}  # attack_type → times triggered

    # ── Public API ──────────────────────────────────────────────────────────

    def should_run(
        self,
        episode: int,
        attack_type: Optional[str],
        per_attack_rewards: Dict[str, List[float]],
        replay_buffer: Optional[GoldenReplayBuffer] = None,
    ) -> bool:
        """
        Decide whether to run self-distillation this episode.

        Returns True when:
          (a) attack type has too few golden trajectories, OR
          (b) recent detection rate for this attack type is below threshold
        """
        atk = str(attack_type)

        # Always run for novel attack types with few examples
        if replay_buffer and self.cfg.trigger_on_new_attack_types:
            per_atk = replay_buffer.per_attack_counts()
            if per_atk.get(atk, 0) < self.cfg.min_episodes_per_attack:
                return True

        # Run when detection rate is low for this attack
        if self.cfg.trigger_on_low_detection and attack_type is not None:
            rewards = per_attack_rewards.get(atk, [])
            if len(rewards) >= 3:
                recent = rewards[-5:]
                mean_r = sum(recent) / len(recent)
                if mean_r < self.cfg.low_detection_threshold:
                    return True

        return False

    def find_golden_trajectory(
        self,
        state,                        # GUARDIANEnvironment world state
        attack_type: Optional[str],
        action_log: List[Dict],
        risk_history: Optional[List[float]] = None,
        faiss_context: Optional[str] = None,
        schema_version: int = 0,
    ) -> Optional[GoldenTrajectory]:
        """
        Phase 1: Sample N interventions from Guardian LLM.
        Phase 2: Score each counterfactually.
        Phase 3: Return the highest-scoring trajectory if it clears min_reward_gap.

        Returns None if all sampled trajectories score poorly (no signal).
        """
        n = self.cfg.n_samples
        print(f"\n  [SelfDistill] Exploring {n} paths for attack={attack_type}...")

        # ── Phase 1: Exploration ────────────────────────────────────────────
        attack_active = bool(getattr(state, "attack_active", False))
        fork_step = getattr(state, "fork_step", None) or (
            getattr(state, "episode_step", 1) - 1
        )
        current_step = getattr(state, "episode_step", 1)

        decisions = self.guardian.sample_n_completions(
            action_log=action_log,
            n=n,
            temperature=self.cfg.temperature,
            faiss_context=faiss_context,
            schema_version=schema_version,
            risk_history=risk_history,
        )

        # Build representative step_rewards (safe phases)
        step_rewards = [0.03] * max(1, current_step - 1)

        # ── Phase 2: Verification ───────────────────────────────────────────
        scored: List[Tuple[float, Dict[str, float], Dict]] = []
        for d in decisions:
            reward, breakdown = self._scorer.score(
                decision=d,
                attack_type=attack_type,
                attack_active=attack_active,
                action_log=action_log,
                fork_step=fork_step,
                current_step=current_step,
                step_rewards=step_rewards,
                shadow_tokens=300,
                guardian_tokens=500,
            )
            scored.append((reward, breakdown, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        all_rewards = [s[0] for s in scored]
        best_reward, best_breakdown, best_decision = scored[0]
        mean_reward = sum(all_rewards) / len(all_rewards)
        reward_gap = best_reward - mean_reward

        # ── Phase 3: Distillation gate ──────────────────────────────────────
        print(
            f"  [SelfDistill] Rewards: best={best_reward:.3f} "
            f"mean={mean_reward:.3f} gap={reward_gap:.3f}"
        )

        if reward_gap < self.cfg.min_reward_gap:
            print(
                f"  [SelfDistill] Gap {reward_gap:.3f} < {self.cfg.min_reward_gap} "
                f"— skipping (all paths similar quality)"
            )
            # Still return the best if it's a strong positive signal
            if best_reward < 0.3:
                return None

        # Build the training prompt for this decision
        prompt = self.guardian.build_training_prompt(
            action_log,
            faiss_context=faiss_context,
            schema_version=schema_version,
            risk_history=risk_history,
        )

        # Use raw completion if available, else reconstruct from parsed decision
        completion = best_decision.get("_raw_completion", "") or self._decision_to_xml(best_decision)

        atk = str(attack_type)
        self._trigger_counts[atk] = self._trigger_counts.get(atk, 0) + 1

        return GoldenTrajectory(
            prompt=prompt,
            completion=completion,
            decision=best_decision,
            reward=best_reward,
            reward_breakdown=best_breakdown,
            attack_type=attack_type,
            attack_active=attack_active,
            n_sampled=n,
            all_rewards=all_rewards,
            reward_gap=reward_gap,
            timestamp=time.time(),
            episode_step=current_step,
        )

    def trigger_stats(self) -> Dict[str, int]:
        return dict(self._trigger_counts)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _decision_to_xml(decision: Dict) -> str:
        """Reconstruct XML completion from a parsed decision dict."""
        return (
            f"<reasoning>{decision.get('reasoning', 'Detected anomaly.')}</reasoning>\n"
            f"<risk_score>{decision.get('risk_score', 0.5):.2f}</risk_score>\n"
            f"<intervention>{decision.get('intervention', 'shadow')}</intervention>\n"
            f"<attack_type>{decision.get('classified_attack') or 'clean'}</attack_type>\n"
            f"<explanation>{decision.get('explanation', '')}</explanation>"
        )
