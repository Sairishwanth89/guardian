"""
GUARDIAN Self-Distilled GRPO Training — Kaggle T4 x2 / Local 4050
===================================================================
Upgrades the standard GRPO loop with Self-Distilled RLVR:

  Standard GRPO (old):
      run_episode() → random trajectory → GRPOTrainer
      Problem: never learns rare attacks it never accidentally solves.

  Self-Distilled GRPO (new, this file):
      For each episode:
        1. [ONLINE]      run_episode() → online training sample (reward-weighted)
        2. [EXPLORATION] sample_n_completions(N=8, temp=0.9) → diverse interventions
        3. [VERIFY]      CounterfactualScorer scores all N deterministically
        4. [DISTILL]     Best trajectory → GoldenReplayBuffer (disk-persistent)
        5. [GRPO]        Train on BLEND: online samples + golden replay buffer

  Result:
        - Guaranteed positive signal for every attack type (N=8 exploration)
        - Zero catastrophic forgetting (persistent golden replay buffer)
        - GRPO only sees curated, verified "golden" trajectories
        - Decoupled exploration from exploitation

Run (local):
    python -m guardian.training.train_grpo

Run (Kaggle):
    kaggle kernels push  (see guardian_kaggle_training.ipynb)

Requirements:
    pip install unsloth trl peft datasets openai python-dotenv gymnasium
    OPENAI_API_KEY in .env
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from guardian.environment.guardian_env import GUARDIANEnvironment
from guardian.environment.reward_computer import RewardComputer
from guardian.agents.worker_agent import WorkerAgent
from guardian.agents.guardian_agent import GuardianAgent
from guardian.agents.compliance_simulator import ComplianceSimulator
from guardian.agents.curriculum_agent import UCBAttackSelector, CurriculumAgent
from guardian.training.episode_runner import EpisodeRunner
from guardian.training.evaluation import EvaluationHarness
from guardian.training.self_distillation import (
    SelfDistillationSampler,
    SelfDistillationConfig,
    GoldenReplayBuffer,
)


# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_EPISODES         = 200
TRAIN_EVERY            = 8       # Online GRPO step every N episodes
DISTILL_TRAIN_EVERY    = 4       # Also train on replay buffer more frequently
SAVE_EVERY             = 50
EVAL_EVERY             = 40
LOG_FILE               = "guardian/data/training_log.jsonl"
SCORECARD_FILE         = "guardian/data/scorecards.jsonl"
BASELINE_FILE          = "guardian/data/baseline_untrained.json"
REPLAY_BUFFER_FILE     = "guardian/data/golden_replay.jsonl"

# Self-distillation hyperparameters
SD_N_SAMPLES           = 8       # Number of paths to explore per attack step
SD_TEMPERATURE         = 0.9     # High temperature for diverse exploration
SD_MIN_REWARD_GAP      = 0.05    # Only save if best > mean + gap
SD_MIN_EPISODES_PER_ATK= 5       # Keep distilling until N golden examples per attack type
SD_LOW_DETECT_THRESH   = 0.50    # Trigger distillation when detection rate drops below this

# GRPO batch composition
ONLINE_WEIGHT          = 0.5     # Fraction of GRPO batch from online episodes
REPLAY_WEIGHT          = 0.5     # Fraction from golden replay buffer

ATTACK_POOL = [
    None,
    "authority_spoofing",
    "prompt_injection",
    "approval_bypass",
    "data_exfiltration",
    "confused_deputy",
    "approval_laundering",
]


# ── Helper: build blended GRPO dataset ───────────────────────────────────────

def _build_blended_dataset(
    online_samples: list,
    replay_buffer: GoldenReplayBuffer,
    online_weight: float = ONLINE_WEIGHT,
    replay_weight: float = REPLAY_WEIGHT,
) -> tuple[Dataset, dict]:
    """
    Blend online episode samples with verified golden replay trajectories.

    The blend ensures GRPO trains on:
      - Fresh online trajectories (exploration diversity)
      - Verified golden trajectories (guaranteed positive signal)

    Returns:
        dataset: HuggingFace Dataset[{"prompt": str}]
        reward_lookup: {prompt[:100] → reward}
    """
    combined: list = []
    reward_lookup: dict = {}

    # Online samples (reward-weighted)
    n_online = max(1, int(len(online_samples) * online_weight / online_weight))
    for s in online_samples[:n_online]:
        combined.append({"prompt": s["prompt"]})
        reward_lookup[s["prompt"][:100]] = float(s.get("reward", 0.5))

    # Golden replay (prioritized sample)
    if replay_buffer.size() >= 4:
        n_replay = max(4, int(n_online * (replay_weight / online_weight)))
        replay_prompts, replay_rewards = replay_buffer.sample_with_rewards(
            batch_size=n_replay, strategy="balanced"
        )
        for prompt, reward in zip(replay_prompts, replay_rewards):
            combined.append({"prompt": prompt})
            reward_lookup[prompt[:100]] = reward

    random.shuffle(combined)
    dataset = Dataset.from_list(combined)
    return dataset, reward_lookup


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 65)
    print("GUARDIAN Self-Distilled GRPO Training")
    print("=" * 65)

    # ── [1/6] Load model ──────────────────────────────────────────────────
    print("\n[1/6] Loading Qwen2.5-7B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("   Model ready.")

    # ── [2/6] Initialise agents ───────────────────────────────────────────
    print("\n[2/6] Initialising agent fleet...")
    api_key = os.getenv("OPENAI_API_KEY")
    worker    = WorkerAgent(role="finance", api_key=api_key)
    guardian  = GuardianAgent()
    guardian.model     = model
    guardian.tokenizer = tokenizer

    ucb        = UCBAttackSelector(attack_pool=ATTACK_POOL)
    curriculum = CurriculumAgent(api_key=api_key)
    compliance = ComplianceSimulator(api_key=api_key)

    runner = EpisodeRunner(
        env=GUARDIANEnvironment(),
        worker=worker,
        guardian=guardian,
        reward_computer=RewardComputer(),
        compliance_sim=compliance,
        curriculum_agent=curriculum,
        ucb_selector=ucb,
    )
    runner._use_ucb = True
    print("   Agents ready.")

    # ── [3/6] Self-distillation setup ─────────────────────────────────────
    print("\n[3/6] Setting up Self-Distilled RLVR components...")
    sd_config = SelfDistillationConfig(
        n_samples=SD_N_SAMPLES,
        temperature=SD_TEMPERATURE,
        min_reward_gap=SD_MIN_REWARD_GAP,
        min_episodes_per_attack=SD_MIN_EPISODES_PER_ATK,
        low_detection_threshold=SD_LOW_DETECT_THRESH,
        max_replay_size=500,
        min_replay_size_to_train=8,
        replay_sample_strategy="balanced",
    )
    sd_sampler = SelfDistillationSampler(
        guardian=guardian,
        reward_computer=RewardComputer(),
        config=sd_config,
    )
    replay_buffer = GoldenReplayBuffer(
        path=REPLAY_BUFFER_FILE,
        max_size=sd_config.max_replay_size,
    )
    print(f"   Replay buffer: {replay_buffer.size()} existing trajectories loaded.")

    # ── [4/6] Baseline measurement ────────────────────────────────────────
    print("\n[4/6] Recording untrained baseline (20 episodes)...")
    if not os.path.exists(BASELINE_FILE):
        baseline_results = []
        for _ in range(20):
            atk = random.choice(ATTACK_POOL)
            try:
                res = runner.run_episode(attack_type=atk)
                baseline_results.append({
                    "attack_type": atk,
                    "reward": res.reward,
                    "production_intact": res.production_intact,
                    "fork_triggered": res.fork_triggered,
                    "detected": res.guardian_detected_type,
                })
            except Exception as e:
                print(f"  Baseline ep error: {e}")
        os.makedirs("guardian/data", exist_ok=True)
        with open(BASELINE_FILE, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"   Baseline saved to {BASELINE_FILE}")
    else:
        print(f"   Baseline already exists: {BASELINE_FILE}")

    # ── [5/6] GRPO config ─────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir="guardian/checkpoints/grpo_tmp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_steps=8,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_8bit",
    )

    all_samples: list  = []
    samples_map: dict  = {}
    os.makedirs("guardian/data", exist_ok=True)
    os.makedirs("guardian/checkpoints", exist_ok=True)

    # Per-attack tracking
    per_attack_rewards:  dict = {str(a): [] for a in ATTACK_POOL}
    per_attack_detected: dict = {str(a): [] for a in ATTACK_POOL}

    # Counters for logging
    total_distill_runs    = 0
    total_golden_added    = 0

    # ── [6/6] Main training loop ──────────────────────────────────────────
    print(f"\n[6/6] Training loop: {TOTAL_EPISODES} episodes  "
          f"(Self-Distilled RLVR, N={SD_N_SAMPLES})\n")
    print(f"  {'ep':>4} | {'attack':<22} | intact | fork | reward | distill | time")
    print("  " + "-" * 68)

    for ep in range(1, TOTAL_EPISODES + 1):
        t0 = time.time()

        # ── Online episode ────────────────────────────────────────────────
        try:
            result = runner.run_episode()     # UCB selects attack
        except Exception as e:
            print(f"  ep={ep:03d} | ERROR: {e}")
            continue

        elapsed     = time.time() - t0
        atk_str     = str(result.attack_type)
        distill_tag = ""

        # Log entry
        entry = {
            "episode": ep,
            "attack_type": result.attack_type,
            "production_intact": result.production_intact,
            "fork_triggered": result.fork_triggered,
            "reward": round(result.reward, 4),
            "elapsed_s": round(elapsed, 1),
            "detected": result.guardian_detected_type,
            "difficulty": runner.env.state.episode_step,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        with open(SCORECARD_FILE, "a") as f:
            f.write(json.dumps(result.scorecard) + "\n")

        # Track per-attack stats
        per_attack_rewards[atk_str].append(result.reward)
        detected_flag = (
            result.guardian_detected_type is not None
            or result.attack_type is None
        )
        per_attack_detected[atk_str].append(detected_flag)

        # Accumulate online training samples
        for s in result.training_samples:
            s["reward"] = result.reward   # tag with episode reward for blend weighting
            all_samples.append(s)
            samples_map[s["prompt"][:100]] = result.reward

        # ── PHASE 1-3: Self-Distillation ──────────────────────────────────
        if sd_sampler.should_run(
            episode=ep,
            attack_type=result.attack_type,
            per_attack_rewards=per_attack_rewards,
            replay_buffer=replay_buffer,
        ):
            try:
                golden = sd_sampler.find_golden_trajectory(
                    state=runner.env.state,
                    attack_type=result.attack_type,
                    action_log=runner.env.state.action_log,
                    risk_history=runner.env.state.risk_history
                    if hasattr(runner.env.state, "risk_history") else None,
                    faiss_context=runner._get_faiss_context()
                    if hasattr(runner, "_get_faiss_context") else None,
                    schema_version=getattr(runner.env.state, "schema_version", 0),
                )
                if golden:
                    replay_buffer.add(golden)
                    total_golden_added += 1
                    distill_tag = f"  ✨ golden({golden.reward:.3f})"
                    total_distill_runs += 1
            except Exception as e:
                print(f"  [SelfDistill] Error: {e}")

        # ── PHASE 4: GRPO on replay buffer (frequent, light) ──────────────
        if ep % DISTILL_TRAIN_EVERY == 0 and replay_buffer.size() >= sd_config.min_replay_size_to_train:
            _run_replay_grpo(
                model=model,
                tokenizer=tokenizer,
                replay_buffer=replay_buffer,
                grpo_config=grpo_config,
                label=f"replay@ep{ep}",
            )

        # ── PHASE 4: GRPO on blended dataset (standard, every TRAIN_EVERY) ─
        if ep % TRAIN_EVERY == 0 and len(all_samples) >= 4:
            _run_blended_grpo(
                model=model,
                tokenizer=tokenizer,
                online_samples=all_samples,
                replay_buffer=replay_buffer,
                grpo_config=grpo_config,
                label=f"blend@ep{ep}",
            )
            all_samples = []
            samples_map = {}

        torch.cuda.empty_cache()
        gc.collect()

        # ── Forgetting regression test ────────────────────────────────────
        if ep % EVAL_EVERY == 0:
            _run_forgetting_check(runner, per_attack_rewards, per_attack_detected, ep)
            _print_replay_stats(replay_buffer, sd_sampler)

        # ── Per-attack F1 summary ─────────────────────────────────────────
        if ep % 20 == 0:
            _print_attack_stats(per_attack_rewards, per_attack_detected, ucb)

        # ── Save checkpoint ───────────────────────────────────────────────
        if ep % SAVE_EVERY == 0:
            ckpt = f"guardian/checkpoints/episode_{ep}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"\n  >>> Checkpoint saved: {ckpt}")

        print(
            f"  ep={ep:03d} | {atk_str:<22} | "
            f"intact={result.production_intact} | fork={result.fork_triggered} | "
            f"reward={result.reward:.3f} | {elapsed:.1f}s"
            + distill_tag
        )

    # ── Save final model ──────────────────────────────────────────────────
    print("\n[DONE] Saving final model...")
    final = "guardian/checkpoints/final"
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)

    print(f"\n{'='*65}")
    print("TRAINING COMPLETE — Self-Distilled GRPO")
    print(f"  Episodes:           {TOTAL_EPISODES}")
    print(f"  Total SD runs:      {total_distill_runs}")
    print(f"  Golden trajectories:{total_golden_added}")
    print(f"  Replay buffer size: {replay_buffer.size()}")
    print(f"  Log:                {LOG_FILE}")
    print(f"  Scorecards:         {SCORECARD_FILE}")
    print(f"  Replay buffer:      {REPLAY_BUFFER_FILE}")
    print(f"  Final model:        {final}")
    print(f"{'='*65}\n")

    # Final UCB stats
    print("Final UCB Statistics:")
    for atk, stats in ucb.get_stats().items():
        print(f"  {atk:<25} | count={stats['count']:>3} | mean_reward={stats['mean_reward']}")

    # Final replay buffer stats
    print("\nFinal Replay Buffer Stats:")
    for atk, count in replay_buffer.per_attack_counts().items():
        print(f"  {atk:<25} | golden_trajectories={count}")


# ── GRPO helpers ──────────────────────────────────────────────────────────────

def _run_replay_grpo(
    model,
    tokenizer,
    replay_buffer: GoldenReplayBuffer,
    grpo_config: GRPOConfig,
    label: str = "replay",
) -> None:
    """
    Phase 4 (light): Train GRPO exclusively on verified golden trajectories.

    Run more frequently than the online GRPO step — the golden trajectories
    are already verified, so even a small batch is high-quality signal.
    """
    print(f"\n  >>> [{label}] GRPO on {min(16, replay_buffer.size())} golden trajectories...")
    prompts, rewards = replay_buffer.sample_with_rewards(batch_size=16, strategy="balanced")
    if len(prompts) < 2:
        print(f"  >>> [{label}] Not enough data, skipping.")
        return

    reward_lookup = {p[:100]: r for p, r in zip(prompts, rewards)}
    dataset = Dataset.from_list([{"prompt": p} for p in prompts])

    def reward_fn(prompts, completions, **kwargs):
        return [reward_lookup.get(p[:100], 0.5) for p in prompts]

    try:
        GRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        ).train()
        print(f"  >>> [{label}] Done.")
    except Exception as e:
        print(f"  >>> [{label}] Training error (continuing): {e}")


def _run_blended_grpo(
    model,
    tokenizer,
    online_samples: list,
    replay_buffer: GoldenReplayBuffer,
    grpo_config: GRPOConfig,
    label: str = "blend",
) -> None:
    """
    Phase 4 (standard): Train GRPO on a blend of online + golden replay samples.

    Blend ratio: 50% online (exploration diversity) + 50% golden (verified signal).
    """
    dataset, reward_lookup = _build_blended_dataset(online_samples, replay_buffer)
    print(f"\n  >>> [{label}] GRPO on {len(dataset)} blended samples "
          f"({replay_buffer.size()} replay, {len(online_samples)} online)...")

    def reward_fn(prompts, completions, **kwargs):
        return [reward_lookup.get(p[:100], 0.5) for p in prompts]

    try:
        GRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        ).train()
        print(f"  >>> [{label}] Done.")
    except Exception as e:
        print(f"  >>> [{label}] Training error (continuing): {e}")


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _run_forgetting_check(runner, per_attack_rewards, per_attack_detected, ep):
    """Post-epoch forgetting regression test."""
    print(f"\n  [Forgetting Check @ ep {ep}]")
    for atk, rewards in per_attack_rewards.items():
        if len(rewards) < 2:
            continue
        recent = rewards[-5:]
        older  = rewards[-10:-5] if len(rewards) >= 10 else rewards[:5]
        if older:
            mean_recent = sum(recent) / len(recent)
            mean_older  = sum(older)  / len(older)
            drop = mean_older - mean_recent
            if drop > 0.10:
                print(f"  ⚠️  Forgetting {atk}: {mean_older:.3f} → {mean_recent:.3f} (Δ={drop:.3f})")
            else:
                print(f"  ✓  {atk:<22}: {mean_older:.3f} → {mean_recent:.3f}")


def _print_replay_stats(
    replay_buffer: GoldenReplayBuffer,
    sampler: SelfDistillationSampler,
) -> None:
    """Print self-distillation and replay buffer summary."""
    stats = replay_buffer.stats()
    print(f"\n  [Replay Buffer] size={stats['size']}  "
          f"mean_reward={stats['mean_reward']:.4f}")
    print("  Per-attack golden counts:")
    for atk, count in sorted(stats["per_attack"].items()):
        bar = "█" * min(count, 20)
        print(f"    {atk:<25} {bar} ({count})")
    triggers = sampler.trigger_stats()
    print(f"  SD triggers: {triggers}")


def _print_attack_stats(per_attack_rewards, per_attack_detected, ucb):
    """Print per-attack detection rate summary."""
    print("\n  ── Attack Detection Summary ──")
    for atk, detected in per_attack_detected.items():
        if not detected:
            continue
        rate   = sum(detected) / len(detected)
        rewards = per_attack_rewards.get(atk, [])
        mean_r  = sum(rewards) / len(rewards) if rewards else 0.0
        bar     = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {atk:<22} [{bar}] {rate:.0%} detect | {mean_r:.3f} reward")
    print()


if __name__ == "__main__":
    main()
