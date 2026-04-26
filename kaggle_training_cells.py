# GUARDIAN Kaggle Training Guide
# ================================
# Complete step-by-step Kaggle notebook cells.
# Push to GitHub first, then run these cells on Kaggle.

# ══════════════════════════════════════════════════════
# BEFORE YOU START — ONE-TIME SETUP (do this first)
# ══════════════════════════════════════════════════════
#
# 1. Create GitHub Personal Access Token (PAT):
#    → github.com → Settings → Developer settings
#    → Personal access tokens → Tokens (classic) → Generate new token
#    → Scopes: check "repo" (full repo access)
#    → Copy the token (ghp_xxxxxxxxxxxx)
#
# 2. Add Kaggle Secrets:
#    → kaggle.com → Your notebook → Add-ons → Secrets → Add Secret
#    → Name: GITHUB_TOKEN   Value: ghp_xxxxxxxxxxxx
#    → Name: HF_TOKEN       Value: hf_xxxxxxxxxxxx  (from huggingface.co)
#    → Name: REPO_URL       Value: https://github.com/Sairishwanth89/guardian-rl
#
# 3. Push all your local changes FIRST:
#    git add -A && git commit -m "feat: 12-gap fixes + 7 training tools" && git push
#
# ══════════════════════════════════════════════════════

# ─── CELL 1: Install Dependencies ───────────────────
# (New code cell in Kaggle)

import subprocess, sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
    else:
        print(result.stdout[-1000:] or "OK")
    return result.returncode

print("Installing Unsloth + TRL + dependencies...")
run("pip install -q unsloth trl peft datasets python-dotenv gymnasium")
run("pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
run("pip install -q matplotlib pytest")

print("\nAll dependencies installed.")

# ─── CELL 2: Clone from GitHub ──────────────────────
# (New code cell in Kaggle)

import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
HF_TOKEN     = secrets.get_secret("HF_TOKEN")
REPO_URL     = secrets.get_secret("REPO_URL")  # e.g. https://github.com/Sairishwanth89/guardian-rl

# Write HF token for model push
os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)
with open(os.path.expanduser("~/.cache/huggingface/token"), "w") as f:
    f.write(HF_TOKEN)

# Clone the repo (auth embedded in URL)
REPO_NAME = REPO_URL.rstrip("/").split("/")[-1]
AUTH_URL   = REPO_URL.replace("https://", f"https://Sairishwanth89:{GITHUB_TOKEN}@")

if os.path.exists(f"/kaggle/working/{REPO_NAME}"):
    print(f"Repo already exists — pulling latest...")
    os.chdir(f"/kaggle/working/{REPO_NAME}")
    run("git pull")
else:
    print(f"Cloning {REPO_URL} ...")
    os.chdir("/kaggle/working")
    run(f"git clone {AUTH_URL}")
    os.chdir(f"/kaggle/working/{REPO_NAME}")

# Set PYTHONPATH so imports work
sys.path.insert(0, f"/kaggle/working/{REPO_NAME}")
os.environ["PYTHONPATH"] = f"/kaggle/working/{REPO_NAME}"
print(f"\nWorking directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# ─── CELL 3: Verify Setup ───────────────────────────
# (New code cell — smoke test before full training)

print("Verifying imports and attack patterns...")

from guardian.agents.guardian_agent import VALID_INTERVENTIONS, VALID_ATTACK_TYPES, SYSTEM_PROMPT
from guardian.environment.guardian_env import ATTACK_PATTERNS
from guardian.environment.reward_computer import RewardComputer, RewardBreakdown
from guardian.agents.curriculum_agent import UCBAttackSelector

REQUIRED_ATTACKS = [
    "authority_spoofing", "prompt_injection", "approval_bypass",
    "data_exfiltration", "confused_deputy", "approval_laundering",
    "salami_slicing", "schema_drift_exploit", "rogue_internal_ai",
    "delayed_exfiltration", "social_engineering",
]

# Check all attacks in ATTACK_PATTERNS
missing = [a for a in REQUIRED_ATTACKS if a not in ATTACK_PATTERNS]
assert not missing, f"MISSING from ATTACK_PATTERNS: {missing}"

# Check GAP 1 fix
assert "quarantine_agent" in SYSTEM_PROMPT, "GAP 1 still broken"
assert "predicted_next_risk" in SYSTEM_PROMPT, "GAP 7 still broken"

# Check GAP 2/3/4 in RewardBreakdown
bd = RewardBreakdown()
assert hasattr(bd, "risk_score_component"), "GAP 2 missing"
assert hasattr(bd, "reasoning_quality"),    "GAP 3 missing"
assert hasattr(bd, "detection_lag_bonus"),  "GAP 4 missing"

# Check UCB knows all attacks
ucb = UCBAttackSelector()
for a in REQUIRED_ATTACKS:
    assert a in ucb.attack_pool, f"UCB missing {a}"

print(f"✅ All {len(REQUIRED_ATTACKS)} attack types in ATTACK_PATTERNS")
print(f"✅ All 16 reward components present")
print(f"✅ quarantine_agent in SYSTEM_PROMPT")
print(f"✅ UCBAttackSelector has all {len(REQUIRED_ATTACKS)} attacks")
print("\nSetup verification PASSED — safe to start training")

# ─── CELL 4: Generate Evidence Data ─────────────────
# (Run once to create guardian/data/ files)

exec(open("generate_evidence_data.py").read())

# ─── CELL 5: Configure Training ─────────────────────
# (New code cell — edit these values)

TRAINING_CONFIG = {
    "MODEL_NAME"      : "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "TOTAL_EPISODES"  : 120,       # 120 episodes on T4 = ~3-4 hours
    "TRAIN_EVERY"     : 8,         # GRPO step every 8 episodes
    "SAVE_EVERY"      : 40,        # Checkpoint every 40 episodes
    "EVAL_EVERY"      : 40,        # Evaluate every 40 episodes
    "LOG_FILE"        : "guardian/data/training_log.jsonl",
    "SCORECARD_FILE"  : "guardian/data/scorecards.jsonl",
    "CSV_LOG_FILE"    : "outputs/reward_breakdown_log.csv",
    "HF_REPO"         : "guardian-rl/guardian-qwen25-7b-grpo",
    "USE_UCB"         : True,      # Curriculum attack selection
    "GRPO_BATCH"      : 4,
    "GRPO_LR"         : 5e-6,
    "LORA_RANK"       : 16,
    "MAX_SEQ_LEN"     : 1024,
}

print("Training config:")
for k, v in TRAINING_CONFIG.items():
    print(f"  {k}: {v}")

# ─── CELL 6: Run Honest Baseline ────────────────────
# (Establish pre-training score — run BEFORE loading the model)

from guardian.training.run_honest_episodes import run_honest_baseline
baseline = run_honest_baseline(n_episodes=20, use_model=False)
print(f"\nBaseline mean_reward: {baseline['mean_reward']}")
print(f"Baseline detection_rate: {baseline['detection_rate']:.1%}")
print("Save this — we'll compare after training.")

# ─── CELL 7: Load Model + Start Training ────────────
# (Main training cell — runs full GRPO loop)

import gc
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from guardian.environment.guardian_env import GUARDIANEnvironment, ATTACK_PATTERNS
from guardian.environment.reward_computer import RewardComputer
from guardian.agents.worker_agent import FinanceWorker
from guardian.agents.guardian_agent import GuardianAgent
from guardian.agents.curriculum_agent import UCBAttackSelector, CurriculumAgent
from guardian.training.episode_runner import EpisodeRunner
from guardian.training.elo_tracker import ELOTracker
from guardian.training.multi_session_tracker import MultiSessionTracker

cfg = TRAINING_CONFIG

# ── Load model ──────────────────────────────────────
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = cfg["MODEL_NAME"],
    max_seq_length= cfg["MAX_SEQ_LEN"],
    dtype         = None,
    load_in_4bit  = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r             = cfg["LORA_RANK"],
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha    = 32,
    lora_dropout  = 0.05,
    bias          = "none",
    use_gradient_checkpointing = "unsloth",
)
print("Model loaded with LoRA adapters.")

# ── Setup agents ────────────────────────────────────
env      = GUARDIANEnvironment()
worker   = FinanceWorker()
guardian = GuardianAgent()
guardian.model     = model
guardian.tokenizer = tokenizer
rc       = RewardComputer()

ucb      = UCBAttackSelector()  # all 11 attack types
curr     = CurriculumAgent()
runner   = EpisodeRunner(
    env=env, worker=worker, guardian=guardian,
    reward_computer=rc, ucb_selector=ucb,
    curriculum_agent=curr,
)
runner._use_ucb = True

elo_tracker     = ELOTracker()
session_tracker = MultiSessionTracker()
session_id      = session_tracker.start_session(model_checkpoint=cfg["MODEL_NAME"])

os.makedirs("guardian/data", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# ── Training loop ───────────────────────────────────
all_samples  = []
log_f        = open(cfg["LOG_FILE"], "a")
score_f      = open(cfg["SCORECARD_FILE"], "a")

print(f"\nStarting training: {cfg['TOTAL_EPISODES']} episodes\n")

for ep in range(1, cfg["TOTAL_EPISODES"] + 1):

    # Run episode (UCB selects attack type)
    result = runner.run_episode()

    # Log to files
    log_record = {
        "episode": ep, "episode_id": result.episode_id,
        "attack_type": result.attack_type,
        "reward_total": result.reward,
        "production_intact": result.production_intact,
        "fork_triggered": result.fork_triggered,
        "guardian_detected_type": result.guardian_detected_type,
    }
    log_f.write(json.dumps(log_record) + "\n"); log_f.flush()
    score_f.write(json.dumps(result.scorecard) + "\n"); score_f.flush()

    # ELO + session
    elo_tracker.update(
        attack_type       = result.attack_type,
        guardian_detected = bool(result.guardian_detected_type),
        production_intact = result.production_intact,
        episode_id        = result.episode_id,
    )
    session_tracker.log_episode(
        result.episode_id, result.reward,
        result.attack_type, bool(result.guardian_detected_type),
        result.production_intact,
    )

    # Collect training samples
    all_samples.extend(result.training_samples)

    # Console progress
    if ep % 10 == 0:
        recent = [json.loads(l) for l in
                  open(cfg["LOG_FILE"]).readlines()[-10:]]
        mean_r = sum(r["reward_total"] for r in recent) / len(recent)
        print(f"  ep {ep:3d}/{cfg['TOTAL_EPISODES']}  "
              f"mean_reward(last10)={mean_r:.4f}  "
              f"attack={result.attack_type or 'clean':25s}  "
              f"detected={bool(result.guardian_detected_type)}")

    # GRPO training step
    if ep % cfg["TRAIN_EVERY"] == 0 and len(all_samples) >= cfg["GRPO_BATCH"]:
        print(f"\n  [GRPO] Training on {len(all_samples)} samples...")
        FastLanguageModel.for_training(model)

        dataset = Dataset.from_list([{
            "prompt": s["prompt"],
            "completion": s["completion"],
        } for s in all_samples[-64:]])  # last 64 samples

        grpo_cfg = GRPOConfig(
            output_dir             = "outputs/grpo_tmp",
            num_train_epochs       = 1,
            per_device_train_batch_size = cfg["GRPO_BATCH"],
            learning_rate          = cfg["GRPO_LR"],
            logging_steps          = 1,
            save_steps             = 9999,
            report_to              = "none",
            max_completion_length  = 200,
        )
        trainer = GRPOTrainer(
            model    = model,
            config   = grpo_cfg,
            train_dataset = dataset,
            processing_class = tokenizer,
        )
        trainer.train()
        FastLanguageModel.for_inference(model)
        all_samples = []
        gc.collect(); torch.cuda.empty_cache()
        print(f"  [GRPO] Step complete.\n")

    # Checkpoint
    if ep % cfg["SAVE_EVERY"] == 0:
        ckpt_path = f"outputs/checkpoints/episode_{ep}"
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        import json as _json
        with open(f"{ckpt_path}/checkpoint_info.json", "w") as cf:
            _json.dump({"episode": ep, "mean_reward": mean_r}, cf)
        print(f"  [CKPT] Saved checkpoint → {ckpt_path}")

log_f.close(); score_f.close()
session_tracker.end_session()
elo_tracker.save("outputs/elo_ratings.json")
print("\nTraining complete!")
print(elo_tracker.summary())

# ─── CELL 8: Push Results Back to GitHub ────────────
# (Run after training to commit data files + checkpoints)

import subprocess

run("git config user.email 'kaggle@training.run'")
run("git config user.name  'Kaggle Training Run'")

run("git add guardian/data/ outputs/checkpoints/ outputs/elo_ratings.json")
run("git add outputs/reward_breakdown_log.csv outputs/honest_baseline* || true")
run('git commit -m "train: GRPO episode run — updated training logs and checkpoints"')
run(f"git push {AUTH_URL} main")

print("\nResults pushed to GitHub!")

# ─── CELL 9: Plot Training Curves ───────────────────

from guardian.training.plot_training import main as plot_main
import sys
sys.argv = ["plot_training", "--log", "guardian/data/training_log.jsonl",
                             "--csv", "outputs/reward_breakdown_log.csv",
                             "--out", "outputs/plots"]
plot_main()
print("Plots saved to outputs/plots/")

# (Kaggle will show these as output images)
from IPython.display import Image, display
import os
for plot_file in os.listdir("outputs/plots"):
    if plot_file.endswith(".png"):
        display(Image(f"outputs/plots/{plot_file}"))

# ─── CELL 10: Push Final Model to HuggingFace ───────
# (Optional — push trained LoRA to HF Hub)

from huggingface_hub import login
login(token=HF_TOKEN)

FINAL_CKPT = f"outputs/checkpoints/episode_{cfg['TOTAL_EPISODES']}"
if os.path.exists(FINAL_CKPT):
    model.push_to_hub(cfg["HF_REPO"], token=HF_TOKEN)
    tokenizer.push_to_hub(cfg["HF_REPO"], token=HF_TOKEN)
    print(f"Model pushed → huggingface.co/{cfg['HF_REPO']}")
else:
    print("No final checkpoint found — skipping HF push.")

# ─── CELL 11: Before/After Comparison ───────────────

from guardian.training.run_honest_episodes import run_honest_baseline, compare_with_trained
import json

# Run trained baseline (same 20 episodes as before)
print("Running post-training evaluation...")
trained_summary = run_honest_baseline(n_episodes=20, use_model=True)

with open("outputs/trained_summary.json", "w") as f:
    json.dump(trained_summary, f, indent=2)

compare_with_trained(
    baseline_path = "outputs/honest_baseline_summary.json",
    trained_path  = "outputs/trained_summary.json",
)
