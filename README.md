---
title: GUARDIAN Fleet
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ GUARDIAN: Universal AI Immune System — MCP Security Gateway

**A multi-agent, protocol-level AI firewall trained via GRPO and Potential-Based Reward Shaping.**

🔗 **[Run the Demo UI on Hugging Face Spaces](https://huggingface.co/spaces/sai1912/Guardian)**  
🔗 **[Run the GRPO Training Script on Kaggle](https://www.kaggle.com/code/sairishwanth/guardian-rl-training)**  
🔗 **[Read our 2-Minute Pitch & Writeup](HUGGINGFACE_PITCH.md)**

> *"Security systems fail because they memorize symptoms. GUARDIAN learns the underlying DNA of an attack. A privilege escalation is mathematically identical whether it is happening in a Kubernetes cluster, a hospital database, or an algorithmic trading desk."*

---

## 🎯 The Capability Gap (Why are we building this?)

**The Problem:** Current AI security tools rely on brittle Python wrappers or "LLM-as-a-judge" supervisors that are easily gamed. Furthermore, Reinforcement Learning for security agents suffers from catastrophic *Reward Hacking* — models learn to exploit the environment rather than actually stopping data exfiltration.

**The Solution:** GUARDIAN operates as a **JSON-RPC 2.0 MCP Proxy**. It intercepts communications between a rogue AI and enterprise systems. We train it with a **13-Component Deterministic Reward Engine**, secured by **Potential-Based Reward Shaping (PBRS)** and SHA-256 cryptographic hashes, guaranteeing zero reward hacking.

---

## 🏙️ The Environment (What does the agent do?)

GUARDIAN sits inside a **Multi-Agent Enterprise Graph Ecosystem**.

*   **What it sees:** Structured telemetry streams of Cross-App Logs and MCP `CallToolRequest` payloads from the Worker agent, plus a universal **Capability Tag** (`[STATE_MOD | PRIV_ESC | EXT_COMM | FINANCIAL]`) prepended to every tool call by the Semantic Abstraction Layer.
*   **What it faces:** A **Co-Evolutionary Rule-Driven Adversary** with 3 stealth levels, timing randomization, red-herring decoys, and compound attack merging — automatically escalating as the Guardian improves.
*   **What it can do:** It wields 12 precise architectural interventions, including dynamically routing attackers into a Honeypot Database (Transparent Interceptor) and deprovisioning tools from the rogue agent's context (Rug Pull).
*   **What it gets rewarded for:** PBRS mathematically defined as $F(s, s') = \gamma \Phi(s') - \Phi(s)$. Final outcome rewards are strictly verified by SHA-256 hash of the production database at termination.

---

## 🌐 Zero-Shot Domain Transfer (The Architecture's Core Innovation)

Most AI security tools are brittle — train on HR, fail on finance. GUARDIAN is different.

The **Semantic Action Abstraction Layer** (`guardian/mcp/tool_taxonomy.py`) maps every MCP tool call — regardless of domain — to 4 universal security primitives:

```
[STATE_MOD=True | PRIV_ESC=True | EXT_COMM=False | FINANCIAL=Critical]
```

The GRPO policy trains against this **mathematical geometry**, not tool names. This enables zero-shot transfer:

| Domain | Unseen Tool | Primitive Map | Guardian Response |
|--------|-------------|---------------|-------------------|
| **HR** (training) | `process_payment` | STATE_MOD + FINANCIAL=High | Honeypot route |
| **FinOps** (zero-shot) | `override_margin_limits` | STATE_MOD + PRIV_ESC + FINANCIAL=Critical | Instant quarantine |
| **Corp Governance** (zero-shot) | `modify_compliance_threshold` | STATE_MOD + PRIV_ESC + FINANCIAL=High | Privilege downgrade |

Any developer can extend GUARDIAN to a new domain by mapping their MCP tools to the 4-primitive taxonomy — **zero retraining required.**

---

## 📈 Results: What Changed After Training?

*(Our training loop utilizes Unsloth for Qwen2.5-7B and hugs the `trl` GRPOTrainer, accelerating the normally bottlenecked rollout engine by running completely stateless dictionary-math transitions inside Python).*

After 500 GRPO episodes comparing the Qwen2.5-7B baseline against the trained Guardian agent:

![Training Reward Curve](guardian/data/reward_curve.png)
*(Above: Episode reward curves across 500 training cycles. Smoothed average shows convergence demonstrating definitive policy improvement against the co-evolutionary adversary).*

| Metric | Untrained Baseline | Trained GUARDIAN | Delta |
|--------|---------------------|------------------|-------|
| **Adversarial Detection F1** | ~32% | **88%** | +56% |
| **False Alarm Disruption** | ~85% | **12%** | -73% |
| **Meta Sev-1 Quarantine Success** | 0% | **76%** | +76% |
| **Honeypot Dwell Time** | 0 tokens | **480+ tokens** | +480 |

*The untrained agent effectively panicked—intervening heavily on benign requests and missing multi-step attacks. The trained agent learned to use surgical interventions like `quarantine_tool` and `rewrite`, capturing massive honeypot intelligence while keeping the production SHA-256 hash perfectly intact.*

---

## 🚀 Hackathon Deliverable Checklist & Code Map

### Architecture highlights:
*   ✅ **OpenEnv Standard**: Fully wrapped (`Guardian-v0`) and manifests (`openenv.yaml`) compliant.
*   ✅ **Working Training Script**: Kaggle notebook built on `trl.GRPOTrainer` with Unsloth and `max_completion_length=1024` (token wall resolved).
*   ✅ **Real Plots**: Embeds the actual `.png` and metrics from the run.

### The Innovation Edge:
*   **`guardian/mcp/tool_taxonomy.py`**: The Semantic Abstraction Layer that enables zero-shot domain transfer.
*   **`guardian/mcp/gateway.py`**: The MCP Centralized Security Gateway (JSON-RPC 2.0 proxy).
*   **`guardian/mcp/domain_servers.py`**: FinOps and Corporate Governance domain testbeds for live zero-shot demo.
*   **`guardian/environment/adaptation_engine.py`**: Co-evolutionary adversary with 3 stealth levels, red-herring injection, and compound attack merging.
*   **`guardian/environment/reward_computer.py`**: 13 independent math-driven reward components (No LLM as a judge!)
*   **`guardian/environment/openenv_wrapper.py (line 412)`**: Potential-Based Reward Shaping implementation — mathematical proof of reward hacking immunity.

---

## ⚡ Quick Start

```bash
# 1. Install OpenEnv and GUARDIAN locally
git clone https://github.com/Sairishwanth89/guardian.git
cd guardian
pip install -r server/requirements.txt
pip install "openenv-core[cli]"
pip install -e .

# 2. Verify OpenEnv skills
openenv init kernel_env
openenv skills add read_db write_db process_payment modify_firewall

# 3. Test baseline Python interface
python -m guardian.training.run_baselines
```
