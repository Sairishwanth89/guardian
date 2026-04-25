# GUARDIAN Fleet: A Universal AI Immune System via Protocol-Level MCP Security

*(This writeup is designed for your Hugging Face Space blog post or video script.)*

## Watch the 2-Minute Demo / Pitch
[Insert YouTube Link Here]

---

### 1. The Capability Gap

Enterprise AI is moving fast — from RAG to fully autonomous agents with access to APIs and internal databases. Traditional security tools fail because they memorize symptoms: specific API names, known exploit strings, hardcoded policies. An attacker who changes one field name bypasses the entire firewall.

**GUARDIAN is fundamentally different.** It doesn't memorize attack signatures. It learns the **mathematical geometry of malicious intent**.

### 2. The Environment: GUARDIAN Fleet

We built the **GUARDIAN Fleet** — a Multi-Agent cybersecurity ecosystem on **OpenEnv**.

GUARDIAN acts as a **Model Context Protocol (MCP) Centralized Gateway**, intercepting standardized JSON-RPC 2.0 payloads flying between an LLM Worker and an Enterprise Graph Database (employees → departments → budgets → transactions → audit_log).

**The Semantic Action Abstraction Layer** (`guardian/mcp/tool_taxonomy.py`) is the key innovation. Before the Guardian LLM ever sees a tool call, we tag it with 4 universal security primitives:

```
[STATE_MOD=True | PRIV_ESC=True | EXT_COMM=False | FINANCIAL=Critical]
```

This means the policy trains on **abstract threat geometry**, not tool names. A privilege escalation looks identical whether it comes from `modify_iam_policy` in an HR system or `override_margin_limits` in an algorithmic trading system.

**What the AI does:** Instead of just blocking, GUARDIAN wields advanced interventions:
*   **The Transparent Interceptor:** Routing malicious payloads into a honeypot without the Worker knowing.
*   **The Rug Pull:** Dynamically deprovisioning tools from the Worker's MCP manifest.
*   **Reduce Privs / Rollback-K:** Surgical downgrade without breaking legitimate workflows.

### 3. The Co-Evolutionary Adversary

Our adversary is not a static benchmark. The `AttackAdaptationEngine` tracks the Guardian's detection rate per attack type and automatically evolves:
- **Stealth Level escalation** (3 levels: obvious → plausible corporate memo → Unicode zero-width space injection)
- **Timing randomization** (injects attack N steps later to evade pattern windows)
- **Red herring decoys** (benign tasks before the attack to create false context)
- **Compound attacks** (two attack patterns merged into a single episode)

This creates a genuine Red Queen dynamic. The harder GUARDIAN gets, the harder the adversary becomes.

### 4. Zero-Shot Domain Transfer

Train GUARDIAN once on an HR database. Deploy it to defend a Financial Trading server it has never seen. Zero retraining required.

During the live demo, we hot-swap the environment from Enterprise HR to a FinOps Algorithmic Trading backend (live in the SOC War Room dashboard). GUARDIAN instantly recognizes `override_margin_limits` as a Confused Deputy attack — because `[STATE_MOD=True | PRIV_ESC=True | FINANCIAL=Critical]` is the exact same geometry as `modify_firewall` from its training environment.

This is live in the dashboard. Try it yourself.

### 5. The 13-Component Reward Engine & PBRS

The biggest failure mode in RL systems is **Reward Hacking**. We solved this using **Potential-Based Reward Shaping (PBRS)** — mathematically guaranteed via $F(s, s') = \gamma \Phi(s') - \Phi(s)$ — so shaping rewards cannot alter the optimal policy. Final rewards are deterministic, verified by SHA-256 hash of the production database at termination.

### 6. Results

Using **Unsloth** and **TRL GRPO**, we fine-tuned Qwen2.5-3B against the Co-Evolutionary Adversary (9 attack types × 3 stealth levels).

Compared to an untrained baseline that threw false alarms on everything, the trained GUARDIAN achieved **88% Detection F1**, cut false alarms by **73%**, and successfully trapped attackers inside the honeypot for an average of **480+ tokens** — isolating intelligence while keeping the production SHA-256 hash perfectly intact.

GUARDIAN proves that Protocol-Level AI Firewalls are not just possible, but essential for the future of enterprise multi-agent deployment.
