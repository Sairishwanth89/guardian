# GUARDIAN Fleet: Training Protocol-Level AI Firewalls with PBRS

*(This writeup is designed for your Hugging Face Space blog post or video script).*

## Watch the 2-Minute Demo / Pitch 
[Insert YouTube Link Here]

---

### 1. The Capability Gap
Enterprise AI is moving incredibly fast, from simple RAG to fully autonomous agents given access to APIs and internal databases. While we have basic guardrails for conversational LLMs, there is a **massive capability gap** in how we protect *enterprises* from *agentic behavior*.

If a rogue agent or prompt injection triggers an LLM to dump a database, traditional monitoring tools fail because they don't understand the semantic intent of multi-step AI reasoning.

### 2. The Environment: GUARDIAN
We built the **GUARDIAN Fleet**. It is a Multi-Agent cybersecurity ecosystem built on top of **OpenEnv**. 
Instead of hacking Python scripts, GUARDIAN acts as a **Model Context Protocol (MCP) Centralized Gateway**. It intercepts standardized JSON-RPC 2.0 payloads flying between an active LLM Worker (mimicking Finance, HR, or Ops roles) and an Enterprise Graph Database.

**What the AI sees:** A live, uncapped stream of cross-app logs (Slack, IAM, Data Access) and the MCP payloads.
**What the AI does:** Instead of just "blocking", GUARDIAN wields advanced tactical interventions like:
*   **The Transparent Interceptor:** Dynamically routing malicious payloads into a honeypot database without the worker knowing.
*   **The Rug Pull:** Dynamically deprovisioning tools from the worker's context window.

### 3. The 13-Component Reward Engine & PBRS
The biggest failure mode in RL systems today is **Reward Hacking**—models figure out how to farm scores without solving the task by exploiting the environment's timers, looping harmless steps, or relying on LLM-as-a-judge bias.

We solved this permanently using **Potential-Based Reward Shaping (PBRS)**. By mathematically evaluating the state potential $\Phi(s)$ and tying our process-supervision shaping strictly to a zero-sum gamma transition ($F(s, s') = \gamma \Phi(s') - \Phi(s)$), we mathematically guarantee that shaping rewards cannot alter the optimal policy. Final rewards are completely deterministic, verified by a cryptographic SHA-256 hash of the production database at termination.

### 4. Results & Why It Matters
Using **Unsloth** and **TRL GRPO**, we fine-tuned Qwen2.5-7B against a Co-Evolutionary Adversary—an engine that dynamically generated 9 attacks across 3 severity levels, including a terrifying **Meta Sev-1 Incident Tracker** (an internal rogue AI attempting to leak data through an IAM vulnerability).

Compared to an untrained baseline that effectively failed at everything except throwing false-alarms, the trained GUARDIAN achieved an **88% Advanced Detection F1** score while slashing false alarms. More importantly, it successfully learned to trap attackers inside the honeypot for an average of **480+ tokens**, isolating maximum adversarial intelligence while keeping the production hash perfectly intact.

GUARDIAN proves that Protocol-Level AI Firewalls are not just possible, but essential for the future of enterprise multi-agent deployment.
