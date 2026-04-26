"""
GUARDIAN OpenEnv Architecture Validation Script
================================================
Validates that all required OpenEnv components are present and importable.
Run from the project root: python validate_openenv.py
"""

import os
import sys
import json
import importlib
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, fn):
    try:
        msg = fn()
        print(f"  {PASS}  {name}" + (f" — {msg}" if msg else ""))
        results.append((name, True, msg))
    except Exception as e:
        print(f"  {FAIL}  {name} — {e}")
        results.append((name, False, str(e)))

print("\n" + "="*60)
print("  GUARDIAN OpenEnv Architecture Validation")
print("="*60)

# ── 1. Required files ────────────────────────────────────────────
print("\n[1] Required files present")
required_files = [
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "models.py",
    "client.py",
    "server/app.py",
    "server/requirements.txt",
    "server/guardian_environment.py",
    "guardian/__init__.py",
    "guardian/environment/guardian_env.py",
    "guardian/environment/reward_computer.py",
    "guardian/agents/guardian_agent.py",
    "guardian/agents/worker_agent.py",
    "guardian/mcp/gateway.py",
]
for f in required_files:
    check(f, lambda f=f: "exists" if os.path.exists(os.path.join(ROOT, f)) else (_ for _ in ()).throw(FileNotFoundError(f"MISSING: {f}")))

# ── 2. Dockerfile validation ─────────────────────────────────────
print("\n[2] Dockerfile validation")
def check_dockerfile():
    content = open(os.path.join(ROOT, "Dockerfile")).read()
    assert "FROM python" in content, "Missing FROM python"
    assert "EXPOSE 7860" in content, "Missing EXPOSE 7860"
    assert "uvicorn server.app:app" in content, "Missing uvicorn entrypoint"
    assert "HEALTHCHECK" in content, "Missing HEALTHCHECK"
    return f"{content.count(chr(10))} lines"
check("Dockerfile structure", check_dockerfile)

# ── 3. openenv.yaml validation ───────────────────────────────────
print("\n[3] openenv.yaml validation")
def check_yaml():
    import yaml
    content = open(os.path.join(ROOT, "openenv.yaml")).read()
    data = yaml.safe_load(content)
    assert "entry_point" in data, "Missing entry_point"
    assert "action_type" in data, "Missing action_type"
    assert "observation_type" in data, "Missing observation_type"
    n_attacks = len(data.get("metadata", {}).get("attack_types", []))
    n_rewards = data.get("metadata", {}).get("reward_components", 0)
    assert n_attacks >= 9, f"Only {n_attacks} attack types declared"
    assert n_rewards >= 13, f"Only {n_rewards} reward components"
    return f"{n_attacks} attacks, {n_rewards} reward components"

try:
    import yaml
    check("openenv.yaml structure", check_yaml)
except ImportError:
    print(f"  {WARN}  openenv.yaml — pyyaml not installed, skipping YAML parse")

# ── 4. Python imports ────────────────────────────────────────────
print("\n[4] Core Python imports")
imports = [
    ("models", "GuardianAction, GuardianObservation, GuardianState"),
    ("guardian.environment.guardian_env", "GUARDIANEnvironment"),
    ("guardian.environment.reward_computer", "RewardComputer"),
    ("guardian.agents.guardian_agent", "GuardianAgent"),
    ("guardian.agents.worker_agent", "WorkerAgent"),
    ("guardian.mcp.gateway", "MCPGateway, ATTACK_MCP_ROUTING"),
]
for mod, symbols in imports:
    def try_import(mod=mod, symbols=symbols):
        m = importlib.import_module(mod)
        for s in [x.strip() for x in symbols.split(",")]:
            assert hasattr(m, s), f"{s} not found in {mod}"
        return f"OK ({symbols})"
    check(f"import {mod}", try_import)

# ── 5. MCP Gateway routing completeness ─────────────────────────
print("\n[5] MCP Gateway routing table")
def check_routing():
    from guardian.mcp.gateway import ATTACK_MCP_ROUTING
    from guardian.environment.guardian_env import ATTACK_PATTERNS
    
    all_attacks = set(ATTACK_PATTERNS.keys())
    routed = set(ATTACK_MCP_ROUTING.keys()) - {"clean"}
    missing = all_attacks - routed
    
    assert not missing, f"UNROUTED attacks: {missing}"
    assert "clean" in ATTACK_MCP_ROUTING, "Missing 'clean' fallback entry"
    
    # Verify each entry has required fields
    for name, cfg in ATTACK_MCP_ROUTING.items():
        assert "ir_action" in cfg, f"{name}: missing ir_action"
        assert "audit_level" in cfg, f"{name}: missing audit_level"
    
    return f"{len(ATTACK_MCP_ROUTING)} entries, all valid"
check("Attack routing completeness", check_routing)

# ── 6. RewardComputer sanity ─────────────────────────────────────
print("\n[6] RewardComputer sanity")
def check_reward():
    from guardian.environment.reward_computer import RewardComputer
    rc = RewardComputer()
    result = rc.compute(
        production_intact=True,
        fork_triggered=True,
        fork_step=2,
        attack_active=True,
        attack_type="prompt_injection",
        guardian_detected_type="prompt_injection",
        last_worker_tool="write_db",
        shadow_tokens=300,
        guardian_tokens=500,
        intervention="shadow",
        risk_score=0.85,
        step_rewards=[0.1, 0.3],
        explanation="Detected prompt injection via write_db call pattern",
        action_log=[{"step":1,"tool":"write_db","parameters":{}}],
    )
    assert 0.0 <= result.total <= 2.0, f"Reward out of range: {result.total}"
    return f"reward={result.total:.4f}, {len(result.to_dict())} components"
check("RewardComputer.compute()", check_reward)

# ── 7. WorkerAgent (no-API-key mode) ─────────────────────────────
print("\n[7] WorkerAgent heuristic mode (no OpenAI)")
def check_worker():
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    from guardian.agents.worker_agent import FinanceWorker
    w = FinanceWorker()
    action = w.get_action("Process vendor payment of $500 to supplier X")
    assert "tool" in action, "Missing 'tool' key in action"
    assert "parameters" in action, "Missing 'parameters' key"
    return f"tool={action['tool']} (heuristic mode OK)"
check("WorkerAgent.get_action() heuristic", check_worker)

# ── 8. GUARDIANEnvironment episode ───────────────────────────────
print("\n[8] Environment episode smoke test")
def check_env():
    from guardian.environment.guardian_env import GUARDIANEnvironment
    env = GUARDIANEnvironment()
    state = env.reset()
    assert state is not None
    return "reset() OK"
check("GUARDIANEnvironment.reset()", check_env)

# ── 9. server/guardian_environment imports ────────────────────────
print("\n[9] Server environment")
def check_server_env():
    from server.guardian_environment import GuardianOpenEnvEnvironment
    env = GuardianOpenEnvEnvironment(difficulty=1)
    obs = env.reset()
    assert hasattr(obs, "current_step"), "Missing current_step"
    assert hasattr(obs, "action_log_json"), "Missing action_log_json"
    state = env.state()
    assert hasattr(state, "episode_id"), "Missing episode_id"
    return f"episode_id={state.episode_id[:8]}..."
check("GuardianOpenEnvEnvironment HTTP env", check_server_env)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "="*60)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"  Results: {passed}/{total} checks passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    print("\n  Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"    ✗ {name}: {msg}")
else:
    print("  -- ALL PASSED (OK)")
print("="*60 + "\n")

sys.exit(0 if failed == 0 else 1)
