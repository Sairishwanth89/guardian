"""
GUARDIAN Fleet — OpenEnv Package Root
======================================
Import GuardianAction, GuardianObservation, GuardianEnv to interact with this environment.

Quick start (async):
    from guardian_env import GuardianAction, GuardianObservation, GuardianEnv

    async with GuardianEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        action = GuardianAction(
            risk_score=0.85,
            intervention="shadow",
            attack_type="prompt_injection",
        )
        result = await env.step(action)
        print(result.observation.production_intact)  # True
        print(result.reward)

Quick start (sync):
    from guardian_env import GuardianEnv, GuardianAction
    with GuardianEnv("http://localhost:8000").sync() as env:
        obs = env.reset().observation
        result = env.step(GuardianAction(intervention="shadow"))
"""

from __future__ import annotations

# Relative imports work when installed as a package;
# absolute imports work when running from the repo root (CWD in sys.path).
try:
    from .models import GuardianAction, GuardianObservation, GuardianState, StepResult
    from .client import GuardianEnv
except ImportError:
    from models import GuardianAction, GuardianObservation, GuardianState, StepResult  # type: ignore
    from client import GuardianEnv  # type: ignore

__version__ = "0.2.0"
__all__ = [
    "GuardianAction",
    "GuardianObservation",
    "GuardianState",
    "StepResult",
    "GuardianEnv",
]
