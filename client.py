"""
GUARDIAN Fleet — OpenEnv Client
=================================
GuardianEnv(EnvClient) — async/sync client for the Guardian FastAPI server.

Usage (async — recommended):
    from client import GuardianEnv
    from models import GuardianAction

    async with GuardianEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation.current_step)   # 0

        action = GuardianAction(
            risk_score=0.85,
            intervention="shadow",
            attack_type="prompt_injection",
            reasoning="Detected suspicious write to exfil_log",
        )
        result = await env.step(action)
        print(result.observation.production_intact)  # True
        print(result.reward)

Usage (sync — via .sync() wrapper):
    with GuardianEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        result = env.step(GuardianAction(intervention="shadow"))
        print(result.observation.fork_triggered)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from models import (
    GuardianAction,
    GuardianObservation,
    GuardianState,
    StepResult,
)

# ── Try openenv-core EnvClient, fall back to pure implementation ────────────
try:
    from openenv.core.client import EnvClient as _OpenEnvBase  # type: ignore
    _HAS_OPENENV = True
except ImportError:
    _OpenEnvBase = object
    _HAS_OPENENV = False


# ── Synchronous wrapper ─────────────────────────────────────────────────────

class _SyncWrapper:
    """Wraps the async GuardianEnv for use in blocking/synchronous code."""

    def __init__(self, async_client: "GuardianEnv"):
        self._client = async_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "_SyncWrapper":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._client._connect())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._loop:
            self._loop.run_until_complete(self._client._disconnect())
            self._loop.close()

    def reset(self, options: Optional[Dict] = None) -> StepResult:
        assert self._loop, "Call within 'with' block"
        return self._loop.run_until_complete(self._client.reset(options=options))

    def step(self, action: GuardianAction) -> StepResult:
        assert self._loop, "Call within 'with' block"
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> GuardianState:
        assert self._loop, "Call within 'with' block"
        return self._loop.run_until_complete(self._client.state())


# ── Main async client ───────────────────────────────────────────────────────

class GuardianEnv:
    """
    Async client for the GUARDIAN Fleet OpenEnv server.

    Implements the OpenEnv EnvClient interface:
      reset()       → StepResult
      step(action)  → StepResult
      state()       → GuardianState
      get_tools()   → List[str]  (with Rug-Pull deprovision filter applied)

    KEY CHANGES vs standard OpenEnv client:
      - Stateful episode_id tracking across all tool calls (Long-Horizon Temporal
        Correlation: the episode context is never dropped between steps).
      - get_tools() fetches the live tool manifest from the server and automatically
        strips quarantined tools — implements the MCP 'Rug Pull' deprovision pattern.
      - step() surfaces mcp_report and adaptation_report from the info dict so
        training loops can log the arms race curve without additional API calls.

    WebSocket transport is kept (not replaced with stdio/SSE) because:
      - OpenEnv compliance requires HTTP+WebSocket.
      - The MCP Gateway runs in-process server-side — no network transport change needed.
      - The HTTP fallback (http_reset/http_step) handles non-WS environments.
    """

    # OpenEnv metadata — used by openenv-core container providers
    action_type = GuardianAction
    observation_type = GuardianObservation
    docker_image = "guardian-env:latest"

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._ws: Any = None  # websockets.WebSocketClientProtocol
        # Stateful episode tracking — persisted across ALL step() calls
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._quarantined_tools: list = []  # Tools stripped from get_tools() output

    # ── Connection ──────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError:
            raise ImportError(
                "websockets is required for GuardianEnv: pip install websockets"
            )
        ws_url = (
            self.base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/ws"
        )
        self._ws = await websockets.connect(ws_url)

    async def _disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _send(self, msg: Dict) -> Dict:
        """Send a JSON message and receive the response."""
        if self._ws is None:
            await self._connect()
        await self._ws.send(json.dumps(msg))
        raw = await self._ws.recv()
        return json.loads(raw)

    # ── Async context manager ────────────────────────────────────────────────

    async def __aenter__(self) -> "GuardianEnv":
        await self._connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._disconnect()

    # ── Core API ─────────────────────────────────────────────────────────────

    async def reset(self, options: Optional[Dict] = None) -> StepResult:
        """Reset the environment and return the initial observation.

        Captures episode_id from the server state so every subsequent step()
        call is associated with this episode. This is the Long-Horizon stateful
        lock — episode context is never dropped across the 60-step trajectory.
        """
        resp = await self._send({"type": "reset", "options": options or {}})
        if resp.get("type") == "error":
            raise RuntimeError(f"Server error on reset: {resp.get('message')}")
        obs = GuardianObservation(**resp["observation"])
        state = GuardianState(**resp["state"]) if resp.get("state") else None
        # Persist episode_id for long-horizon temporal correlation
        if state:
            self._episode_id = state.episode_id
            self._step_count = 0
        self._quarantined_tools = []  # Clear deprovision list on reset
        return StepResult(observation=obs, reward=0.0, done=False, info={}, state=state)

    async def step(self, action: GuardianAction) -> StepResult:
        """Execute one Guardian decision and return the result.

        Maintains the stateful episode_id lock across the full 60-step trajectory.
        Surfaces mcp_report and adaptation_report from the info dict so training
        loops can log the MCP intercept audit trail and arms race curve.
        """
        if self._episode_id is None:
            raise RuntimeError("Call reset() before step()")
        resp = await self._send({"type": "step", "action": action.model_dump()})
        if resp.get("type") == "error":
            raise RuntimeError(f"Server error on step: {resp.get('message')}")
        obs = GuardianObservation(**resp["observation"])
        state = GuardianState(**resp["state"]) if resp.get("state") else None
        # Update stateful counters
        self._step_count += 1
        if state and state.episode_id != self._episode_id:
            # Server auto-reset after episode_done — lock to new episode
            self._episode_id = state.episode_id
            self._step_count = 0
            self._quarantined_tools = []
        # Extract MCP and adaptation reports from info (populated on termination)
        info = resp.get("info", {})
        if "mcp_report" in info:
            action._mcp_report = info["mcp_report"]           # type: ignore[attr-defined]
        if "adaptation_report" in info:
            action._adaptation_report = info["adaptation_report"]  # type: ignore[attr-defined]
        # If a quarantine_agent or quarantine_tool was applied, update deprovision list
        quarantined = info.get("quarantined_tools", [])
        if quarantined:
            self._quarantined_tools = list(set(self._quarantined_tools + quarantined))
        return StepResult(
            observation=obs,
            reward=float(resp.get("reward", 0.0)),
            done=bool(resp.get("done", False)),
            info=info,
            state=state,
        )

    async def state(self) -> GuardianState:
        """Return current episode metadata without advancing the environment."""
        resp = await self._send({"type": "state"})
        if resp.get("type") == "error":
            raise RuntimeError(f"Server error on state: {resp.get('message')}")
        return GuardianState(**resp["state"])

    async def get_tools(self) -> list:
        """
        Fetch the live MCP tool manifest from the server.

        Implements the MCP 'Rug Pull' deprovision pattern:
          - Queries the server for all registered tools and their risk levels.
          - Strips any tools in self._quarantined_tools before returning the list.
          - The Worker agent never sees quarantined tools in its available action space.

        Returns:
            List of available tool names (quarantined tools removed).
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{self.base_url}/tools")
            r.raise_for_status()
            all_tools: list = r.json().get("tools", [])
        # Apply deprovision filter (Rug Pull) — strip quarantined tools
        available = [
            t for t in all_tools
            if t not in self._quarantined_tools
        ]
        if len(available) < len(all_tools):
            stripped = [t for t in all_tools if t in self._quarantined_tools]
            # Log the deprovision event
            _ = stripped  # consumed by caller's monitoring loop
        return available

    def quarantine_tool(self, tool_name: str) -> None:
        """
        Client-side deprovision: add a tool to the quarantine list.
        Next call to get_tools() will not return this tool to the Worker.
        Implements the 'Rug Pull' surgical privilege downgrade.
        """
        if tool_name not in self._quarantined_tools:
            self._quarantined_tools.append(tool_name)

    @property
    def episode_id(self) -> Optional[str]:
        """Current episode ID — persisted across all step() calls."""
        return self._episode_id

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    # ── Sync wrapper ─────────────────────────────────────────────────────────

    def sync(self) -> _SyncWrapper:
        """
        Return a synchronous wrapper.

        Example:
            with GuardianEnv("http://localhost:8000").sync() as env:
                result = env.reset()
                result = env.step(GuardianAction(intervention="shadow"))
        """
        return _SyncWrapper(self)

    # ── HTTP fallback (no WebSocket) ─────────────────────────────────────────

    async def http_reset(self, options: Optional[Dict] = None) -> StepResult:
        """HTTP POST /reset — alternative when WebSocket not available."""
        try:
            import httpx  # type: ignore
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base_url}/reset", json=options or {})
            r.raise_for_status()
            data = r.json()
        obs = GuardianObservation(**data["observation"])
        state = GuardianState(**data["state"]) if data.get("state") else None
        return StepResult(observation=obs, reward=0.0, done=False, info={}, state=state)

    async def http_step(self, action: GuardianAction) -> StepResult:
        """HTTP POST /step — alternative when WebSocket not available."""
        try:
            import httpx  # type: ignore
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.base_url}/step",
                content=action.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
        obs = GuardianObservation(**data["observation"])
        state = GuardianState(**data["state"]) if data.get("state") else None
        return StepResult(
            observation=obs,
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
            state=state,
        )
