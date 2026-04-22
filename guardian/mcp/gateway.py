"""
GUARDIAN MCP Gateway — The Transport-Layer Security Interceptor
===============================================================
Implements the MCP Centralized Security Gateway pattern.

Every tool call the Worker makes is packaged as a JSON-RPC 2.0 MCP Request
and routed through this gateway. Guardian inspects the request and decides:

  FORWARD   → pass request to the real tool (allow)
  REWRITE   → modify the request parameters before forwarding
  HONEYPOT  → silently route to mcp://honeypot-db instead of production
  BLOCK     → return a synthetic MCP Error without touching the tool
  ESCALATE  → forward + trigger async IR flow on mcp://security-ops

This means GUARDIAN is NOT a Python-level monitor bolted onto a script.
It is a standards-compliant MCP proxy that can guard Claude, Copilot,
or any other JSON-RPC 2.0 client — the attack type is irrelevant to the
transport layer.

JSON-RPC 2.0 message format (MCP):
  Request:  {"jsonrpc": "2.0", "id": <int>, "method": "tools/call",
             "params": {"name": <tool>, "arguments": <dict>}}
  Response: {"jsonrpc": "2.0", "id": <int>, "result": <dict>}
  Error:    {"jsonrpc": "2.0", "id": <int>, "error": {"code": <int>, "message": <str>}}
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── MCP JSON-RPC 2.0 Message Primitives ─────────────────────────────────────

@dataclass
class MCPRequest:
    """Represents a single MCP JSON-RPC 2.0 tool invocation from the Worker."""
    tool_name: str
    arguments: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_agent: str = "worker"
    timestamp: float = field(default_factory=time.time)
    episode_step: int = 0

    def to_jsonrpc(self) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": "tools/call",
            "params": {
                "name": self.tool_name,
                "arguments": self.arguments,
                "_meta": {
                    "source_agent": self.source_agent,
                    "episode_step": self.episode_step,
                    "ts": self.timestamp,
                }
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.to_jsonrpc(), indent=2)


@dataclass
class MCPResponse:
    """Represents the MCP JSON-RPC 2.0 response returned to the Worker."""
    id: str
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    intercepted: bool = False       # True if Gateway modified or rerouted the call
    routed_to: str = "production"   # 'production' | 'honeypot' | 'blocked'
    guardian_action: str = "allow"  # allow | rewrite | honeypot | block | escalate
    ir_triggered: bool = False      # True if Incident Response flow was activated

    def to_jsonrpc(self) -> Dict:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            msg["error"] = self.error
        else:
            msg["result"] = self.result or {}
        return msg

    def to_json(self) -> str:
        return json.dumps(self.to_jsonrpc(), indent=2)


# ── MCP Error Codes (subset of JSON-RPC spec + MCP extensions) ──────────────

class MCPErrorCode:
    PARSE_ERROR      = -32700
    INVALID_REQUEST  = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS   = -32602
    INTERNAL_ERROR   = -32603
    # MCP Security Extensions
    TOOL_BLOCKED_BY_GUARDIAN   = -32001
    SESSION_REVOKED            = -32002
    PRIVILEGE_INSUFFICIENT     = -32003
    APPROVAL_CHAIN_INVALID     = -32004
    CANARY_TRIGGERED           = -32005


# ── Attack → MCP Server routing table ────────────────────────────────────────
# Maps every Guardian-classified attack type to which MCP servers handle it.
# This is the generalization engine — not hardcoded per-attack, but classification-driven.

ATTACK_MCP_ROUTING: Dict[str, Dict] = {
    "authority_spoofing": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "revoke_session",
        "audit_level": "CRITICAL",
        "description": "Executive override fabrication → session revoke + role downgrade",
    },
    "prompt_injection": {
        "primary": "honeypot-db",
        "secondary": "audit-log",
        "ir_action": "honeypot_proxy",
        "audit_level": "HIGH",
        "description": "Hidden instruction injection → transparent honeypot routing",
    },
    "approval_bypass": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "flag_bypass",
        "audit_level": "HIGH",
        "description": "Skipped approve gate → audit flag + security ops alert",
    },
    "data_exfiltration": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "block_outbound_route",
        "audit_level": "CRITICAL",
        "description": "Data export attempt → block outbound + RCA to #security-ops",
    },
    "confused_deputy": {
        "primary": "iam-control",
        "secondary": "audit-log",
        "ir_action": "privilege_separation",
        "audit_level": "HIGH",
        "description": "Self-authorized action → enforce privilege separation boundary",
    },
    "approval_laundering": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "verify_approval_chain",
        "audit_level": "CRITICAL",
        "description": "Fabricated CISO approval → hash-verify audit chain",
    },
    "salami_slicing": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "aggregate_payment_analysis",
        "audit_level": "MEDIUM",
        "description": "Sub-threshold payments → cross-step aggregation analysis",
    },
    "schema_drift_exploit": {
        "primary": "audit-log",
        "secondary": "iam-control",
        "ir_action": "verify_schema_version",
        "audit_level": "HIGH",
        "description": "Legacy field name exploit → schema version mismatch detection",
    },
    "rogue_internal_ai": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "quarantine_agent_and_rollback_iam",
        "audit_level": "CRITICAL",   # Sev-1
        "description": "Rogue AI auto-post → quarantine agent, revert IAM, post incident RCA",
    },
    # Default for unknown/clean
    "clean": {
        "primary": "audit-log",
        "secondary": None,
        "ir_action": "log_only",
        "audit_level": "INFO",
        "description": "Clean episode — audit log only",
    },
}


# ── The MCP Gateway ───────────────────────────────────────────────────────────

class MCPGateway:
    """
    The GUARDIAN MCP Centralized Security Gateway.

    Mediates ALL tool calls from Worker agent to environment backends.
    Designed to be drop-in compatible with any MCP JSON-RPC 2.0 transport
    (stdio, HTTP+SSE, or WebSocket as per the MCP spec).

    In this simulation, the gateway operates in-process using Python objects
    instead of network sockets — the JSON-RPC message structure is identical
    to what a real MCP client/server pair would exchange.

    Usage:
        gateway = MCPGateway()
        request = MCPRequest(tool_name="write_db", arguments={...}, episode_step=5)
        response = gateway.dispatch(request, classified_attack="prompt_injection")
    """

    def __init__(self):
        from guardian.mcp.mock_servers import (
            MockIAMServer, MockAuditServer,
            MockHoneypotServer, MockSecurityOpsServer,
        )
        self._servers = {
            "iam-control":  MockIAMServer(),
            "audit-log":    MockAuditServer(),
            "honeypot-db":  MockHoneypotServer(),
            "security-ops": MockSecurityOpsServer(),
        }
        self._intercept_log: List[Dict] = []   # Full MCP intercept audit trail
        self._ir_log: List[Dict] = []          # Incident Response activation log

    # ── Public API ─────────────────────────────────────────────────────────

    def dispatch(
        self,
        request: MCPRequest,
        classified_attack: Optional[str] = None,
        guardian_intervention: str = "allow",
        risk_score: float = 0.0,
    ) -> MCPResponse:
        """
        Core dispatch method. Called for every Worker tool invocation.

        Args:
            request: The MCP tool call request from the Worker.
            classified_attack: Guardian's current attack classification (or None).
            guardian_intervention: The Guardian's chosen intervention string.
            risk_score: Guardian's risk score for this step (0.0 - 1.0).

        Returns:
            MCPResponse with full audit metadata.
        """
        routing = ATTACK_MCP_ROUTING.get(classified_attack or "clean")
        response = self._route(request, routing, guardian_intervention, risk_score)

        # Record every intercept in the gateway audit trail
        self._intercept_log.append({
            "mcp_request_id": request.id,
            "tool": request.tool_name,
            "attack": classified_attack,
            "guardian_action": response.guardian_action,
            "routed_to": response.routed_to,
            "ir_triggered": response.ir_triggered,
            "step": request.episode_step,
            "ts": request.timestamp,
        })
        return response

    def get_intercept_log(self) -> List[Dict]:
        """Returns the full MCP intercept audit trail for this episode."""
        return list(self._intercept_log)

    def get_ir_log(self) -> List[Dict]:
        """Returns all Incident Response activations for this episode."""
        return list(self._ir_log)

    def get_security_ops_posts(self) -> List[Dict]:
        """Returns all posts made to the mock #security-ops Slack channel."""
        return self._servers["security-ops"].get_posts()

    def get_audit_entries(self) -> List[Dict]:
        """Returns all entries in the mock audit log."""
        return self._servers["audit-log"].get_entries()

    def reset(self) -> None:
        """Reset all server state between episodes."""
        self._intercept_log = []
        self._ir_log = []
        for server in self._servers.values():
            server.reset()

    # ── Internal routing logic ─────────────────────────────────────────────

    def _route(
        self,
        request: MCPRequest,
        routing: Dict,
        intervention: str,
        risk_score: float,
    ) -> MCPResponse:
        """
        Implements the four-way routing decision:
          HONEYPOT  → when intervention is 'rewrite' or attack is prompt_injection
          BLOCK     → when intervention is 'emergency_fork' or 'require_justification'
          ESCALATE  → when risk_score > 0.7, forward + trigger IR
          FORWARD   → allow, shadow, interrogate (observer mode)
        """
        action = self._determine_action(intervention, routing, risk_score)

        if action == "honeypot":
            return self._handle_honeypot(request, routing)
        elif action == "block":
            return self._handle_block(request, routing)
        elif action == "escalate":
            return self._handle_escalate(request, routing)
        else:
            return self._handle_forward(request, routing, action)

    def _determine_action(
        self, intervention: str, routing: Dict, risk_score: float
    ) -> str:
        """Map Guardian intervention → MCP gateway action."""
        if intervention in ("rewrite",) or routing.get("ir_action") == "honeypot_proxy":
            return "honeypot"
        if intervention in ("emergency_fork", "require_justification", "block",
                            "quarantine_tool", "quarantine_agent", "reduce_privs"):
            return "block"
        if risk_score > 0.7 or routing.get("audit_level") == "CRITICAL":
            return "escalate"
        return "allow"

    def _handle_honeypot(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Route the Worker's tool call to the honeypot server instead of production.
        The Worker receives a perfectly-formatted success response — it never knows."""
        honeypot = self._servers["honeypot-db"]
        result = honeypot.handle(request)
        # Also write to audit log
        self._servers["audit-log"].record_intercept(request, routing, "honeypot")
        return MCPResponse(
            id=request.id,
            result=result,
            intercepted=True,
            routed_to="honeypot",
            guardian_action="honeypot",
        )

    def _handle_block(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Return a synthetic MCP Error. The tool is never called."""
        primary_server = self._servers.get(routing.get("primary", "audit-log"))
        if primary_server:
            primary_server.handle_block(request, routing)
        self._servers["audit-log"].record_intercept(request, routing, "blocked")
        return MCPResponse(
            id=request.id,
            error={
                "code": MCPErrorCode.TOOL_BLOCKED_BY_GUARDIAN,
                "message": (
                    f"GUARDIAN blocked tool '{request.tool_name}' — "
                    f"classified as {routing.get('description', 'security violation')}. "
                    f"Action: {routing.get('ir_action', 'block')}."
                ),
                "data": {
                    "blocked_tool": request.tool_name,
                    "guardian_routing": routing,
                    "mcp_server_chain": [routing.get("primary"), routing.get("secondary")],
                },
            },
            intercepted=True,
            routed_to="blocked",
            guardian_action="block",
        )

    def _handle_escalate(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Forward the request (don't block), but simultaneously trigger IR flow
        on the relevant MCP servers. This is the 'Autonomous IR Swarm' pattern."""
        # Fire IR on primary server
        primary = self._servers.get(routing.get("primary", "audit-log"))
        ir_result = {}
        if primary:
            ir_result = primary.handle_ir(request, routing)

        # Fire secondary (e.g., security-ops Slack post)
        secondary_name = routing.get("secondary")
        if secondary_name and secondary_name in self._servers:
            self._servers[secondary_name].handle_ir(request, routing)

        self._ir_log.append({
            "mcp_request_id": request.id,
            "tool": request.tool_name,
            "ir_action": routing.get("ir_action"),
            "primary_server": routing.get("primary"),
            "secondary_server": routing.get("secondary"),
            "ir_result": ir_result,
            "step": request.episode_step,
        })
        self._servers["audit-log"].record_intercept(request, routing, "escalated")

        return MCPResponse(
            id=request.id,
            result={"status": "forwarded", "ir_activated": True, "ir_action": routing.get("ir_action")},
            intercepted=True,
            routed_to="production+ir",
            guardian_action="escalate",
            ir_triggered=True,
        )

    def _handle_forward(
        self, request: MCPRequest, routing: Dict, action: str
    ) -> MCPResponse:
        """Pass the request through unchanged. Log to audit server."""
        self._servers["audit-log"].record_intercept(request, routing, "forwarded")
        return MCPResponse(
            id=request.id,
            result={"status": "forwarded", "tool": request.tool_name},
            intercepted=False,
            routed_to="production",
            guardian_action=action,
        )
