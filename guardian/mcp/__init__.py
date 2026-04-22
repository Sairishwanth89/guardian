"""
GUARDIAN MCP Layer
===================
Model Context Protocol (JSON-RPC 2.0) security gateway for the Guardian Fleet.

Architecture:
  - MCPGateway     : Sits between Worker and all environment tools. Every tool call
                     becomes a JSON-RPC 2.0 MCP Request that Guardian can inspect,
                     forward, rewrite, or silently redirect to a honeypot server.
  - MockMCPServers : Four lightweight in-process mock MCP servers:
                       - mcp://iam-control      (session revocation, role downgrade)
                       - mcp://audit-log        (approval chain verification, hash audit)
                       - mcp://honeypot-db      (transparent honeypot proxy)
                       - mcp://security-ops     (Slack RCA post, Jira ticket creation)

This implements the "MCP Centralized Security Gateway" pattern, making GUARDIAN
compatible with ANY MCP-compliant worker agent (Claude, Copilot, custom) connecting
to ANY MCP-compliant enterprise tool server.

References:
  - MCP Spec: https://modelcontextprotocol.io/specification
  - JSON-RPC 2.0: https://www.jsonrpc.org/specification
"""
from guardian.mcp.gateway import MCPGateway, MCPRequest, MCPResponse
from guardian.mcp.mock_servers import MockIAMServer, MockAuditServer, MockHoneypotServer, MockSecurityOpsServer

__all__ = [
    "MCPGateway",
    "MCPRequest",
    "MCPResponse",
    "MockIAMServer",
    "MockAuditServer",
    "MockHoneypotServer",
    "MockSecurityOpsServer",
]
