"""MCP SDK client/server exchange without a second model process.

The in-process transport still performs initialize and tools/call protocol
requests. Only public policy is exported. Business snapshots never cross MCP.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import anyio
from mcp import ClientSession
from mcp.server import MCPServer
from mcp.shared.memory import create_client_server_memory_streams

from servemind.llm.deepseek_client import redact_identifiers


class KnowledgeMCPClient:
    def __init__(self, knowledge: Any) -> None:
        self.knowledge = knowledge

    def search(self, query: str, intent: str, top_k: int = 3, queries: list[str] | None = None) -> dict[str, Any]:
        return asyncio.run(self._search(redact_identifiers(query), intent, top_k,
                                       [redact_identifiers(q,max_chars=200) for q in (queries or [])[:2]]))

    async def _search(self, query: str, intent: str, top_k: int, queries: list[str]) -> dict[str, Any]:
        server = MCPServer("ServeMind Public Policy")

        @server.tool()
        def search_policy(query: str, intent: str, top_k: int = 3, queries: list[str] | None = None) -> dict:
            return self.knowledge.search_for_intent(query, intent, top_k=max(1, min(top_k, 5)), queries=queries)

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as group:
                # MCPServer's SDK transport methods use this same low-level runner.
                lowlevel = server._lowlevel_server
                group.start_soon(lowlevel.run, *server_streams, lowlevel.create_initialization_options())
                async with ClientSession(*client_streams) as session:
                    await session.initialize()
                    response = await session.call_tool("search_policy", {
                        "query": query, "intent": intent, "top_k": top_k, "queries":queries})
                    if response.is_error:
                        raise RuntimeError("mcp_policy_failed")
                    result = json.loads(next(block.text for block in response.content if block.type == "text"))
                group.cancel_scope.cancel()
        return {**result, "transport": "mcp_sdk_memory", "tool": "search_policy"}
