"""Read-only MCP boundary for public policy knowledge.

Historical order and buyer data are intentionally excluded until MCP requests
carry a verified participant identity and tool-scope authorization context.
"""
from __future__ import annotations

from mcp.server import MCPServer

from servemind.core.knowledge_base import KnowledgeBase
from servemind.llm.deepseek_client import redact_identifiers

mcp = MCPServer("ServeMind Knowledge")
knowledge = KnowledgeBase()


@mcp.tool()
def search_policy(query: str, intent: str, top_k: int = 3) -> dict:
    """Retrieve public, non-transactional customer-support policy passages."""
    safe_query = redact_identifiers(query)
    result = knowledge.search_for_intent(safe_query, intent, top_k=max(1, min(top_k, 5)))
    return {
        "enabled": result["enabled"],
        "backend": result.get("backend", "disabled"),
        "items": [
            {"title": item["title"], "source": item.get("source"),
             "content": item["content"][:700], "score": item["score"],
             "chunk_id": item.get("chunk_id"),
             "document_version": item.get("document_version"),
             "valid_from": item.get("valid_from"),
             "valid_until": item.get("valid_until")}
            for item in result["items"]
        ],
    }


@mcp.resource("servemind://knowledge/catalog")
def knowledge_catalog() -> dict:
    """Return the public knowledge catalog and configured retrieval backend."""
    return knowledge.summary()


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
