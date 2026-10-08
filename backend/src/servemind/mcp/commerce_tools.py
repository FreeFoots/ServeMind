"""Commerce tools consume only server-authorized current conversation snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from servemind.agents.task_graph import TaskGraph, TaskNode, ROLE_FOR_INTENT
from servemind.agents.commerce_agents import build_agent_pool
from servemind.core.intent_recognizer import Intent
from servemind.mcp.knowledge_client import KnowledgeMCPClient
from servemind.mcp.tool_manager import Tool, ToolManager


ROLE_SCOPES = {role.value: agent.profile.commerce_tool_scope for role, agent in build_agent_pool().items()}


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=128)


class PurchaseInput(ProductInput):
    pass


class PolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    intent: str = Field(min_length=1, max_length=40)
    queries: list[str] = Field(default_factory=list, max_length=2)


def _check_scope(params: dict, context: dict) -> dict:
    product = context["product"]
    if params["product_id"] != product.get("id", "current_product"):
        raise PermissionError("product_scope_mismatch")
    if (context.get("merchant_id") and product.get("merchant_id")
            and context["merchant_id"] != product["merchant_id"]):
        raise PermissionError("merchant_scope_mismatch")
    return product


def _evidence(product_id: str, field: str, value: Any, kind: str, source: str) -> dict:
    return {"evidence_id": f"{source}:{product_id}:{field}", "kind": kind,
            "source": source, "field": field, "value": value,
            "facts": [{"field": field, "value": value}], "scope": "current_conversation",
            "retrieved_at": datetime.now(timezone.utc).isoformat()}


def _product(params: dict, context: dict) -> list[dict]:
    product = _check_scope(params, context)
    fields = ("title", "description", "category", "catalog_status", "display_price", "price_basis")
    return [_evidence(params["product_id"], field, product[field], "product", "commerce.products")
            for field in fields if product.get(field) is not None]


def _purchase(params: dict, context: dict) -> list[dict]:
    _check_scope(params, context)
    purchase = context.get("purchase")
    if not purchase:
        return []
    for key in ("buyer_id", "merchant_id"):
        if context.get(key) and purchase.get(key) and context[key] != purchase[key]:
            raise PermissionError("purchase_scope_mismatch")
    if purchase.get("sku_id") and context["product"].get("sku_id") and purchase["sku_id"] != context["product"]["sku_id"]:
        raise PermissionError("purchase_product_mismatch")
    return [_evidence(params["product_id"], field, purchase[field], "purchase", "commerce.purchases")
            for field in ("order_alias", "fulfillment_status", "paid_amount", "discount_amount", "estimated_delivery")
            if purchase.get(field) is not None]


def build_commerce_tools(knowledge: Any) -> ToolManager:
    manager = ToolManager()
    mcp = KnowledgeMCPClient(knowledge)
    manager.register(Tool("get_current_product", "读取当前会话授权商品", _product,
                          input_model=ProductInput, required_permissions=("catalog_read",)))
    manager.register(Tool("get_current_purchase", "读取当前买家已绑定订单", _purchase,
                          input_model=PurchaseInput, required_permissions=("purchase_read",)))

    def policy(params: dict, context: dict) -> list[dict]:
        result = mcp.search(**params)
        return [{**item, "kind": "knowledge", "transport": result["transport"],
                 "retrieval_backend": result.get("backend"),
                 "cache_hit": result.get("cache_hit", False),
                 "evidence_id": f"knowledge:{item.get('chunk_id') or item['title']}",
                 "field": "policy", "value": item["content"],
                 "facts": [{"field": "policy", "value": item["content"]}],
                 "scope": "public_policy"} for item in result["items"]]

    manager.register(Tool("search_knowledge", "经 MCP 检索有效公共政策", policy,
                          input_model=PolicyInput, required_permissions=("policy_public",),
                          timeout_seconds=30))
    return manager


def commerce_graph(intents: tuple[str, ...], product_id: str, message: str) -> TaskGraph:
    nodes = []
    for topic in intents:
        role = ROLE_FOR_INTENT.get(Intent(topic), "general")
        if topic in {"inventory_query", "sku_query"} or (topic == "price_breakdown" and not any(
                term in message for term in ("优惠", "实付", "成交", "订单价格"))):
            tool, params = "get_current_product", {"product_id": product_id}
        elif topic in {"order_query", "delivery_status", "delivery_exception", "price_breakdown"}:
            tool, params = "get_current_purchase", {"product_id": product_id}
        elif topic in {"refund_policy", "cancel_policy", "complaint", "human_handoff", "invoice_query"}:
            tool, params = "search_knowledge", {"query": message, "intent": topic}
        else:
            continue
        nodes.append(TaskNode(f"task-{len(nodes) + 1}", topic, role, tool, params))
    graph = TaskGraph(intents, tuple(nodes), deadline_seconds=35)
    graph.validate()
    return graph
