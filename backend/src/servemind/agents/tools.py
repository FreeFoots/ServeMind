from __future__ import annotations

from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field

from servemind.mcp.tool_manager import Tool, ToolManager


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class OrderInput(ToolInput):
    order_id: str = Field(min_length=1, max_length=80)


class SKUInput(ToolInput):
    sku_id: str = Field(min_length=1, max_length=80)


class InventoryInput(SKUInput):
    dc_id: str | None = Field(default=None, max_length=80)
    date: str | None = Field(default=None, max_length=32)


class BrowsingInput(ToolInput):
    limit: int = Field(default=20, ge=1, le=20)


class KnowledgeInput(ToolInput):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


def build_tool_manager(provider: Any, knowledge: Any) -> ToolManager:
    manager = ToolManager()
    historical = {"required_permissions": ("historical_read",),
                  "timeout_seconds": 8.0, "max_retries": 1,
                  "retryable_exceptions": (psycopg.OperationalError,)}
    manager.register(Tool(
        "get_order_facts", "查询订单事实并执行用户归属校验",
        lambda p, c: provider.order_facts(p["order_id"], c.get("user_id")),
        ("order_id",), input_model=OrderInput, **historical,
    ))
    manager.register(Tool(
        "get_delivery_timeline", "查询订单配送和包裹时间线",
        lambda p, c: [item for item in provider.order_facts(p["order_id"], c.get("user_id"))
                      if item.kind == "delivery"],
        ("order_id",), input_model=OrderInput, **historical,
    ))
    manager.register(Tool(
        "get_price_breakdown", "查询订单价格、优惠和实付字段",
        lambda p, c: provider.price_breakdown(p["order_id"], c.get("user_id")),
        ("order_id",), input_model=OrderInput, **historical,
    ))
    manager.register(Tool(
        "get_sku", "查询匿名 SKU 属性和履约类型",
        lambda p, c: provider.sku_facts(p["sku_id"]),
        ("sku_id",), input_model=SKUInput, **historical,
    ))
    manager.register(Tool(
        "get_inventory", "查询指定日期仓库是否存在库存记录",
        lambda p, c: provider.inventory_facts(p["sku_id"], p.get("dc_id"), p.get("date")),
        ("sku_id",), input_model=InventoryInput, **historical,
    ))
    manager.register(Tool(
        "get_browsing_history", "查询受限时间窗内的用户浏览记录",
        lambda p, c: provider.browsing_facts(c.get("user_id", ""), p.get("limit", 20)),
        input_model=BrowsingInput, **historical,
    ))
    manager.register(Tool(
        "search_knowledge", "检索电商政策和帮助知识库",
        lambda p, c: knowledge.search(p["query"], p.get("top_k", 3)),
        ("query",), input_model=KnowledgeInput,
        required_permissions=("policy_public",), timeout_seconds=60.0,
    ))
    return manager
