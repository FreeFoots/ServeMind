from __future__ import annotations

from datetime import date, timedelta

from servemind.core.knowledge_base import KnowledgeBase
from servemind.core.skill_loader import Skill, SkillManager
from servemind.mcp.tool_manager import Tool, ToolManager
from servemind.service.commerce_support import CommerceSupport


def test_stock_and_price_multi_intent_uses_labeled_product_evidence():
    product = {
        "id": "prod_demo", "catalog_status": "暂时缺货", "display_price": "67.90",
        "price_basis": "historical_reference_simulated", "price_as_of": "2018-03-31",
        "provenance": "merchant_declared_unverified",
    }
    answer, metadata = CommerceSupport(use_llm=False).reply("商品还有货吗，价格是多少", product)
    assert "暂时缺货" in answer
    assert "¥67.90" in answer
    assert "商家确认现在的售价" in answer
    assert metadata["needs_merchant"] is False
    assert len(metadata["evidence_ids"]) >= 2
    assert metadata["grounding"]["passed"] is True
    assert all(section["evidence_ids"] for section in metadata["answer_sections"])
    assert metadata["grounded"] is True


def test_price_without_evidence_escalates_to_merchant():
    product = {
        "id": "prod_demo", "catalog_status": "在售", "display_price": None,
        "price_basis": "not_provided", "provenance": "merchant_declared_unverified",
    }
    answer, metadata = CommerceSupport(use_llm=False).reply("现在多少钱", product)
    assert "向商家确认价格" in answer
    assert metadata["needs_merchant"] is True


def test_tool_schema_and_permission_are_enforced_before_handler():
    calls = []
    manager = ToolManager()
    manager.register(Tool(
        "read_public", "read", lambda params, context: calls.append(params),
        required_permissions=("policy_public",),
    ))
    denied = manager.call("read_public", {}, {"allowed_tools": {"read_public"}})
    assert denied.error == "forbidden:tool_scope"
    allowed = manager.call("read_public", {}, {
        "allowed_tools": {"read_public"}, "permissions": {"policy_public"},
    })
    assert allowed.success is True
    assert calls == [{}]


def test_equal_priority_skill_conflict_fails_closed():
    manager = SkillManager()
    manager.skills = [
        Skill("one", "buyer", ("退款",), "A", ("refund_policy",), 10,
              ("policy_public",), conflict_group="refund"),
        Skill("two", "buyer", ("退款",), "B", ("refund_policy",), 10,
              ("policy_public",), conflict_group="refund"),
        Skill("expired", "buyer", ("退款",), "C", ("refund_policy",), 20,
              ("policy_public",), valid_until=date.today() - timedelta(days=1)),
    ]
    selected = manager.select("我要退款", "refund_policy", "buyer")
    assert selected.skills == ()
    assert selected.conflicts == ("refund",)
    assert selected.expired == ("expired",)


def test_keyword_fallback_exposes_version_and_validity():
    kb = KnowledgeBase()
    kb.vector = None
    results = kb.search("退款")
    assert results
    assert all(item["document_version"] and item["valid_from"] for item in results)
