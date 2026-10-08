from __future__ import annotations

import json
import threading

import pytest

from test_model_agent_runtime import ScriptedModel
from servemind.agents.domain_router import CommerceDomainRouter
from servemind.agents.model_runtime import RequestLedger
from servemind.agents.task_graph import TaskGraph, TaskNode
from servemind.core.intent_recognizer import Intent, IntentResult
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.llm.deepseek_client import ChatTurn, Completion
from servemind.mcp.tool_manager import ToolResult
from servemind.service.commerce_support import CommerceSupport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SERVEMIND_RAG_BACKEND", "keyword")
    monkeypatch.setenv("SERVEMIND_SEMANTIC_INTENT", "false")
    monkeypatch.setenv("SERVEMIND_AGENT_MODE", "model")


def test_main_support_order_is_supplied_to_composer_and_not_completion_order():
    model = ScriptedModel()
    _, meta = CommerceSupport(llm_client=model).reply("多少钱，还有货吗", PRODUCT)
    assert meta["routing"]["primary"] == "billing"
    assert meta["routing"]["supporting"] == ["catalog"]
    assert meta["routing"]["reason"] and meta["routing"]["scores"]["billing"] > 0
    assert [c["collaboration_position"] for c in meta["agent_contributions"]] == ["primary", "supporting"]
    composer = next(json.loads(m[1]["content"]) for m in model.requests if "Response Composer" in m[0]["content"])
    assert composer["routing"] == meta["routing"]
    assert [c["agent"] for c in composer["agent_contributions"]] == ["billing", "catalog"]
    assert [s["intent"] for s in composer["sections"]] == ["price_breakdown", "inventory_query"]


def test_explicit_handoff_is_main_but_preserves_requested_fact_domains():
    model = ScriptedModel(["price_breakdown", "inventory_query", "human_handoff"])
    _, meta = CommerceSupport(llm_client=model).reply("把价格和库存一起交给店家", PRODUCT)
    assert meta["routing"]["primary"] == "escalation"
    assert set(meta["routing"]["supporting"]) == {"billing", "catalog"}
    assert meta["needs_merchant"]
    assert set(meta["topics"]) == {"price_breakdown", "inventory_query", "human_handoff"}


def test_negated_keywords_cannot_add_a_domain_excluded_by_recognition():
    model = ScriptedModel(["price_breakdown"])
    _, meta = CommerceSupport(llm_client=model).reply("不要查库存，也不要转人工，只问多少钱", PRODUCT)
    assert meta["agent_roles"] == ["billing"]
    assert not meta["routing"]["multi_agent"]
    assert not meta["needs_merchant"]


def test_same_role_topics_are_one_dispatch_and_keep_separate_verification():
    model = ScriptedModel(["inventory_query", "sku_query"])
    _, meta = CommerceSupport(llm_client=model).reply("有货吗，品牌规格是什么", PRODUCT)
    assert meta["task_graph"]["role_count"] == 1
    assert meta["task_graph"]["total"] == meta["task_graph"]["completed"] == 2
    assert not meta["routing"]["multi_agent"]
    packets = [json.loads(m[1]["content"]) for m in model.requests if "专业客服 Agent" in m[0]["content"]]
    assert all(p["assigned_topics"] == ["inventory_query", "sku_query"] for p in packets)
    assert {p["topic"] for p in packets} == {"inventory_query", "sku_query"}
    assert all(not p["dependency_evidence"] for p in packets)


def test_distinct_roles_overlap_and_role_exception_cannot_block_other_role(monkeypatch):
    support = CommerceSupport(llm_client=ScriptedModel())
    barrier = threading.Barrier(2)
    called = []
    lock = threading.Lock()

    def run_node(node, message, context, ledger):
        with lock:
            called.append((node.role, threading.get_ident()))
        barrier.wait(timeout=3)  # Sequential role dispatch would fail this check.
        if node.role == "billing":
            raise RuntimeError("isolated billing failure")
        assert context["collaboration_position"] == "supporting"
        return ToolResult(True, [{"evidence_id": "catalog-evidence"}])

    monkeypatch.setattr(support.model_runtime, "_run_node", run_node)
    router = CommerceDomainRouter()
    message = "价格和库存"
    recognized = IntentResult(Intent.PRICE_BREAKDOWN, [Intent.PRICE_BREAKDOWN, Intent.INVENTORY_QUERY], .9, {})
    decision = router.route(message, recognized, ("price_breakdown", "inventory_query"),
                            available_roles={role.value for role in support.agents})
    graph = router.tasks(decision, ("price_breakdown", "inventory_query"), product_id=PRODUCT["id"], message=message)
    results, status = support.model_runtime.run(graph, message, {"routing": decision.as_dict()}, RequestLedger())
    assert len({tid for _, tid in called}) == 2
    assert status["role_count"] == 2 and status["roles_completed"] == 1
    by_role = {node.role: result for node, result in results}
    assert by_role["billing"].error == "agent_execution_failed"
    assert by_role["catalog"].success and by_role["catalog"].data[0]["evidence_id"] == "catalog-evidence"


def test_low_confidence_other_clarifies_before_specialist_or_composer():
    class Unsure(ScriptedModel):
        def chat(self, *, messages, **kwargs):
            assert "电商客服意图识别器" in messages[0]["content"]
            text = json.dumps({"primary": "other", "confidence": .1, "intents": ["other"]})
            return ChatTurn({"role": "assistant", "content": text}, Completion(text, "fixture", 20, 10))

    answer, meta = CommerceSupport(llm_client=Unsure()).reply("嗯那个事情怎么说", PRODUCT)
    assert meta["routing"]["clarification"]
    assert meta["status"] == "clarifying" and not meta["needs_merchant"]
    assert meta["model_use"]["calls"] == 1
    assert meta["tools_used"] == [] and meta["agent_contributions"] == []
    assert "哪一项" in answer


def test_relative_score_threshold_cannot_erase_an_explicit_domain():
    intent = IntentResult(Intent.PRICE_BREAKDOWN, [Intent.PRICE_BREAKDOWN, Intent.ORDER_QUERY], .9, {})
    decision = CommerceDomainRouter().route("价格多少钱优惠折扣金额实付", intent,
        ("price_breakdown", "order_query"), available_roles={"general", "billing", "order"})
    assert decision.primary == "billing" and decision.supporting == ("order",)
    assert "retained_requested_domains=order" in decision.reason


def test_commercial_runner_rejects_dependency_edges():
    support = CommerceSupport(llm_client=ScriptedModel())
    graph = TaskGraph(("sku_query", "inventory_query"), (
        TaskNode("a", "sku_query", "catalog", "", {}),
        TaskNode("b", "inventory_query", "catalog", "", {}, ("a",))))
    with pytest.raises(ValueError, match="commercial_dependencies_not_supported"):
        support.model_runtime.run(graph, "库存", {}, RequestLedger())


def test_recognition_failure_is_counted_under_new_contract():
    support = CommerceSupport(llm_client=ScriptedModel(attack="cycle"))
    _, meta = support.reply("多少钱有货吗", PRODUCT)
    assert meta["intent_three_way"]["mode"] == "intent_recognition_failed_rule_fallback"
    stats = support.monitor.summary()
    assert stats["model_agent_requests"] == 1
    assert stats["intent_recognition_fallback_count"] == 1
    assert all(not n["depends_on"] for n in meta["task_graph"]["nodes"])
