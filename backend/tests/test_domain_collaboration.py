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


def test_explicit_handoff_workflow_preserves_requested_fact_domains():
    model = ScriptedModel(["price_breakdown", "inventory_query", "human_handoff"])
    _, meta = CommerceSupport(llm_client=model).reply("把价格和库存一起交给店家", PRODUCT)
    assert set(meta["agent_roles"]) == {"billing", "catalog", "escalation"}
    assert meta["routing"]["handoff_requested"]
    assert meta["handoff_workflow"]["mode"] == "agent_led_guarded_handoff"
    assert {c["intent"] for c in meta["agent_contributions"] if c.get("phase") != "handoff_planning"} == {"price_breakdown", "inventory_query", "human_handoff"}
    assert any(t["role"] == "escalation" and t["name"] == "search_knowledge" for t in meta["tools_used"])
    assert meta["task_graph"]["role_count"] == 3
    assert meta["needs_merchant"]
    assert set(meta["topics"]) == {"price_breakdown", "inventory_query", "human_handoff"}


def test_handoff_primary_with_greeting_cannot_displace_fact_role():
    model = ScriptedModel(["human_handoff", "greeting", "price_breakdown"])
    _, meta = CommerceSupport(llm_client=model).reply("您好，请转人工并查价格", PRODUCT)
    assert meta["routing"]["primary"] == "billing"
    assert meta["agent_roles"] == ["billing", "escalation"]
    assert meta["needs_merchant"] and meta["handoff_workflow"]["policy_completed"] == 1
    assert {c["intent"] for c in meta["agent_contributions"] if c.get("phase") != "handoff_planning"} == {"price_breakdown", "human_handoff"}
    assert set(meta["topics"]) == {"human_handoff", "greeting", "price_breakdown"}


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
        ("price_breakdown", "order_query"), available_roles={"general", "billing", "order_fulfillment"})
    assert decision.primary == "billing" and decision.supporting == ("order_fulfillment",)
    assert "retained_requested_domains=order_fulfillment" in decision.reason


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


@pytest.mark.parametrize("message,topic,status", [
    ("你好", "greeting", "answered"),
    ("客服你好", "greeting", "answered"),
    ("再见", "greeting", "answered"),
    ("谢谢客服，服务很好", "feedback", "answered"),
    ("我想咨询一下", "query", "clarifying"),
    ("帮我处理一下", "request", "clarifying"),
])
def test_general_interactions_preserve_labels_without_tools_or_merchant(message, topic, status):
    answer, meta = CommerceSupport(use_llm=False).reply(message, PRODUCT)
    assert meta["intent"] == topic and meta["topics"] == [topic]
    assert meta["agent_roles"] == ["general"]
    assert meta["status"] == status and not meta["needs_merchant"]
    assert not meta["tools_used"] and not meta["rag"]["enabled"]
    assert meta["grounding"]["passed"] and meta["evidence"] == []
    assert answer and "退款成功" not in answer and "已取消" not in answer


@pytest.mark.parametrize("message,topic,role", [
    ("你好，多少钱", "price_breakdown", "billing"),
    ("谢谢，请查一下库存", "inventory_query", "catalog"),
    ("帮我查询订单物流", "delivery_status", "order_fulfillment"),
    ("你好，帮我取消订单", "cancel_policy", "billing"),
    ("谢谢，请转人工", "human_handoff", "escalation"),
])
def test_specific_business_topics_override_generic_prefixes(message, topic, role):
    _, meta = CommerceSupport(use_llm=False).reply(message, PRODUCT)
    assert topic in meta["topics"] and role in meta["agent_roles"]
    assert not {"query", "request", "greeting", "feedback"}.intersection(meta["topics"])


def test_model_generic_labels_have_no_specialist_call_but_keep_fact_verification():
    # Scripted provider checks protocol and routing; it is not a quality score.
    model = ScriptedModel(["greeting", "price_breakdown"])
    answer, meta = CommerceSupport(llm_client=model).reply("你好，多少钱", PRODUCT)
    assert meta["topics"] == ["greeting", "price_breakdown"]
    assert meta["intent"] == "price_breakdown" and meta["routing"]["primary"] == "billing"
    assert {c["intent"] for c in meta["agent_contributions"] if c.get("phase") != "handoff_planning"} == {"price_breakdown"}
    assert {t["name"] for t in meta["tools_used"]} == {"get_current_product"}
    assert meta["grounding"]["factual_sections"] == 1 and meta["grounding"]["passed"]
    assert meta["model_use"].get("output_rejected") is not True and "99.00" in answer


def test_generic_operation_request_never_becomes_a_trade_confirmation():
    model = ScriptedModel(["request"], attack="price")
    answer, meta = CommerceSupport(llm_client=model).reply("帮我处理一下", PRODUCT)
    assert meta["intent"] == "request" and meta["status"] == "clarifying"
    assert meta["model_use"]["output_rejected"]
    assert meta["grounding"]["passed"] and not meta["needs_merchant"]
    assert meta["tools_used"] == [] and meta["agent_contributions"] == []
    assert "退款成功" not in answer and "1.00" not in answer


@pytest.mark.parametrize("message,needs_merchant", [
    ("请联系客服", True),
    ("不要联系客服，只问多少钱", False),
])
def test_explicit_customer_service_request_respects_handoff_consent(message, needs_merchant):
    _, meta = CommerceSupport(use_llm=False).reply(message, PRODUCT)
    assert meta["needs_merchant"] is needs_merchant
    if needs_merchant:
        assert meta["routing"]["primary"] == "escalation"
        assert meta["routing"]["handoff_requested"]
        assert meta["handoff_summary"] and meta["status"] == "escalated"


@pytest.mark.parametrize("role,topic", [("order_fulfillment", "order_query"), ("escalation", "human_handoff")])
def test_pool_has_five_roles_and_scoped_roles_cannot_read_products(role, topic):
    from servemind.agents.commerce_agents import build_agent_pool
    from servemind.mcp.commerce_tools import ROLE_SCOPES
    support = CommerceSupport(use_llm=False)
    assert {role.value for role in build_agent_pool()} == {"general", "catalog", "order_fulfillment", "billing", "escalation"}
    node = TaskNode("task-1", topic, role, "get_current_product", {"product_id": PRODUCT["id"]})
    from servemind.agents.task_graph import ParallelAgentRunner
    results, _ = ParallelAgentRunner(support.tools).run(TaskGraph((topic,), (node,)), {
        "product": PRODUCT, "allowed_tools": set(support.tools.tools), "permissions": {"catalog_read", "purchase_read"},
        "role_scopes": ROLE_SCOPES})
    assert results[0][1].error == "forbidden:tool_scope"


@pytest.mark.parametrize("topics,message,consent,attack", [
    (["human_handoff"], "请转人工", True, None),
    (["complaint"], "我要投诉", True, None),
    (["human_handoff"], "不要转人工", False, "overhandoff"),
])
def test_escalation_agent_checks_policy_and_server_respects_consent(topics, message, consent, attack):
    model = ScriptedModel(topics, attack=attack)
    answer, meta = CommerceSupport(llm_client=model).reply(message, PRODUCT)
    assert meta["agent_roles"] == ["escalation"]
    assert meta["task_graph"]["role_count"] == 1 and meta["task_graph"]["completed"] == 1
    assert meta["task_graph"]["model_success_rate"] == 1
    assert {c["agent"] for c in meta["agent_contributions"]} == {"escalation"}
    assert any("escalation 专业客服 Agent" in request[0]["content"] for request in model.requests)
    assert sum(t["name"] == "search_knowledge" for t in meta["tools_used"]) == 1
    assert any(t["name"] == "submit_merchant_handoff" for t in meta["tools_used"])
    if attack == "overhandoff":
        assert meta["agent_contributions"][0]["needs_merchant"]
    assert meta["needs_merchant"] is consent
    assert meta["handoff_workflow"]["policy_completed"] == 1
    assert meta["grounding"]["passed"] and answer


def test_order_and_delivery_are_verified_with_one_role_dispatch():
    model = ScriptedModel(["order_query", "delivery_status"])
    answer, meta = CommerceSupport(llm_client=model).reply("查询订单信息和物流进度", PRODUCT,
        {"order_alias": "我的订单", "fulfillment_status": "运输中"})
    assert meta["agent_roles"] == ["order_fulfillment"]
    assert meta["task_graph"]["role_count"] == 1 and meta["task_graph"]["completed"] == 2
    assert meta["task_graph"]["model_success_rate"] == 1
    assert {c["intent"] for c in meta["agent_contributions"] if c.get("phase") != "handoff_planning"} == {"order_query", "delivery_status"}
    assert all(t["name"] == "get_current_purchase" for t in meta["tools_used"])
    assert meta["grounding"]["passed"] and "运输中" in answer and not meta["needs_merchant"]


def test_handoff_policy_read_cannot_bypass_skill_denial():
    from servemind.core.skill_loader import Skill
    support = CommerceSupport(use_llm=False)
    support.skills.skills = [Skill("deny", "buyer", ("人工",), "禁止政策查询", ("human_handoff",),
                                  denied_tools=("search_knowledge",))]
    _, meta = support.reply("请转人工", PRODUCT)
    assert meta["handoff_workflow"]["policy_completed"] == 0
    assert meta["tools_used"][0]["error"] == "forbidden:tool_scope"
    assert meta["evidence"] == [] and meta["needs_merchant"]
