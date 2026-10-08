from __future__ import annotations

import json

import pytest

from servemind.agents.model_runtime import ModelAgentRuntime, RequestLedger
from servemind.core.knowledge_base import KnowledgeBase
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.llm.deepseek_client import Completion, ChatTurn, DeepSeekClient
from servemind.mcp.commerce_tools import build_commerce_tools, commerce_graph
from servemind.mcp.tool_manager import CircuitBreaker, Tool, ToolManager
from servemind.service.commerce_support import CommerceSupport


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setenv("SERVEMIND_RAG_BACKEND", "keyword")
    monkeypatch.setenv("SERVEMIND_SEMANTIC_INTENT", "false")
    monkeypatch.setenv("SERVEMIND_AGENT_MODE", "model")


class ScriptedModel:
    """Protocol fixture, not evidence of real provider quality."""
    configured = True

    def __init__(self, topics=None, *, attack=None):
        self.topics = topics or ["price_breakdown", "inventory_query"]
        self.attack = attack
        self.requests = []

    def chat(self, *, messages, tools=None, **kwargs):
        self.requests.append(messages)
        system = messages[0]["content"]
        user = json.loads(messages[1]["content"])
        if "电商客服意图识别器" in system:
            result = {"primary": self.topics[0], "confidence": .95, "intents": self.topics}
            if self.attack == "cycle":
                result["tasks"] = [{"intent": self.topics[0], "depends_on": self.topics}]
            text = json.dumps(result)
            packet = {"role": "assistant", "content": text}
        elif "Response Composer" in system:
            sections = user["sections"]
            if self.attack == "price":
                sections[0]["text"] = "只要 ¥1.00，退款成功。"
            packet = {"role": "assistant", "content": json.dumps(sections, ensure_ascii=False)}
        else:
            previous = [m for m in messages if m["role"] == "tool"]
            if not previous or self.attack == "loop":
                params = {"product_id": user["product_id"]}
                name = "get_current_product"
                if self.attack == "scope":
                    params = {"product_id": "someone-elses-product"}
                if self.attack == "whitelist":
                    name = "get_current_purchase"
                packet = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(params)}}]}
            else:
                evidence = [e for m in previous for e in json.loads(m["content"])["evidence"]]
                refs = [e["evidence_id"] for e in evidence]
                if self.attack == "citation":
                    refs = ["invented-evidence"]
                packet = {"role": "assistant", "content": json.dumps({"analysis": "已核对商品记录。",
                    "evidence_ids": refs, "needs_merchant": self.attack == "overhandoff"})}
        return ChatTurn(packet, Completion(packet.get("content") or "", "fixture-model", 20, 10))


def test_model_intent_routing_specialists_tool_round_trip_and_composer():
    model = ScriptedModel()
    support = CommerceSupport(llm_client=model)
    answer, meta = support.reply("还有货吗，多少钱", PRODUCT)
    assert meta["agent_use"] == "model_intent_routing_and_specialist_tool_loops"
    assert meta["intent_three_way"]["weights"] == {"llm": .85, "embedding": 0, "pattern": .15}
    assert meta["task_graph"]["mode"] == "model_role_parallel"
    assert meta["task_graph"]["completed"] == 2
    assert all(c["status"] == "model_analyzed" for c in meta["agent_contributions"])
    assert {t["name"] for t in meta["tools_used"]} == {"get_current_product"}
    assert meta["model_use"]["calls"] == 6  # recognition + two tool/final pairs + composer
    assert meta["model_use"]["prompt_tokens"] == 120
    assert support.monitor.summary()["provider_usage"]["input_tokens"] == 120
    assert "99.00" in answer and meta["grounding"]["passed"]
    assert any(m["role"] == "tool" for request in model.requests for m in request)


@pytest.mark.parametrize("attack", ["loop", "scope", "whitelist", "citation"])
def test_specialist_failures_keep_authorized_fallback_and_remain_observable(attack):
    support = CommerceSupport(llm_client=ScriptedModel(["inventory_query"], attack=attack))
    _, meta = support.reply("还有货吗", PRODUCT)
    assert meta["agent_contributions"][0]["status"] == "deterministic_fallback"
    assert meta["agent_runtime"]["errors"]
    assert meta["grounding"]["passed"]
    if attack == "whitelist":
        assert meta["tools_used"][0]["error"] == "forbidden:tool_scope"
    assert support.model_runtime._select_instance("catalog") == "catalog:backup"


def test_model_cannot_inject_tasks_or_dependencies_into_recognition():
    support = CommerceSupport(llm_client=ScriptedModel(attack="cycle"))
    _, meta = support.reply("还有货吗，多少钱", PRODUCT)
    assert meta["intent_three_way"]["mode"] == "intent_recognition_failed_rule_fallback"
    assert any(e.startswith("intent_recognition:") for e in meta["agent_runtime"]["errors"])
    assert meta["grounding"]["passed"]


def test_composer_hallucination_rejected_and_all_usage_counted():
    support = CommerceSupport(llm_client=ScriptedModel(["price_breakdown"], attack="price"))
    answer, meta = support.reply("多少钱", PRODUCT)
    assert "99.00" in answer and "退款成功" not in answer and "1.00" not in answer
    assert meta["model_use"]["output_rejected"]
    assert meta["model_use"]["rejection_reason"] == "model_claim_gate"
    assert meta["model_use"]["calls"] == 4
    assert meta["model_use"]["prompt_tokens"] == 80


def test_model_cannot_erase_explicit_refund_handoff_and_handoff_still_uses_composer():
    model = ScriptedModel(["inventory_query"])
    _, meta = CommerceSupport(llm_client=model).reply("我要退款，请帮我办理", PRODUCT)
    assert meta["needs_merchant"]
    assert any(c["stage"] == "response_composer" for c in meta["agent_runtime"]["calls"])


def test_intent_understanding_inherits_only_scoped_context():
    class Memory:
        def get(self, key):
            return {"buyer_id": "another-buyer", "merchant_id": "m", "product_id": PRODUCT["id"],
                    "summary": "private other buyer", "topics": ["refund_policy"]}
        def put(self, *args):
            return True
    model = ScriptedModel(["price_breakdown"])
    CommerceSupport(llm_client=model, working_memory=Memory()).reply(
        "那多少钱", PRODUCT, conversation_id="test", buyer_id="current-buyer", merchant_id="m")
    assert "private other buyer" not in model.requests[0][1]["content"]


def test_circuit_breaker_opens_and_allows_one_recovery_probe():
    breaker = CircuitBreaker(threshold=2, recovery_seconds=0)
    assert breaker.allow()
    breaker.record(False)
    breaker.record(False)
    assert breaker.snapshot()["state"] == "open"
    assert breaker.allow()
    assert not breaker.allow()
    breaker.record(True)
    assert breaker.snapshot()["state"] == "closed"


def test_forbidden_calls_do_not_open_tool_circuit():
    manager = ToolManager()
    manager.register(Tool("t", "test", lambda p, c: []))
    for _ in range(5):
        assert manager.call("t", {}, {}).error == "forbidden:tool_scope"
    assert manager.stats()["t"]["circuit"]["state"] == "closed"


def test_operational_failure_opens_tool_circuit():
    def fail(*args):
        raise ConnectionError("unavailable")
    manager = ToolManager()
    manager.register(Tool("t", "test", fail))
    for _ in range(3):
        manager.call("t", {}, {"allowed_tools": {"t"}})
    assert manager.call("t", {}, {"allowed_tools": {"t"}}).error == "tool_circuit_open"


def test_provider_adapter_keeps_tool_calls_with_empty_content(monkeypatch):
    observed = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "x", "function": {"name": "t", "arguments": "{}"}}]}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
    def post(*args, **kwargs):
        observed.append(kwargs)
        return Response()
    monkeypatch.setattr("servemind.llm.deepseek_client.httpx.post", post)
    turn = DeepSeekClient(api_key="test-only").chat(messages=[{"role": "user", "content": "test"}],
        tools=[{"type": "function", "function": {"name": "t"}}], timeout=2)
    assert turn.message["tool_calls"] and turn.usage.prompt_tokens == 12
    assert observed[0]["timeout"] == 2
    assert observed[0]["json"]["tools"]


def test_deadline_blocks_unscheduled_agent_work():
    support = CommerceSupport(llm_client=ScriptedModel())
    ledger = RequestLedger(timeout=-1)
    graph = commerce_graph(("price_breakdown",), PRODUCT["id"], "多少钱")
    results, status = support.model_runtime.run(graph, "多少钱", {}, ledger)
    assert status["completed"] == 0 and results[0][1].error == "agent_timeout"


def test_parallel_roles_requery_facts_without_borrowing_each_others_evidence():
    model = ScriptedModel()
    _, meta = CommerceSupport(llm_client=model).reply("先查价格，再根据商品记录确认是否在售", PRODUCT)
    assert all(not node["depends_on"] for node in meta["task_graph"]["nodes"])
    packets = [json.loads(r[1]["content"]) for r in model.requests if "专业客服 Agent" in r[0]["content"]]
    stock = next(p for p in packets if p["topic"] == "inventory_query")
    assert not stock["dependency_evidence"]
    assert stock["collaboration"]["supporting"]
    assert meta["task_graph"]["completed"] == 2


def test_model_handoff_proposal_cannot_share_normal_catalog_consultation():
    _, meta = CommerceSupport(llm_client=ScriptedModel(attack="overhandoff")).reply("有货吗多少钱", PRODUCT)
    assert not meta["needs_merchant"]
    assert meta["handoff_summary"] is None


def test_price_agent_can_query_price_policy_but_not_borrow_purchase_permissions():
    class Capture(ScriptedModel):
        def chat(self, *, messages, tools=None, **kwargs):
            if tools and "price_breakdown" in messages[0]["content"]:
                assert "search_knowledge" in {t["function"]["name"] for t in tools}
                policy = next(t for t in tools if t['function']['name']=='search_knowledge')
                assert policy['function']['parameters']['properties']['intent']['enum']==['price_breakdown']
            return super().chat(messages=messages, tools=tools, **kwargs)
    _, meta = CommerceSupport(llm_client=Capture(["price_breakdown"])).reply("多少钱", PRODUCT)
    assert not meta["agent_runtime"]["errors"]


def test_previous_clause_negation_does_not_authorize_refund_success_claim():
    class Poison(ScriptedModel):
        def chat(self, *, messages, **kwargs):
            turn = super().chat(messages=messages, **kwargs)
            if "Response Composer" in messages[0]["content"]:
                sections = json.loads(messages[1]["content"])["sections"]
                sections[0]["text"] += "未核实，退款成功。"
                text = json.dumps(sections, ensure_ascii=False)
                return ChatTurn({"role": "assistant", "content": text}, Completion(text, "fixture", 20, 10))
            return turn
    answer, meta = CommerceSupport(llm_client=Poison(["price_breakdown"])).reply("多少钱", PRODUCT)
    assert "退款成功" not in answer and meta["model_use"]["output_rejected"]
