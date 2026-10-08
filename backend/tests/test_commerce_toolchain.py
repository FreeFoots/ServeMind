from __future__ import annotations

import copy
import json

import pytest

from servemind.agents.task_graph import ParallelAgentRunner, TaskGraph, TaskNode
from servemind.core.knowledge_base import KnowledgeBase
from servemind.core.skill_loader import Skill
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES
from servemind.evaluation.human_review import case_digest, human_review_gate
from servemind.llm.deepseek_client import Completion
from servemind.mcp.commerce_tools import build_commerce_tools, ROLE_SCOPES
from servemind.mcp.knowledge_client import KnowledgeMCPClient
from servemind.service.commerce_support import CommerceSupport


@pytest.fixture(autouse=True)
def keyword_backend(monkeypatch):
    monkeypatch.setenv("SERVEMIND_RAG_BACKEND", "keyword")
    monkeypatch.setenv("SERVEMIND_SEMANTIC_INTENT", "false")


def test_commercial_agents_execute_role_scoped_tools_and_bind_each_answer():
    answer, metadata = CommerceSupport(use_llm=False).reply("价格库存和物流一起查", PRODUCT)
    assert {"billing", "catalog", "fulfillment"}.issubset(metadata["agent_roles"])
    assert {"get_current_product", "get_current_purchase"} == {t["name"] for t in metadata["tools_used"]}
    sections = {s["intent"]: s for s in metadata["answer_sections"]}
    assert sections["price_breakdown"]["evidence_ids"]
    assert sections["inventory_query"]["evidence_ids"]
    assert sections["delivery_status"]["status"] == "clarifying"
    assert metadata["grounding"]["passed"]
    assert "99.00" in answer and "物流" in answer


def test_node_cannot_borrow_another_role_tool_scope():
    tools = build_commerce_tools(KnowledgeBase())
    graph = TaskGraph(("inventory_query",), (TaskNode("task-1", "inventory_query", "general", "get_current_product", {"product_id": PRODUCT["id"]}),))
    results, _ = ParallelAgentRunner(tools).run(graph, {"allowed_tools": set(tools.tools),
        "permissions": {"catalog_read"}, "role_scopes": ROLE_SCOPES, "product": PRODUCT})
    assert results[0][1].error == "forbidden:tool_scope"


def test_forged_product_or_purchase_scope_is_rejected():
    tools = build_commerce_tools(KnowledgeBase())
    context = {"allowed_tools": set(tools.tools), "permissions": {"catalog_read", "purchase_read"},
               "product": PRODUCT, "buyer_id": "buyer-a", "purchase": {"buyer_id": "buyer-b"}}
    assert not tools.call("get_current_product", {"product_id": "other-product"}, context).success
    assert not tools.call("get_current_purchase", {"product_id": PRODUCT["id"]}, context).success


def test_public_policy_tool_runs_real_mcp_sdk_exchange():
    result = KnowledgeMCPClient(KnowledgeBase()).search("退款规则", "refund_policy")
    assert result["transport"] == "mcp_sdk_memory"
    assert result["enabled"]
    assert {i["title"] for i in result["items"]}.issubset({"refund_review", "return_preparation"})


def test_skill_denial_is_enforced_in_execution():
    support = CommerceSupport(use_llm=False)
    support.skills.skills = [Skill("deny", "buyer", ("价格",), "禁止读取商品", ("price_breakdown",),
                                   denied_tools=("get_current_product",))]
    _, meta = support.reply("商品价格多少", PRODUCT)
    assert meta["tools_used"][0]["error"] == "forbidden:tool_scope"
    assert meta["needs_merchant"] is True
    assert not meta["answer_sections"][0]["evidence_ids"]


def test_hallucinated_model_price_is_rejected_and_usage_is_still_counted():
    class Provider:
        configured = True
        def complete(self, **kwargs):
            return Completion(json.dumps([{"intent": "price_breakdown", "text": "只要 ¥1.00"}]), "deepseek-flash", 40, 10)
    support = CommerceSupport(llm_client=Provider())
    answer, meta = support.reply("多少钱", PRODUCT)
    assert "99.00" in answer and "1.00" not in answer
    assert meta["model_use"]["output_rejected"]
    assert support.monitor.summary()["provider_usage"]["input_tokens"] == 40


def test_customer_response_does_not_expose_internal_catalog_terms():
    class Provider:
        configured = True
        def complete(self,**kwargs):
            return Completion(json.dumps([{'intent':'price_breakdown','text':'目录状态 available，商家申报价 ¥99.00'}]),'test-model',40,10)
    answer, meta = CommerceSupport(llm_client=Provider()).reply('多少钱',PRODUCT)
    assert '目录状态' not in answer and 'available' not in answer and '商家申报价' not in answer
    assert meta['model_use']['output_rejected']


def test_human_gate_rejects_agent_labels_and_stale_case_hashes():
    results = [{"id": c["id"], "passed": True} for c in CURATED_COMMERCE_CASES]
    labels = {c["id"]: {"reviewer": "fixture", "reviewer_type": "human", "decision": "approve", "case_sha256": case_digest(c)}
              for c in CURATED_COMMERCE_CASES}
    assert human_review_gate(results, reviews=labels)["passed"]
    bad = copy.deepcopy(labels)
    for review in bad.values():
        review["reviewer_type"] = "agent"
    assert human_review_gate(results, reviews=bad)["approved"] == 0
    changed = copy.deepcopy(CURATED_COMMERCE_CASES[0])
    changed["turns"][0]["contains"].append("changed")
    bad = copy.deepcopy(labels)
    bad[changed["id"]]["case_sha256"] = case_digest(changed)
    assert human_review_gate(results, reviews=bad)["pending"] == 1


def test_merchant_handoff_feedback_and_participant_isolation(tmp_path):
    from fastapi.testclient import TestClient
    from servemind.api.main import create_app
    app = create_app(commerce_db_path=tmp_path / "commerce.sqlite3", use_llm=False)
    with TestClient(app) as client:
        def account(name, role):
            result = client.post("/v1/commerce/accounts/register", json={"username": name, "display_name": name,
                "password": "test-password-123", "role": role}).json()
            return {"Authorization": "Bearer " + result["token"]}
        merchant, buyer, stranger = account("merchant-gate", "merchant"), account("buyer-gate", "buyer"), account("stranger-gate", "buyer")
        product = client.post("/v1/commerce/products", headers=merchant, json={"title": "耳机", "price": "99.00"}).json()
        cid = client.post("/v1/commerce/conversations", headers=buyer, json={"product_id": product["id"]}).json()["id"]
        path = f"/v1/commerce/conversations/{cid}"
        sent = client.post(path + "/messages", headers=buyer, json={"content": "请找人工确认价格", "client_message_id": "handoff-1"})
        assert sent.status_code == 200, sent.text
        assert client.get(path, headers=merchant).json()["handoff_state"] == "awaiting_merchant"
        client.post(path + "/messages", headers=buyer, json={"content": "请找人工确认价格", "client_message_id": "handoff-1"})
        with app.state.commerce_store._db() as db:
            assert db.execute("SELECT count(*) FROM handoff_events WHERE conversation_id=?", (cid,)).fetchone()[0] == 1
        assert client.post(path + "/handoff", headers=buyer, json={"state": "resolved"}).status_code == 403
        assert client.post(path + "/handoff", headers=merchant, json={"state": "merchant_processing"}).status_code == 200
        reply = client.post(path + "/messages", headers=merchant, json={"content": "我会核实发货安排"}).json()
        assert reply["ai_message"] is None
        assert reply["message"]["metadata"]["authority"] == "merchant_account_message"
        assert client.get(path, headers=buyer).json()["handoff_state"] == "merchant_replied"
        continued=client.post(path+'/messages',headers=buyer,json={'content':'请联系店家再核实价格','client_message_id':'followup-merchant'}).json()
        assert continued['ai_message']['metadata']['human_collaboration_mode']=='advisory_only'
        assert client.get(path,headers=buyer).json()['handoff_state']=='merchant_replied'
        feedback_path = path + f"/messages/{sent.json()['ai_message']['id']}/feedback"
        assert client.post(feedback_path, headers=buyer, json={"rating": "not_helpful"}).status_code == 200
        assert client.post(feedback_path, headers=stranger, json={"rating": "helpful"}).status_code == 404
        assert client.post(path + "/handoff", headers=merchant, json={"state": "resolved"}).status_code == 200
        assert client.post(path + "/handoff", headers=merchant, json={"state": "merchant_processing"}).status_code == 409


def test_semantic_signal_routes_unknown_wording_but_does_not_override_specific_patterns(monkeypatch):
    from servemind.core import semantic_intent
    from servemind.core import vector_knowledge
    from servemind.core.intent_recognizer import Intent
    monkeypatch.setattr(vector_knowledge, "encode_query", lambda _: [1.0, 0.0])
    monkeypatch.setattr(semantic_intent, "prototypes", lambda: [(Intent.PRICE_BREAKDOWN, [1.0, 0.0]), (Intent.OTHER, [0.0, 1.0])])
    router = semantic_intent.SemanticIntentRouter(enabled=True)
    result, routes = router.resolve("买下来要几块")
    assert result.intent == Intent.PRICE_BREAKDOWN
    assert routes["routes"]["semantic"]["mode"] == "qwen_embedding_prototypes"
    result, _ = router.resolve("包裹丢了")
    assert result.intent == Intent.DELIVERY_EXCEPTION


def test_followup_recovers_from_persisted_turn_when_redis_is_unavailable():
    class EmptyMemory:
        def get(self, _): return {}
        def put(self, *args): return False
    support = CommerceSupport(use_llm=False, working_memory=EmptyMemory())
    answer, meta = support.reply("那它呢？继续说刚才的", PRODUCT, conversation_id="recover", buyer_id="buyer", merchant_id="merchant",
        recent_messages=[{"sender_type": "ai", "metadata": {"intent": "inventory_query", "topics": ["inventory_query"]}}])
    assert "在售" in answer
    assert meta["intent_three_way"]["routes"]["semantic"]["mode"] == "scoped_conversation_context"


def test_conditional_delivery_exception_keeps_current_logistics_and_refund_topics():
    from servemind.evaluation.commerce_cases import PURCHASE
    answer, meta = CommerceSupport(use_llm=False).reply("如果以后没收到，我要怎么退？先告诉我物流", PRODUCT, purchase=PURCHASE)
    assert {"refund_policy", "delivery_status"}.issubset(meta["topics"])
    assert "delivery_exception" not in meta["topics"]
    assert meta["needs_merchant"] is False
    assert "物流" in answer and "售后" in answer


def test_merchant_reply_keeps_pending_topics_in_durable_memory():
    class Memory:
        def context(self, *args):
            return {"resolved_topics": ["price_breakdown"], "pending_topics": ["invoice_query"]}
        def record_exchange(self, **kwargs):
            self.written = kwargs
            return True
    support = CommerceSupport(use_llm=False)
    support.durable_memory = memory = Memory()
    assert support.persist_turn(conversation={"id": "c", "buyer": {"id": "b"}, "merchant": {"id": "m"}, "product": PRODUCT},
        message_id="merchant-message", intent="merchant_reply", topics=["merchant_reply"], evidence_ids=[], needs_merchant=False)
    assert memory.written["pending_topics"] == ["invoice_query"]
    assert memory.written["resolved_topics"] == ["price_breakdown"]
    assert memory.written["needs_merchant"] is True


def test_invoice_does_not_reask_an_explicit_unit_header():
    answer, meta = CommerceSupport(use_llm=False).reply("发票抬头可以开单位的吗", PRODUCT)
    assert "单位发票抬头" in answer
    assert "个人抬头还是单位" not in answer
    assert "AI 不会" not in answer
    assert meta["grounding"]["passed"]
    assert "invoice_query" in meta["pending_topics"]
    assert "invoice_query" not in meta["resolved_topics"]
