"""Agent-led decisions, scoped staging, and storage transaction contracts."""
import json
from copy import deepcopy

import pytest

from test_model_agent_runtime import ScriptedModel
from servemind.core.skill_loader import Skill
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.service.commerce_handoff import CommerceHandoffWorkflow
from servemind.service.commerce_support import CommerceSupport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SERVEMIND_RAG_BACKEND", "keyword")
    monkeypatch.setenv("SERVEMIND_SEMANTIC_INTENT", "false")
    monkeypatch.setenv("SERVEMIND_AGENT_MODE", "model")


class Planner(ScriptedModel):
    def __init__(self, topics, override):
        super().__init__(topics)
        self.override = override

    def chat(self, *, messages, tools=None, **kwargs):
        turn = super().chat(messages=messages, tools=tools, **kwargs)
        if "升级 Agent 交接规划器" in messages[0]["content"] and turn.message.get("tool_calls"):
            call = turn.message["tool_calls"][0]
            plan = json.loads(call["function"]["arguments"])
            plan.update(self.override)
            call["function"]["arguments"] = json.dumps(plan)
        return turn


def test_agent_selects_summary_and_action_after_all_facts_are_available():
    model = Planner(["price_breakdown", "inventory_query", "human_handoff"], {
        "question_summary": "请转人工", "merchant_action": "核对商品信息并回复买家"})
    _, meta = CommerceSupport(llm_client=model).reply("查价格和库存，请转人工", PRODUCT)
    assert meta["handoff_workflow"]["source"] == "model_agent"
    summary = meta["handoff_summary"]
    assert summary["question_summary"] == "请转人工"
    assert summary["merchant_action"] == "核对商品信息并回复买家"
    assert {f["field"] for f in summary["verified_facts"]} >= {"display_price", "catalog_status"}
    packet = next(json.loads(m[1]["content"]) for m in model.requests if "升级 Agent 交接规划器" in m[0]["content"])
    assert {c["agent"] for c in packet["role_results"]} == {"billing", "catalog", "escalation"}
    assert set(summary["pending_topics"]) == {"human_handoff"}
    assert meta["handoff_submission"]["status"] == "pending_commit"  # Replay has no DB commit.
    assert any(any(m["role"] == "tool" and "pending_commit" in m["content"] for m in req)
               for req in model.requests)
    assert any(c["stage"] == "agent:escalation:handoff_plan" for c in meta["agent_runtime"]["calls"])


def test_agent_pending_items_are_kept_in_memory_topics():
    model = Planner(["price_breakdown", "human_handoff"], {
        "pending_topics": ["price_breakdown", "human_handoff"]})
    _, meta = CommerceSupport(llm_client=model).reply("查价格并请转人工", PRODUCT)
    assert "price_breakdown" in meta["handoff_summary"]["pending_topics"]
    assert "price_breakdown" in meta["pending_topics"]
    assert "price_breakdown" not in meta["resolved_topics"]
    assert meta["grounding"]["passed"]


def test_agent_can_choose_clarification_for_optional_missing_price():
    model = Planner(["price_breakdown"], {"decision": "clarify", "reason_code": "none"})
    _, meta = CommerceSupport(llm_client=model).reply("价格多少", {**PRODUCT, "display_price": None})
    assert "price_missing" in meta["handoff_workflow"]["allowed_reasons"]
    assert meta["handoff_workflow"]["source"] == "model_agent"
    assert not meta["needs_merchant"] and meta["handoff_summary"] is None
    phase = next(c for c in meta["agent_contributions"] if c.get("phase") == "handoff_planning")
    assert phase["handoff_plan"]["decision"] == "clarify"


@pytest.mark.parametrize("override", [
    {"evidence_ids": ["invented"]},
    {"question_summary": "退款成功，商家同意赔偿"},
    {"pending_topics": []},
    {"merchant_id": "another-merchant"},
    {"decision": "none", "reason_code": "none"},
])
def test_invalid_or_omitted_explicit_handoff_uses_visible_same_tool_fallback(override):
    support = CommerceSupport(llm_client=Planner(["human_handoff"], override))
    _, meta = support.reply("请转人工", PRODUCT)
    assert support.monitor.summary()["handoff_planning_failure_count"] == 1
    assert meta["handoff_workflow"]["model_plan_failed"]
    assert meta["handoff_workflow"]["source"] == "deterministic_agent_fallback"
    assert meta["needs_merchant"] and meta["handoff_summary"]["question_summary"] == "请转人工"
    assert "invented" not in meta["handoff_summary"]["evidence_ids"]
    assert any(t["name"] == "submit_merchant_handoff" and t["status"] == "error" for t in meta["tools_used"])


def test_skill_denial_blocks_model_and_fallback_without_announcing_handoff():
    model = ScriptedModel(["human_handoff"])
    support = CommerceSupport(llm_client=model)
    support.skills.skills = [Skill("deny-submit", "buyer", ("人工",), "不可提交", ("human_handoff",),
                                  denied_tools=("submit_merchant_handoff",))]
    answer, meta = support.reply("请转人工", PRODUCT)
    assert not meta["needs_merchant"] and meta["handoff_submission"] is None
    assert "未能提交" in answer and "不会通知商家" in answer
    assert not any("升级 Agent 交接规划器" in req[0]["content"] for req in model.requests)


@pytest.mark.parametrize("role,phase,writes,permission", [
    ("billing", True, True, True), ("escalation", False, True, True),
    ("escalation", True, False, True), ("escalation", True, True, False),
])
def test_handoff_tool_requires_role_phase_permission_and_write_capability(role, phase, writes, permission):
    support = CommerceSupport(use_llm=False)
    _, meta = support.reply("请转人工", PRODUCT)
    plan = meta["handoff_submission"]["plan"]
    result = support.tools.call("submit_merchant_handoff", plan, {
        "agent_role": role, "handoff_phase": phase, "allow_writes": writes,
        "allowed_tools": {"submit_merchant_handoff"}, "permissions": {"handoff_prepare"} if permission else set()})
    assert not result.success



def test_commit_rejects_cross_conversation_scope_and_summary_tampering():
    _, meta = CommerceSupport(use_llm=False).reply("请转人工", PRODUCT,
        conversation_id="conv", buyer_id="buyer", merchant_id="merchant")
    conversation = {"id": "conv", "buyer": {"id": "buyer"}, "merchant": {"id": "merchant"},
                    "product": PRODUCT, "handoff_state": "ai_support"}
    for mutate in (lambda m: m["handoff_submission"]["scope"].update(conversation_id="another"),
                   lambda m: m.update(handoff_summary={"merchant_action": "退款成功"})):
        tampered = deepcopy(meta)
        mutate(tampered)
        with pytest.raises(PermissionError):
            CommerceHandoffWorkflow.validate_commit(tampered, conversation)
    CommerceHandoffWorkflow.validate_commit(meta, conversation)
    assert meta["handoff_submission"]["status"] == "committed"
