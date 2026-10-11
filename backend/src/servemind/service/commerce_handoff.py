"""Validation and transactional envelope for Agent-led merchant handoff."""
from __future__ import annotations

import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from servemind.agents.task_graph import active_escalation
from servemind.llm.deepseek_client import redact_identifiers

HANDOFF_TOPICS = frozenset({"complaint", "human_handoff"})
HANDOFF_ACTIONS = (
    "核实待确认事项并回复买家", "核实签收及配送情况并回复买家",
    "确认售后办理条件并回复买家", "核对商品信息并回复买家", "处理买家投诉并回复买家",
)


class HandoffPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["handoff", "clarify", "none"]
    reason_code: Literal["explicit_request_or_dispute", "invoice_requires_merchant", "skill_conflict",
                         "verification_failed", "price_missing", "requested_field_missing", "none"]
    question_summary: str = Field(min_length=1, max_length=500)
    pending_topics: list[str] = Field(max_length=7)
    evidence_ids: list[str] = Field(max_length=100)
    merchant_action: Literal["核实待确认事项并回复买家", "核实签收及配送情况并回复买家",
                             "确认售后办理条件并回复买家", "核对商品信息并回复买家", "处理买家投诉并回复买家"]


class CommerceHandoffWorkflow:
    def constraints(self, message, topics, *, safety_topics, evidence, conflicts, failed_topics,
                    proposals=(), policy_results=()):
        """Admissible reasons and consent gates, not the model's business decision."""
        deferred = any(x in message for x in (
            "别通知", "不要通知", "不需要转", "暂时不要联系", "先了解规则", "只问规则",
            "不要转人工", "不用转人工", "不想转人工")) or bool(re.search(
                r"(?:不需要|不要|不用|不想|暂时不要|别)(?:找|联系|转)客服", message))
        requested = active_escalation(message, safety_topics)
        reasons = []
        if requested:
            reasons.append("explicit_request_or_dispute")
        if "invoice_query" in safety_topics:
            reasons.append("invoice_requires_merchant")
        if conflicts:
            reasons.append("skill_conflict")
        if failed_topics:
            reasons.append("verification_failed")
        if ("price_breakdown" in topics and not any(x in message for x in ("优惠", "实付", "成交"))
                and not any(e.get("field") == "display_price" for e in evidence)):
            reasons.append("price_missing")
        proposed = {c["intent"] for c in proposals if c.get("needs_merchant")}
        missing = {"inventory_query": "catalog_status", "sku_query": "description"}
        if any(t in proposed and not any(e.get("field") == f and e.get("value") for e in evidence)
               for t, f in missing.items()):
            reasons.append("requested_field_missing")
        return {"mode": "agent_led_guarded_handoff", "requested": requested, "deferred": deferred,
                "allowed_reasons": reasons, "required": (requested or "invoice_query" in safety_topics) and not deferred,
                "policy_total": len(policy_results),
                "policy_completed": sum(result.success for _, result in policy_results)}

    @staticmethod
    def submit(params, context):
        """Stage a validated request; the store commits it with the AI reply and memory."""
        if context.get("agent_role") != "escalation" or not context.get("handoff_phase"):
            raise PermissionError("handoff_role_or_phase")
        if time.monotonic() >= context.get("deadline", float("inf")):
            raise TimeoutError("handoff_deadline")
        if (context.get("merchant_id") and context["product"].get("merchant_id")
                and context["merchant_id"] != context["product"]["merchant_id"]):
            raise PermissionError("handoff_merchant_scope")
        plan = HandoffPlan.model_validate(params)
        gate = context["handoff_constraints"]
        evidence = {e["evidence_id"]: e for e in context["verified_evidence"]}
        if len(set(plan.evidence_ids)) != len(plan.evidence_ids) or not set(plan.evidence_ids) <= evidence.keys():
            raise ValueError("handoff_unknown_evidence")
        # Summaries quote the buyer, rather than inventing facts or completed trades.
        question = redact_identifiers(context["message"], max_chars=500)
        if not plan.question_summary.strip() or plan.question_summary not in question:
            raise ValueError("handoff_unattributed_summary")
        if len(set(plan.pending_topics)) != len(plan.pending_topics) or not set(plan.pending_topics) <= set(context["topics"]):
            raise ValueError("handoff_unknown_topic")
        if plan.decision != "handoff":
            if gate["required"]:
                raise ValueError("handoff_explicit_request_omitted")
            return {"status": "not_requested", "needs_merchant": False, "plan": plan.model_dump(), "summary": None}
        if gate["deferred"] or plan.reason_code not in gate["allowed_reasons"]:
            raise PermissionError("handoff_consent_or_reason")
        required_pending = set(context["failed_topics"]) | (set(context["topics"]) & HANDOFF_TOPICS)
        required_pending |= {s["intent"] for s in context["verified_sections"] if s["status"] != "answered"}
        if gate["required"]:
            required_pending |= set(context["topics"]) & {"refund_policy", "cancel_policy", "invoice_query", "delivery_exception"}
        if not required_pending <= set(plan.pending_topics):
            raise ValueError("handoff_pending_coverage")
        facts = [evidence[k] for k in plan.evidence_ids if evidence[k].get("kind") in {"product", "purchase"}]
        answered = [s["intent"] for s in context["verified_sections"]
                    if s["status"] == "answered" and s["intent"] not in plan.pending_topics]
        summary = {"reason": plan.reason_code, "question_summary": plan.question_summary,
                   "customer_question": question, "pending_topics": plan.pending_topics,
                   "answered_topics": answered, "evidence_ids": plan.evidence_ids,
                   "verified_facts": [{k: e[k] for k in ("field", "value", "evidence_id")} for e in facts],
                   "order_summary": {e["field"]: e["value"] for e in facts
                                     if e["field"] in {"order_alias", "fulfillment_status", "estimated_delivery"}}
                                    or {"availability": "未引用订单证据"},
                   "merchant_action": plan.merchant_action, "simulated": True}
        return {"status": "pending_commit", "needs_merchant": True, "plan": plan.model_dump(), "summary": summary,
                "scope": {k: context.get(k) for k in ("request_id", "conversation_id", "buyer_id", "merchant_id")}
                         | {"product_id": context["product"].get("id", "current_product")}}

    @staticmethod
    def validate_commit(metadata, conversation):
        """Recheck scope inside the existing store transaction, before making a chat visible."""
        if not metadata.get("needs_merchant"):
            return
        submission = metadata.get("handoff_submission") or {}
        expected = {"conversation_id": conversation["id"], "buyer_id": conversation["buyer"]["id"],
                    "merchant_id": conversation["merchant"]["id"], "product_id": conversation["product"]["id"],
                    "request_id": metadata.get("request_id")}
        if (submission.get("status") != "pending_commit" or submission.get("scope") != expected
                or submission.get("summary") != metadata.get("handoff_summary") or not submission.get("needs_merchant")):
            raise PermissionError("invalid_handoff_submission")
        submission["status"] = ("already_active" if conversation.get("handoff_state") in
                                {"merchant_processing", "merchant_replied"} else "committed")
