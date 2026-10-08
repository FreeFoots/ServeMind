from __future__ import annotations

import re
import os
import uuid
from typing import Any

from servemind.agents.tools import build_tool_manager
from servemind.config.policy_registry import policy_metadata
from servemind.core.intent_recognizer import Intent, IntentResult, classify_message, classify_message_three_way
from servemind.core.knowledge_base import KnowledgeBase
from servemind.core.policy_evaluator import evaluate
from servemind.core.response_grounder import ResponseGrounder
from servemind.core.skill_loader import SkillManager
from servemind.memory.conversation_memory import SessionStore
from servemind.mcp.customer_data_provider import DATA_VERSION, DEFAULT_ORDER_ID, MSOMCustomerDataProvider, evidence_to_dict
from servemind.mcp.postgres_data_provider import PostgresMSOMProvider
from servemind.monitor.performance_monitor import Trace
from servemind.monitor.performance_monitor import OnlineMonitor
from servemind.service.handoff_service import HandoffOutbox
from servemind.agents.commerce_agents import AgentRole, build_agent_pool
from servemind.agents.task_graph import (IntentDecomposer, ParallelAgentRunner,
                                            EvidenceMerger, ResponseComposer, active_escalation)


class ServeMindAgent:
    """Supervisor runtime: understand, plan read-only tools, ground, and hand off safely."""

    def __init__(self) -> None:
        self.sessions = SessionStore()
        self.provider = (PostgresMSOMProvider() if os.getenv("SERVEMIND_DATA_BACKEND") == "postgres"
                         else MSOMCustomerDataProvider())
        self.knowledge = KnowledgeBase()
        self.skills = SkillManager()
        self.tools = build_tool_manager(self.provider, self.knowledge)
        self.grounder = ResponseGrounder()
        self.outbox = HandoffOutbox()
        # Supervisor 负责路由，领域角色负责受限贡献。
        self.agent_pool = build_agent_pool()
        self.monitor = OnlineMonitor()
        self.routing_penalties: dict[str, float] = {}
        self.decomposer = IntentDecomposer()
        self.parallel_runner = ParallelAgentRunner(self.tools)

    def create_session(self, *, channel: str, locale: str, actor_type: str = "buyer", user_id: str | None = None) -> dict[str, Any]:
        conversation_id = f"vc_{uuid.uuid4().hex[:12]}"
        self.sessions.create(conversation_id, channel=channel, locale=locale, actor_type=actor_type, user_id=user_id)
        return {"conversation_id": conversation_id, "current_focus": None, "actor_type": actor_type, "expires_at": None}

    def get_session(self, conversation_id: str) -> dict[str, Any] | None:
        session = self.sessions.get(conversation_id)
        if not session:
            return None
        return {"conversation_id": session.conversation_id, "channel": session.channel, "locale": session.locale, "actor_type": session.actor_type, "user_id": session.user_id, "messages": session.messages, "current_focus": session.current_focus, "pending_slots": session.pending_slots, "evidence": session.evidence}

    def agent_summary(self) -> dict[str, Any]:
        return {role.value: {"mission": agent.profile.mission, "workflow": list(agent.profile.workflow), "tool_scope": list(agent.profile.tool_scope), "success_rate": round(agent.success_rate, 3), "routing_penalty": self.monitor.penalties.get(role.value, 0.0)} for role, agent in self.agent_pool.items()}

    def _route_agents(self, intent: IntentResult, message: str) -> dict[str, Any]:
        text = (message or "").lower()
        if intent.intent in (Intent.HUMAN_HANDOFF, Intent.COMPLAINT) or any(word in text for word in ("投诉", "人工", "赔偿")):
            primary = AgentRole.ESCALATION
        elif intent.intent in (Intent.REFUND_POLICY, Intent.CANCEL_POLICY, Intent.PRICE_BREAKDOWN):
            primary = AgentRole.BILLING
        elif intent.intent in (Intent.DELIVERY_STATUS, Intent.DELIVERY_EXCEPTION):
            primary = AgentRole.FULFILLMENT
        elif intent.intent in (Intent.SKU_QUERY, Intent.INVENTORY_QUERY, Intent.BROWSING_HISTORY):
            primary = AgentRole.CATALOG
        elif intent.intent == Intent.ORDER_QUERY:
            primary = AgentRole.ORDER
        else:
            primary = AgentRole.GENERAL
        supporting = []
        if primary == AgentRole.FULFILLMENT and ("订单" in text or intent.entities.get("order_id")):
            supporting.append(AgentRole.ORDER)
        if primary == AgentRole.BILLING and intent.entities.get("order_id"):
            supporting.append(AgentRole.ORDER)
        if primary == AgentRole.ORDER and any(word in text for word in ("物流", "配送", "包裹")):
            supporting.append(AgentRole.FULFILLMENT)
        return {"primary": primary.value, "supporting": [item.value for item in supporting], "multi_agent": bool(supporting), "penalties": {role.value: round(self.monitor.penalties.get(role.value, 0.0), 3) for role in [primary, *supporting]}}

    def evidence(self) -> list[dict[str, Any]]:
        return self.evidence_for_order("81a6fa818d")

    def evidence_for_order(self, order_id: str) -> list[dict[str, Any]]:
        items = [evidence_to_dict(item) for item in self.provider.list_evidence(order_id)]
        if items:
            items.append({"evidence_id": "ev_policy_01", "kind": "policy", "display_label": "规则模拟层", "retrieved_at": "", "source": f"policy/{policy_metadata()['id']}", "facts": [{"field": "decision", "value": "policy_match", "confidence": 1.0}, {"field": "limitation", "value": "不确认支付、退款、取消、签收人或赔偿资格", "confidence": 1.0}], "scope": "policy", "version": policy_metadata()["version"]})
        return items

    def _resolve_order(self, message: str, session: Any) -> str | None:
        ids = re.findall(r"\b[0-9a-f]{10}\b", (message or "").lower())
        return ids[0] if ids else (session.current_focus or {}).get("order_id")

    def _plan(self, intent: IntentResult, order_id: str | None) -> list[str]:
        plan: list[str] = []
        if intent.intent in (Intent.ORDER_QUERY, Intent.DELIVERY_STATUS, Intent.DELIVERY_EXCEPTION) and order_id:
            plan += ["get_order_facts", "get_delivery_timeline"]
        if intent.intent == Intent.PRICE_BREAKDOWN and order_id:
            plan.append("get_price_breakdown")
        if intent.intent in (Intent.REFUND_POLICY, Intent.CANCEL_POLICY):
            plan.append("search_knowledge")
        if intent.intent == Intent.SKU_QUERY and intent.entities.get("sku_id"):
            plan.append("get_sku")
        if intent.intent == Intent.INVENTORY_QUERY and intent.entities.get("sku_id"):
            plan.append("get_inventory")
        if intent.intent == Intent.BROWSING_HISTORY:
            plan.append("get_browsing_history")
        return list(dict.fromkeys(plan))

    def respond(self, conversation_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.ensure(conversation_id)
        trace = Trace()
        intent = classify_message(message)
        intent_three_way = classify_message_three_way(message)
        routing = self._route_agents(intent, message)
        selected_roles = [routing["primary"], *routing["supporting"]]
        skills = self.skills.relevant_for_intent(message, intent.intent.value, session.actor_type)
        order_id = self._resolve_order(message, session)
        if not order_id and intent.intent in (Intent.ORDER_QUERY, Intent.DELIVERY_STATUS, Intent.DELIVERY_EXCEPTION) and any(token in message for token in ("查询订单", "查询配送", "查询物流")):
            order_id = DEFAULT_ORDER_ID
        explicit_sku = re.search(r"\bsku\s+([0-9a-f]{10})\b", message, re.IGNORECASE)
        sku_values = intent.entities.get("sku_id") or []
        sku_id = explicit_sku.group(1).lower() if explicit_sku else (
            sku_values[0] if sku_values and not order_id else None)
        graph = self.decomposer.decompose(message, intent, order_id=order_id, sku_id=sku_id)
        plan = [node.tool for node in graph.nodes]
        graph_roles = list(dict.fromkeys([routing["primary"], *routing["supporting"],
                                          *(node.role for node in graph.nodes)]))
        selected_roles = graph_roles
        routing["supporting"] = [role for role in graph_roles if role != routing["primary"]]
        routing["multi_agent"] = bool(routing["supporting"])
        allowed_tools = {tool for role in selected_roles
                         for tool in self.agent_pool[AgentRole(role)].profile.tool_scope}
        context = {"user_id": session.user_id, "allowed_tools": allowed_tools,
                   "permissions": {"historical_read", "policy_public"},
                   "agent_roles": selected_roles, "allow_writes": False,
                   "role_scopes": {role.value: agent.profile.tool_scope for role, agent in self.agent_pool.items()}}
        graph_status = {"total": len(graph.nodes), "completed": 0, "failed": 0,
                        "timed_out": False, "completion_rate": 0.0}
        factual_intents = {"order_query", "delivery_status", "delivery_exception",
                           "price_breakdown", "sku_query", "inventory_query", "browsing_history"}
        factual_requested = bool(factual_intents.intersection(graph.intents))
        if (intent.intent in (Intent.OTHER, Intent.CLARIFICATION) and intent.confidence < 0.5
            and not active_escalation(message, graph.intents)):
            answer = "我还不能确定您要查询订单、配送、价格、商品、库存，还是退款/取消规则。请补充一个具体目标或订单号。"
            session.pending_slots = ["具体问题或订单号"]
            status = "clarifying"
            evidence: list[dict[str, Any]] = []
            tools_used: list[dict[str, Any]] = []
        elif not order_id and intent.intent in (Intent.ORDER_QUERY, Intent.DELIVERY_STATUS, Intent.DELIVERY_EXCEPTION, Intent.PRICE_BREAKDOWN):
            wants_handoff = active_escalation(message, graph.intents)
            answer = ("我无法在没有本人订单绑定时核验配送或价格；已建议人工处理，尚未执行任何交易操作。"
                      if wants_handoff else
                      "请提供订单号，或补充下单时间、金额和商品信息，我才能定位订单并核对事实。")
            session.pending_slots = ["订单号或可定位订单的信息"]
            status = "escalated" if wants_handoff else "clarifying"
            evidence = []
            tools_used = []
        elif not sku_id and intent.intent in (Intent.SKU_QUERY, Intent.INVENTORY_QUERY):
            answer = "请提供具体 SKU 或商品编号；没有编号时我不能核验商品属性或历史库存。"
            session.pending_slots = ["SKU 或商品编号"]
            status = "clarifying"
            evidence = []
            tools_used = []
        else:
            results, graph_status = self.parallel_runner.run(graph, context)
            evidence = EvidenceMerger.merge(results)
            if factual_requested and not evidence:
                wants_handoff = active_escalation(message, graph.intents)
                answer = ("当前没有可核验的订单、商品或库存事实；已建议人工核验，AI 没有执行交易操作。"
                          if wants_handoff else
                          "当前没有可核验的订单、商品或库存事实，不能确认查询结果。请核对编号或转人工。")
                status = "escalated" if wants_handoff else "unsupported"
                if re.findall(r"\b[0-9a-f]{10}\b", message.lower()):
                    session.current_focus = None
            else:
                escalated = active_escalation(message, graph.intents)
                answer = self._answer(intent, evidence, escalated)
                answer = ResponseComposer.compose(graph.intents, evidence,
                                                  escalated=escalated, base_answer=answer)
                status = "escalated" if escalated else "answered"
                if escalated:
                    session.pending_slots = ["人工进一步核验配送或售后信息"]
            tools_used = [{"name": node.tool, "status": "ok" if result.success else
                           ("blocked" if result.error == "forbidden:tool_scope" else "failed"),
                           "latency_ms": result.latency_ms, "read_only": True} for node, result in results]
        # 角色 Agent 只读取已聚合 Evidence，不直接接触原始数据；工具范围由角色 profile 白名单声明。
        contributions = [self.agent_pool[AgentRole(role)].contribute(intent=intent, evidence=evidence, message=message).__dict__ for role in selected_roles if role in {item.value for item in AgentRole}]
        if (any(item.get("needs_escalation") for item in contributions)
            and active_escalation(message, graph.intents) and status == "answered"):
            status = "escalated"
        self.monitor.record(key=routing["primary"], success=status not in {"failed", "unsupported"}, latency_ms=trace.latency_ms, status=status)
        self.monitor.record_request(
            latency_ms=trace.latency_ms,
            status=status,
            route_success=status not in {"failed", "unsupported"},
            evidence_count=len(evidence),
            grounded=bool(evidence),
            grounding_required=factual_requested and status == "answered",
            tools=tools_used,
            handoff_recommended=status == "escalated",
            handoff_expected=status == "escalated",
            input_chars=len(message),
            output_chars=len(answer),
            # The current runtime exposes only read-only, role-scoped tools. Any
            # attempted write or role mismatch is rejected before execution; this
            # field remains ready for provider/tool-policy adapters.
            unauthorized_blocked=any(tool["status"] == "blocked" for tool in tools_used),
            cache_hit=False,
            rule_conflict=False,
            rule_expired=False,
        )
        if order_id and self.provider.current_focus(order_id):
            session.current_focus = self.provider.current_focus(order_id)
        session.messages.extend(({"role": "user", "content": message}, {"role": "assistant", "content": answer}))
        session.evidence = evidence
        self.sessions.record_turn(conversation_id, message=message, answer=answer, intent=intent.intent.value, evidence_ids=[item["evidence_id"] for item in evidence])
        return {"request_id": trace.request_id, "conversation_id": conversation_id, "answer": answer, "status": status, "intent": intent.intent.value, "intent_candidates": list(dict.fromkeys([item.value for item in intent.candidates] + list(graph.intents))), "intent_confidence": intent.confidence, "intent_three_way": intent_three_way, "actor_type": session.actor_type, "current_focus": session.current_focus, "clarification_needed": status == "clarifying", "tools_used": tools_used, "tool_role_whitelist": {role: list(self.agent_pool[AgentRole(role)].profile.tool_scope) for role in selected_roles}, "plan": plan, "task_graph": {"intents": list(graph.intents), "nodes": [{"id": node.id, "intent": node.intent, "role": node.role, "tool": node.tool, "depends_on": list(node.depends_on)} for node in graph.nodes], **graph_status}, "routing": routing, "agent_contributions": contributions, "evidence_ids": [item["evidence_id"] for item in evidence], "policy": evaluate(escalated=status == "escalated", status=status), "grounded": bool(evidence), "handoff": {"status": "recommended"} if status == "escalated" else None, "data_version": DATA_VERSION, "skills": [skill.name for skill in skills], "skill_prompt_injected": bool(skills), "rag": self.knowledge.search_for_intent(message, intent.intent.value), "memory": self.sessions.context(conversation_id), "context_used": len(session.recent_context()), "latency_ms": trace.latency_ms}

    def _answer(self, intent: IntentResult, evidence: list[dict[str, Any]], escalated: bool) -> str:
        if intent.intent in (Intent.DELIVERY_STATUS, Intent.DELIVERY_EXCEPTION, Intent.ORDER_QUERY) and evidence:
            return self.grounder.answer(escalated=escalated, evidence=evidence)
        if intent.intent == Intent.PRICE_BREAKDOWN and evidence:
            return "我已核对订单价格和优惠字段。当前回答只说明数据中的原价、优惠和实付记录，不代表支付成功或退款结果。"
        if intent.intent in (Intent.REFUND_POLICY, Intent.CANCEL_POLICY):
            docs = [item for item in evidence if item.get("kind") == "knowledge"]
            return "根据当前政策资料，我可以说明适用规则；是否实际退款或取消仍需订单状态和人工审核确认。" if not docs else f"我找到相关政策资料：{docs[0].get('title')}。具体是否适用还需要结合订单状态核验。"
        if intent.intent == Intent.SKU_QUERY:
            return "我可以核对匿名 SKU 的属性和履约字段，但不会根据匿名数据编造商品名称或规格。"
        if intent.intent == Intent.INVENTORY_QUERY:
            return "当前只能确认数据中是否存在指定日期和仓库的库存记录，不能把记录存在解释为实时库存数量。"
        return "我已记录这个问题，可以继续根据当前会话中的订单焦点和可核验数据处理。"
