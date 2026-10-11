from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    GENERAL = "general"
    ORDER_FULFILLMENT = "order_fulfillment"
    BILLING = "billing"
    CATALOG = "catalog"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class AgentProfile:
    role: AgentRole
    mission: str
    workflow: tuple[str, ...]
    tool_scope: tuple[str, ...]
    handoff_conditions: tuple[str, ...] = ()

    @property
    def commerce_tool_scope(self) -> tuple[str, ...]:
        return {
            AgentRole.GENERAL: ("search_knowledge",),
            AgentRole.ORDER_FULFILLMENT: ("get_current_purchase", "search_knowledge"),
            AgentRole.BILLING: ("get_current_product", "get_current_purchase", "search_knowledge"),
            AgentRole.CATALOG: ("get_current_product", "search_knowledge"),
            AgentRole.ESCALATION: ("search_knowledge", "submit_merchant_handoff"),
        }[self.role]


@dataclass
class AgentContribution:
    agent: AgentRole
    status: str = "ready"
    answer: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    needs_escalation: bool = False


class CommerceAgent:
    profile: AgentProfile

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.total = 0
        self.success = 0
        self.total_ms = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    def contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        self.total += 1
        result = self._contribute(intent=intent, evidence=evidence, message=message)
        self.success += 1
        return result

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        return AgentContribution(agent=self.profile.role, evidence_ids=[item.get("evidence_id", "") for item in evidence])


class GeneralAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.GENERAL, "首轮接待、澄清和结果汇总", ("复述诉求", "判断领域", "提出最小澄清"), ("search_knowledge",)))


class OrderFulfillmentAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.ORDER_FULFILLMENT, "本人订单信息、配送进度、预计送达及履约异常核验", ("确认订单归属", "读取订单与配送节点", "解释进度及预计时间", "识别待人工核验事项"), ("get_order_facts", "get_delivery_timeline"), ("无法确认签收人或赔偿资格",)))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        ids = [item.get("evidence_id", "") for item in evidence if item.get("kind") in {"order", "delivery", "purchase"}]
        escalated = any(word in message for word in ("没收到", "未收到", "丢件", "赔偿", "投诉"))
        return AgentContribution(agent=self.profile.role, status="escalated" if escalated else ("verified_fact" if ids else "unsupported"), evidence_ids=ids, limitations=["无法确认签收人、投递位置或赔偿资格"], needs_escalation=escalated)


class BillingAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.BILLING, "商品价格、订单优惠和退货退款、取消、发票规则核验", ("区分标价与订单金额", "解释优惠与售后规则", "标明待商家确认事项"), ("get_price_breakdown", "search_knowledge"), ("实际退款、扣款和到账必须人工核验",)))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        ids = [item.get("evidence_id", "") for item in evidence if item.get("kind") in ("price", "knowledge", "order")]
        needs = any(word in message for word in ("退款", "扣款", "支付", "取消", "发票"))
        return AgentContribution(agent=self.profile.role, status="policy_review" if needs else ("verified_fact" if ids else "unsupported"), evidence_ids=ids, limitations=["当前数据不含支付流水、退款执行或到账结果"], needs_escalation=needs)


class CatalogAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.CATALOG, "商品描述、属性、页面库存与可售状态核验", ("确认商品", "读取页面属性及库存状态", "说明缺失字段与实时性边界"), ("get_sku", "get_inventory", "get_browsing_history")))


class EscalationAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.ESCALATION, "人工交接业务规划、结构化摘要与受控交接提交", ("核验公共政策", "汇总已核验事实与未解决诉求", "决定交接或澄清", "组织摘要并调用受控交接工具"), ("search_knowledge", "create_handoff_summary"), ("交接工具验证意愿、权限和证据，持久状态随回复事务提交",)))

    def fallback_plan(self, context):
        """Explicit offline/failure fallback; never reported as model planning."""
        gate = context["handoff_constraints"]
        needs = bool(gate["allowed_reasons"]) and not gate["deferred"]
        pending = {s["intent"] for s in context["verified_sections"] if s["status"] != "answered"}
        pending |= set(context["failed_topics"]) | (set(context["topics"]) & {"complaint", "human_handoff"})
        if gate["required"]:
            pending |= set(context["topics"]) & {"refund_policy", "cancel_policy", "invoice_query", "delivery_exception"}
        from servemind.llm.deepseek_client import redact_identifiers
        return {"decision": "handoff" if needs else "none",
                "reason_code": gate["allowed_reasons"][0] if needs else "none",
                "question_summary": redact_identifiers(context["message"], max_chars=500),
                "pending_topics": [t for t in context["topics"] if t in pending],
                "evidence_ids": [e["evidence_id"] for e in context["verified_evidence"]],
                "merchant_action": "核实待确认事项并回复买家"}

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        from servemind.agents.task_graph import active_escalation
        requested = active_escalation(message, (intent.intent.value,))
        return AgentContribution(agent=self.profile.role, status="handoff_review",
            evidence_ids=[item.get("evidence_id", "") for item in evidence],
            limitations=["不执行退款、赔偿或通知商家；交接须经服务端核验"], needs_escalation=requested)


def build_agent_pool() -> dict[AgentRole, CommerceAgent]:
    return {role: cls() for role, cls in ((AgentRole.GENERAL, GeneralAgent), (AgentRole.ORDER_FULFILLMENT, OrderFulfillmentAgent), (AgentRole.BILLING, BillingAgent), (AgentRole.CATALOG, CatalogAgent), (AgentRole.ESCALATION, EscalationAgent))}
