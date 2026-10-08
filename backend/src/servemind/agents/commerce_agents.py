from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    GENERAL = "general"
    ORDER = "order"
    FULFILLMENT = "fulfillment"
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
            AgentRole.ORDER: ("get_current_purchase", "search_knowledge"),
            AgentRole.FULFILLMENT: ("get_current_purchase", "search_knowledge"),
            AgentRole.BILLING: ("get_current_product", "get_current_purchase", "search_knowledge"),
            AgentRole.CATALOG: ("get_current_product", "search_knowledge"),
            AgentRole.ESCALATION: ("search_knowledge",),
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


class OrderAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.ORDER, "订单事实、价格和订单归属核验", ("确认订单", "核验归属", "解释订单字段"), ("get_order_facts", "get_price_breakdown")))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        ids = [item.get("evidence_id", "") for item in evidence if item.get("kind") in ("order", "price")]
        return AgentContribution(agent=self.profile.role, status="verified_fact" if ids else "unsupported", evidence_ids=ids, limitations=["订单记录不等同于支付成功"])


class FulfillmentAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.FULFILLMENT, "配送、承诺、拆包和履约事实核验", ("读取配送节点", "比较承诺时间", "判断配送边界"), ("get_delivery_timeline", "get_order_facts"), ("无法确认签收人或赔偿资格",)))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        ids = [item.get("evidence_id", "") for item in evidence if item.get("kind") == "delivery"]
        escalated = any(word in message for word in ("没收到", "未收到", "丢件", "赔偿", "投诉"))
        return AgentContribution(agent=self.profile.role, status="escalated" if escalated else ("verified_fact" if ids else "unsupported"), evidence_ids=ids, limitations=["无法确认签收人、投递位置或赔偿资格"], needs_escalation=escalated)


class BillingAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.BILLING, "价格、优惠、支付和售后边界核验", ("区分订单金额与支付", "解释优惠", "判断人工审核"), ("get_price_breakdown", "search_knowledge"), ("实际退款、扣款和到账必须人工核验",)))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        ids = [item.get("evidence_id", "") for item in evidence if item.get("kind") in ("price", "knowledge", "order")]
        needs = any(word in message for word in ("退款", "扣款", "支付", "取消", "发票"))
        return AgentContribution(agent=self.profile.role, status="policy_review" if needs else ("verified_fact" if ids else "unsupported"), evidence_ids=ids, limitations=["当前数据不含支付流水、退款执行或到账结果"], needs_escalation=needs)


class CatalogAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.CATALOG, "匿名 SKU、库存和浏览事实查询", ("确认 SKU", "读取库存记录", "限制敏感画像披露"), ("get_sku", "get_inventory", "get_browsing_history")))


class EscalationAgent(CommerceAgent):
    def __init__(self):
        super().__init__(AgentProfile(AgentRole.ESCALATION, "人工升级和交接摘要", ("整理已知事实", "列出缺失字段", "生成交接原因"), ("create_handoff_summary",), ("投诉、争议、敏感或数据缺失场景",)))

    def _contribute(self, *, intent: Any, evidence: list[dict[str, Any]], message: str) -> AgentContribution:
        return AgentContribution(agent=self.profile.role, status="escalated", evidence_ids=[item.get("evidence_id", "") for item in evidence], limitations=["需要人工继续核验未提供的业务字段"], needs_escalation=True)


def build_agent_pool() -> dict[AgentRole, CommerceAgent]:
    return {role: cls() for role, cls in ((AgentRole.GENERAL, GeneralAgent), (AgentRole.ORDER, OrderAgent), (AgentRole.FULFILLMENT, FulfillmentAgent), (AgentRole.BILLING, BillingAgent), (AgentRole.CATALOG, CatalogAgent), (AgentRole.ESCALATION, EscalationAgent))}
