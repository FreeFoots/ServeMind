"""Embedding semantic signal for wording not covered by deterministic patterns."""
from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache

from servemind.core.intent_recognizer import Intent, classify_message, classify_message_three_way

DESCRIPTIONS = {
    Intent.ORDER_QUERY: "查询本人订单详情、买了什么和订单记录",
    Intent.DELIVERY_STATUS: "查询快递物流进度以及什么时候收到货、发货和预计送达时间",
    Intent.DELIVERY_EXCEPTION: "包裹丢失或未收到货，签收争议和配送异常",
    Intent.PRICE_BREAKDOWN: "询问多少钱、商品售价、优惠、折扣和订单实付金额",
    Intent.SKU_QUERY: "询问商品品牌、尺寸、材质、型号和设备兼容性",
    Intent.INVENTORY_QUERY: "查询商品有没有现货、还能不能买、预售和补货",
    Intent.REFUND_POLICY: "咨询退货退款条件、流程和售后申请",
    Intent.CANCEL_POLICY: "撤销或取消已经下单的订单",
    Intent.HUMAN_HANDOFF: "不想继续和机器人对话，希望店家人工接待",
    Intent.INVOICE_QUERY: "询问个人或公司发票及开票抬头",
    Intent.COMPLAINT: "投诉商品服务或商家，希望处理不满、争议或赔偿诉求",
    Intent.BROWSING_HISTORY: "查询本人最近浏览或看过的商品记录",
    Intent.CLARIFICATION: "问题缺少关键指代或信息，需要澄清是哪件商品或哪个订单",
    Intent.OTHER: "你好，谢谢，再见，普通问候，不涉及业务问题",
}


@lru_cache(maxsize=1)
def prototypes():
    from servemind.core.vector_knowledge import encode_document
    return [(intent, encode_document(description)) for intent, description in DESCRIPTIONS.items()]


class SemanticIntentRouter:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (os.getenv("SERVEMIND_SEMANTIC_INTENT", "false").lower() == "true"
                        if enabled is None else enabled)

    def resolve(self, message: str):
        base = classify_message(message)
        routes = classify_message_three_way(message)
        routes["routes"]["semantic"] = {"mode": "not_needed", "intent": base.intent.value, "confidence": 0.0}
        # Precise patterns and actual-risk wording remain authoritative.
        if self.enabled and base.intent == Intent.OTHER and message.strip() not in {"你好", "谢谢", "再见", "您好"}:
            try:
                from servemind.core.vector_knowledge import encode_query
                vector = encode_query(message)
                ranks = sorted(((sum(a*b for a,b in zip(vector, anchor)), intent)
                                for intent, anchor in prototypes()), key=lambda row: row[0], reverse=True)
                score, intent = ranks[0]
                margin = score - ranks[1][0]
                routes["routes"]["semantic"] = {"mode": "qwen_embedding_prototypes", "intent": intent.value,
                                                 "confidence": round(score, 4), "margin": round(margin, 4)}
                if score >= 0.65 and margin >= 0.04 and intent != Intent.OTHER:
                    base = replace(base, intent=intent, candidates=[intent], confidence=min(score, 1.0))
                    routes["primary"], routes["confidence"] = intent.value, round(score, 4)
                    routes["candidates"] = [intent.value]
            except Exception:
                routes["routes"]["semantic"]["mode"] = "unavailable"
        return base, routes
