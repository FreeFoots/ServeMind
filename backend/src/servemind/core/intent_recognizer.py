from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class Intent(str, Enum):
    ORDER_QUERY = "order_query"
    DELIVERY_STATUS = "delivery_status"
    DELIVERY_EXCEPTION = "delivery_exception"
    PRICE_BREAKDOWN = "price_breakdown"
    SKU_QUERY = "sku_query"
    INVENTORY_QUERY = "inventory_query"
    BROWSING_HISTORY = "browsing_history"
    REFUND_POLICY = "refund_policy"
    CANCEL_POLICY = "cancel_policy"
    COMPLAINT = "complaint"
    HUMAN_HANDOFF = "human_handoff"
    INVOICE_QUERY = "invoice_query"
    CLARIFICATION = "clarification"
    OTHER = "other"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    candidates: list[Intent]
    confidence: float
    entities: dict[str, list[str]]
    source_scores: dict[str, float] = field(default_factory=dict)
    urgency: str = "low"
    actor_type: str = "buyer"


_PATTERNS: dict[Intent, tuple[str, ...]] = {
    Intent.DELIVERY_EXCEPTION: ("没收到", "未收到", "没拿到", "丢件", "丢了", "延误", "超时", "投诉物流", "物流没更新"),
    Intent.DELIVERY_STATUS: ("配送", "物流", "包裹", "到哪里", "到哪", "送到", "送达", "发货", "什么时候到", "几天到", "多久能到"),
    Intent.PRICE_BREAKDOWN: ("价格", "多少钱", "价钱", "优惠", "折扣", "实付", "原价", "金额"),
    Intent.SKU_QUERY: ("商品", "sku", "规格", "品牌", "属性"),
    Intent.INVENTORY_QUERY: ("库存", "有货", "仓库", "备货", "缺货", "现货"),
    Intent.BROWSING_HISTORY: ("浏览记录", "看过", "浏览过", "最近看"),
    Intent.REFUND_POLICY: ("退款", "退货", "退钱", "怎么退"),
    Intent.CANCEL_POLICY: ("取消订单", "撤单", "取消"),
    Intent.HUMAN_HANDOFF: ("人工", "客服", "升级", "找商家", "交给店家", "交给商家", "转商家", "联系店家", "找店家", "转人工"),
    Intent.INVOICE_QUERY: ("发票", "开票", "抬头"),
    Intent.COMPLAINT: ("太差", "不满", "生气", "投诉"),
    Intent.ORDER_QUERY: ("订单", "买了什么", "订单状态", "订单详情"),
}


def _entities(message: str) -> dict[str, list[str]]:
    ids = list(dict.fromkeys(re.findall(r"\b[0-9a-f]{10}\b", (message or "").lower())))
    return {"order_id": ids, "sku_id": ids, "date": list(dict.fromkeys(re.findall(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", message or ""))), "amount": list(dict.fromkeys(re.findall(r"(?:¥|￥)?\d+(?:\.\d+)?\s*(?:元|块)?", message or "")))}


def classify_message(message: str) -> IntentResult:
    text = (message or "").lower()
    scores: dict[Intent, float] = {}
    for intent, patterns in _PATTERNS.items():
        hits = sum(1 for pattern in patterns if pattern.lower() in text)
        if hits:
            scores[intent] = min(1.0, 0.5 + hits * 0.15)
    actor = "merchant" if any(x in text for x in ("商家", "店铺", "库存", "备货", "发货量")) else "buyer"
    # 高风险/更具体的表达优先于宽泛关键词，例如“投诉配送”不能被
    # “配送”覆盖，“库存”不能被“SKU”覆盖。
    precedence = (
        Intent.COMPLAINT if "投诉" in text else None,
        Intent.DELIVERY_EXCEPTION if any(x in text for x in ("丢件", "丢了", "没收到", "未收到", "延误", "超时")) else None,
        Intent.INVENTORY_QUERY if any(x in text for x in ("库存", "有货", "缺货")) else None,
        Intent.BROWSING_HISTORY if any(x in text for x in ("浏览记录", "浏览过", "看过", "最近看")) else None,
    )
    preferred = next((item for item in precedence if item is not None and item in scores), None)
    if preferred is not None:
        scores[preferred] += 0.4
    if Intent.DELIVERY_STATUS in scores and Intent.ORDER_QUERY in scores:
        scores[Intent.DELIVERY_STATUS] = max(scores[Intent.DELIVERY_STATUS], scores[Intent.ORDER_QUERY] + 0.01)
    if not scores:
        return IntentResult(Intent.OTHER, [Intent.OTHER], 0.2, _entities(message), {"pattern": 0.2}, actor_type=actor)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    primary, confidence = ordered[0]
    urgency = "high" if primary in (Intent.DELIVERY_EXCEPTION, Intent.HUMAN_HANDOFF, Intent.COMPLAINT) else "low"
    return IntentResult(primary, [item[0] for item in ordered], min(confidence, 1.0), _entities(message), {"pattern": min(confidence, 1.0)}, urgency, actor)


def classify_message_three_way(message: str) -> dict[str, object]:
    """三路意图识别：模式、实体/槽位、会话语义。

    当前实现不依赖外部模型，三路结果由可审计的确定性信号组成；后续可将
    semantic 分支替换为模型或 embedding，而不改变输出契约。
    """
    base = classify_message(message)
    text = (message or "").lower()
    pattern = base.confidence
    entity = min(1.0, 0.35 + 0.2 * len([value for values in base.entities.values() for value in values]))
    semantic = min(1.0, 0.35 + 0.1 * sum(1 for marker in ("请", "查询", "说明", "怎么", "是否") if marker in text))
    fused = round(pattern * 0.5 + entity * 0.25 + semantic * 0.25, 4)
    return {
        "primary": base.intent.value,
        "candidates": [item.value for item in base.candidates],
        "confidence": fused,
        "urgency": base.urgency,
        "actor_type": base.actor_type,
        "entities": base.entities,
        "routes": {
            "pattern": {"intent": base.intent.value, "confidence": round(pattern, 4)},
            "entity_slot": {"entities": base.entities, "confidence": round(entity, 4)},
            "semantic": {"intent": base.intent.value, "confidence": round(semantic, 4)},
        },
    }
