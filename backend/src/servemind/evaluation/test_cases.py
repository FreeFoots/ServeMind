"""可审计的 Agent 评测词条。

词条不包含真实个人信息，只使用项目数据中的匿名订单/SKU 和知识主题。
"""

from __future__ import annotations

from servemind.evaluation.data_cases import DATA_ORDER_CASES

INTENT_CASES = [
    {"id": "intent-001", "message": "查询订单 81a6fa818d 配送到哪里了", "expected": "delivery_status"},
    {"id": "intent-002", "message": "物流还在路上吗", "expected": "delivery_status"},
    {"id": "intent-003", "message": "订单没收到，已经超时了", "expected": "delivery_exception"},
    {"id": "intent-004", "message": "包裹丢了怎么处理", "expected": "delivery_exception"},
    {"id": "intent-005", "message": "原价和优惠后价格怎么算", "expected": "price_breakdown"},
    {"id": "intent-006", "message": "我实付了多少钱", "expected": "price_breakdown"},
    {"id": "intent-007", "message": "查一下 SKU a234e08c57 的属性", "expected": "sku_query"},
    {"id": "intent-008", "message": "这个商品是什么规格和品牌", "expected": "sku_query"},
    {"id": "intent-009", "message": "仓库今天有库存记录吗", "expected": "inventory_query"},
    {"id": "intent-010", "message": "SKU 81a6fa818d 现在有货吗", "expected": "inventory_query"},
    {"id": "intent-011", "message": "我想看最近浏览过的商品", "expected": "browsing_history"},
    {"id": "intent-012", "message": "我买了什么订单", "expected": "order_query"},
    {"id": "intent-013", "message": "订单详情能告诉我吗", "expected": "order_query"},
    {"id": "intent-014", "message": "我要退款", "expected": "refund_policy"},
    {"id": "intent-015", "message": "退货退款需要什么条件", "expected": "refund_policy"},
    {"id": "intent-016", "message": "我想取消订单", "expected": "cancel_policy"},
    {"id": "intent-017", "message": "帮我找人工客服", "expected": "human_handoff"},
    {"id": "intent-018", "message": "我要投诉这个配送问题", "expected": "complaint"},
    {"id": "intent-019", "message": "你好", "expected": "other"},
    {"id": "intent-020", "message": "我不知道该查什么", "expected": "other"},
]

CONTEXT_CASES = [
    {"id": "context-001", "turns": ["查询订单 81a6fa818d 配送", "那它现在物流到哪一步了"], "expected_focus": "81a6fa818d", "expected_last_intent": "delivery_status"},
    {"id": "context-002", "turns": ["订单 81a6fa818d 价格怎么算", "那优惠是多少"], "expected_focus": "81a6fa818d", "expected_last_intent": "price_breakdown"},
    {"id": "context-003", "turns": ["查询订单 81a6fa818d 配送", "换个问题，我要退款"], "expected_focus": "81a6fa818d", "expected_last_intent": "refund_policy"},
    {"id": "context-004", "turns": ["查询订单 ffffffffff 配送", "那就查一下 81a6fa818d 的物流"], "expected_focus": "81a6fa818d", "expected_last_intent": "delivery_status"},
    {"id": "context-005", "turns": ["查 SKU a234e08c57 属性", "它现在有库存吗"], "expected_focus": None, "expected_last_intent": "inventory_query"},
    {"id": "context-006", "turns": ["查询订单 81a6fa818d 配送", "不要看别的订单，继续说这个"], "expected_focus": "81a6fa818d", "expected_last_intent": "order_query"},
]

RAG_CASES = [
    {"id": "rag-001", "query": "退款规则是什么", "intent": "refund_policy", "must_retrieve": ["refund_review"]},
    {"id": "rag-002", "query": "取消订单的政策", "intent": "cancel_policy", "must_retrieve": ["cancellation_review"]},
    {"id": "rag-003", "query": "我要投诉并找人工", "intent": "complaint", "must_retrieve": ["merchant_handoff"]},
    {"id": "rag-004", "query": "商品品牌和 SKU 属性", "intent": "sku_query", "must_retrieve": [], "must_disable": True},
    {"id": "rag-005", "query": "仓库库存记录", "intent": "inventory_query", "must_retrieve": [], "must_disable": True},
    {"id": "rag-006", "query": "不知道怎么办", "intent": "other", "must_retrieve": ["commerce_boundaries"]},
    {"id": "rag-007", "query": "退款什么时候到账，能保证成功吗", "intent": "refund_policy", "must_retrieve": ["refund_review"]},
    {"id": "rag-008", "query": "我想撤销订单，AI 可以直接帮我取消吗", "intent": "cancel_policy", "must_retrieve": ["cancellation_review"]},
    {"id": "rag-009", "query": "包裹显示送达却没收到，想投诉并请商家核实", "intent": "complaint", "must_retrieve": ["delivery_exception"]},
    {"id": "rag-010", "query": "请把问题交给商品所属商家人工回复", "intent": "human_handoff", "must_retrieve": ["merchant_handoff"]},
    {"id": "rag-011", "query": "别的商家可以看到我和 AI 的私聊吗", "intent": "human_handoff", "must_retrieve": ["privacy_and_identity"]},
    {"id": "rag-012", "query": "为什么不能直接说我已退款成功", "intent": "refund_policy", "must_retrieve": ["refund_review"]},
    {"id": "rag-013", "query": "查匿名用户之前浏览了哪些商品", "intent": "browsing_history", "must_retrieve": [], "must_disable": True},
    {"id": "rag-014", "query": "这个订单的真实物流位置在哪里", "intent": "delivery_status", "must_retrieve": [], "must_disable": True},
]

ANSWER_CASES = [
    {"id": "answer-001", "message": "查询订单 81a6fa818d 配送", "expected_status": "answered", "must_ground": True, "must_use": ["get_order_facts", "get_delivery_timeline"], "must_not_claim": ["支付成功", "签收人"]},
    {"id": "answer-002", "message": "查询订单 ffffffffff 配送", "expected_status": "unsupported", "must_ground": False, "must_use": [], "must_not_claim": ["已发货", "已送达"]},
    {"id": "answer-003", "message": "订单 81a6fa818d 没收到，帮我处理", "expected_status": "escalated", "must_ground": True, "must_use": ["get_order_facts", "get_delivery_timeline"], "must_not_claim": ["已赔偿", "退款已完成"]},
    {"id": "answer-004", "message": "我要退款", "expected_status": "escalated", "must_ground": True, "must_use": ["search_knowledge"], "must_not_claim": ["退款已经执行"]},
    {"id": "answer-005", "message": "查一下 SKU a234e08c57 属性", "expected_status": "answered", "must_ground": True, "must_use": ["get_sku"], "must_not_claim": ["真实商品名称"]},
    {"id": "answer-006", "message": "你好", "expected_status": "clarifying", "must_ground": False, "must_use": [], "must_not_claim": ["订单已处理"]},
]

JUDGE_CASES = [
    {"id": "judge-001", "message": "查询订单 81a6fa818d 配送", "criteria": {"helpfulness": 1, "grounding": 1, "boundary": 1, "clarity": 1, "handoff": 1}},
    {"id": "judge-002", "message": "订单 81a6fa818d 没收到，帮我处理", "criteria": {"helpfulness": 1, "grounding": 1, "boundary": 1, "clarity": 1, "handoff": 1}},
    {"id": "judge-003", "message": "我要退款", "criteria": {"helpfulness": 1, "grounding": 0, "boundary": 1, "clarity": 1, "handoff": 1}},
    {"id": "judge-004", "message": "这个 SKU 的真实品牌是什么", "criteria": {"helpfulness": 1, "grounding": 1, "boundary": 1, "clarity": 1, "handoff": 1}},
    {"id": "judge-005", "message": "你好", "criteria": {"helpfulness": 0, "grounding": 1, "boundary": 1, "clarity": 1, "handoff": 1}},
]

END_TO_END_CASES = [
    {"id": "e2e-001", "message": "查询订单 81a6fa818d 配送", "status": "answered", "must_use": {"get_order_facts", "get_delivery_timeline"}},
    {"id": "e2e-002", "message": "查询订单 ffffffffff 配送", "status": "unsupported", "must_use": set()},
    {"id": "e2e-003", "message": "订单 81a6fa818d 没收到，帮我处理", "status": "escalated", "must_use": {"get_order_facts", "get_delivery_timeline"}},
]

# 多意图、多轮交互：每轮允许切换或并行提出两个主题，检查主意图、候选意图、
# 工具集合和升级策略，而不是只测单句分类。
MULTI_INTENT_CASES = [
    {"id": "multi-001", "turns": ["订单 81a6fa818d 现在到哪里了？如果没收到我想退款", "先告诉我物流节点"], "expected_primary": ["delivery_status", "delivery_status"], "expected_any": ["refund_policy"], "must_tool": ["get_order_facts", "get_delivery_timeline"], "must_escalate": False},
    {"id": "multi-002", "turns": ["订单 81a6fa818d 价格怎么算，优惠了多少？", "如果金额不对我要投诉"], "expected_primary": ["price_breakdown", "complaint"], "expected_any": ["complaint"], "must_tool": ["get_price_breakdown"], "must_escalate": True},
    {"id": "multi-003", "turns": ["SKU a234e08c57 是什么规格，有库存吗？", "如果没货请找人工"], "expected_primary": ["sku_query", "human_handoff"], "expected_any": ["inventory_query"], "must_tool": ["get_sku"], "must_escalate": True},
    {"id": "multi-004", "turns": ["订单 81a6fa818d 的商品属性是什么？", "刚才那个订单配送呢"], "expected_primary": ["sku_query", "delivery_status"], "expected_any": ["order_query"], "must_tool": ["get_sku", "get_order_facts", "get_delivery_timeline"], "must_escalate": False},
    {"id": "multi-005", "turns": ["订单 81a6fa818d 没收到，包裹是不是丢了？", "我还要取消订单并退款"], "expected_primary": ["delivery_exception", "refund_policy"], "expected_any": ["cancel_policy"], "must_tool": ["get_order_facts", "get_delivery_timeline", "search_knowledge"], "must_escalate": True},
    {"id": "multi-006", "turns": ["你好，我想问商品属性和价格", "再帮我看一下库存"], "expected_primary": ["sku_query", "inventory_query"], "expected_any": ["price_breakdown"], "must_tool": ["get_sku", "get_inventory"], "must_escalate": False},
    {"id": "multi-007", "turns": ["订单 81a6fa818d 配送正常吗？", "如果晚了我要求赔偿并找商家"], "expected_primary": ["delivery_status", "human_handoff"], "expected_any": ["complaint"], "must_tool": ["get_order_facts", "get_delivery_timeline"], "must_escalate": True},
    {"id": "multi-008", "turns": ["我想取消订单，另外发票怎么处理？", "请直接转人工确认"], "expected_primary": ["cancel_policy", "human_handoff"], "expected_any": ["human_handoff"], "must_tool": ["search_knowledge"], "must_escalate": True},
    {"id": "multi-009", "turns": ["查订单 81a6fa818d 的优惠和配送", "订单已经送达了吗"], "expected_primary": ["price_breakdown", "delivery_status"], "expected_any": ["order_query"], "must_tool": ["get_price_breakdown", "get_order_facts", "get_delivery_timeline"], "must_escalate": False},
    {"id": "multi-010", "turns": ["SKU a234e08c57 有货吗？如果没有推荐别的", "再告诉我这个 SKU 的属性"], "expected_primary": ["inventory_query", "sku_query"], "expected_any": ["sku_query"], "must_tool": ["get_inventory", "get_sku"], "must_escalate": False},
    {"id": "multi-011", "turns": ["我没收到货，也想知道实付金额", "先查配送，再告诉我金额"], "expected_primary": ["delivery_exception", "price_breakdown"], "expected_any": ["delivery_status"], "must_tool": ["get_order_facts", "get_delivery_timeline", "get_price_breakdown"], "must_escalate": True},
    {"id": "multi-012", "turns": ["订单 81a6fa818d 能取消吗？已经发货了吗？", "那就帮我找商家"], "expected_primary": ["cancel_policy", "human_handoff"], "expected_any": ["delivery_status"], "must_tool": ["search_knowledge"], "must_escalate": True},
    {"id": "multi-013", "turns": ["这个商品什么品牌，多少钱？", "我对价格不满意要投诉"], "expected_primary": ["sku_query", "complaint"], "expected_any": ["price_breakdown"], "must_tool": ["get_sku"], "must_escalate": True},
    {"id": "multi-014", "turns": ["查一下订单 81a6fa818d，物流和优惠都说一下", "物流没更新，我要人工"], "expected_primary": ["delivery_status", "human_handoff"], "expected_any": ["price_breakdown"], "must_tool": ["get_order_facts", "get_delivery_timeline"], "must_escalate": True},
    {"id": "multi-015", "turns": ["你好，我想退款并查询订单状态", "订单号是 81a6fa818d"], "expected_primary": ["refund_policy", "order_query"], "expected_any": ["delivery_status"], "must_tool": ["search_knowledge"], "must_escalate": True},
    {"id": "multi-016", "turns": ["看看 SKU a234e08c57 的属性和库存", "这个 SKU 缺货怎么办"], "expected_primary": ["sku_query", "inventory_query"], "expected_any": ["inventory_query"], "must_tool": ["get_sku", "get_inventory"], "must_escalate": False},
    {"id": "multi-017", "turns": ["配送晚了、包裹没收到，还要投诉", "请把问题整理给商家"], "expected_primary": ["delivery_exception", "complaint"], "expected_any": ["human_handoff"], "must_tool": ["get_order_facts", "get_delivery_timeline"], "must_escalate": True},
    {"id": "multi-018", "turns": ["价格优惠和退款规则分别是什么", "我先了解规则，不需要转人工"], "expected_primary": ["price_breakdown", "refund_policy"], "expected_any": ["refund_policy"], "must_tool": ["search_knowledge"], "must_escalate": False},
    {"id": "multi-019", "turns": ["查询订单 81a6fa818d 配送，另外我想看商品属性", "继续说配送情况"], "expected_primary": ["delivery_status", "delivery_status"], "expected_any": ["sku_query"], "must_tool": ["get_order_facts", "get_delivery_timeline"], "must_escalate": False},
    {"id": "multi-020", "turns": ["我不清楚是退款还是取消，订单 81a6fa818d 有问题", "请先帮我判断应该找谁"], "expected_primary": ["order_query", "human_handoff"], "expected_any": ["refund_policy", "cancel_policy"], "must_tool": ["get_order_facts"], "must_escalate": True},
]

def _data_intent_cases() -> list[dict[str, str]]:
    cases = []
    for i, item in enumerate(DATA_ORDER_CASES, 1):
        cases.extend([
            {"id": f"data-intent-{i:03d}-delivery", "message": f"查询订单 {item['order_id']} 的物流", "expected": "delivery_status"},
            {"id": f"data-intent-{i:03d}-price", "message": f"订单 {item['order_id']} 的原价、优惠和实付是多少", "expected": "price_breakdown"},
            {"id": f"data-intent-{i:03d}-sku", "message": f"SKU {item['sku_id']} 的属性是什么", "expected": "sku_query"},
            {"id": f"data-intent-{i:03d}-exception", "message": f"订单 {item['order_id']} 没收到，帮我处理", "expected": "delivery_exception"},
        ])
    return cases


def _data_answer_cases() -> list[dict[str, object]]:
    cases = []
    for i, item in enumerate(DATA_ORDER_CASES, 1):
        cases.append({"id": f"data-answer-{i:03d}-order", "message": f"查询订单 {item['order_id']} 配送", "expected_status": "answered", "must_ground": True, "must_use": ["get_order_facts", "get_delivery_timeline"], "must_not_claim": ["退款已经执行", "已赔偿"]})
        cases.append({"id": f"data-answer-{i:03d}-price", "message": f"订单 {item['order_id']} 价格和优惠怎么算", "expected_status": "answered", "must_ground": True, "must_use": ["get_price_breakdown"], "must_not_claim": ["退款已经执行", "已赔偿"]})
    return cases


def _data_context_cases() -> list[dict[str, object]]:
    cases = []
    for i, item in enumerate(DATA_ORDER_CASES[:10], 1):
        cases.append({"id": f"data-context-{i:03d}", "turns": [f"查询订单 {item['order_id']} 的物流", "它的价格优惠呢", "现在我想退款"], "expected_focus": item["order_id"], "expected_last_intent": "refund_policy"})
    return cases


DATA_INTENT_CASES = _data_intent_cases()
DATA_ANSWER_CASES = _data_answer_cases()
DATA_CONTEXT_CASES = _data_context_cases()

DATA_MULTI_INTENT_CASES = [
    {"id": f"data-multi-{i:03d}", "turns": [f"订单 {item['order_id']} 的物流和价格优惠都查一下", f"如果没收到订单 {item['order_id']}，我要投诉并找人工"], "expected_primary": ["delivery_status", "complaint"], "expected_any": ["price_breakdown", "human_handoff"], "must_tool": ["get_order_facts", "get_delivery_timeline", "get_price_breakdown"], "must_escalate": True}
    for i, item in enumerate(DATA_ORDER_CASES, 1)
]


# Stage-six graph contract: verify every explicitly requested subtask, not a
# single primary-intent label. Cases are generated from 20 distinct anonymous
# orders, with independent positive and escalation turns.
TASK_GRAPH_CASES = [
    {"id": f"graph-{i:03d}-parallel", "message": f"订单 {item['order_id']} 的物流和价格优惠都查一下",
     "must_intents": ["delivery_status", "price_breakdown"],
     "must_tools": ["get_order_facts", "get_delivery_timeline", "get_price_breakdown"],
     "must_escalate": False}
    for i, item in enumerate(DATA_ORDER_CASES, 1)
] + [
    {"id": f"graph-{i:03d}-handoff", "message": f"订单 {item['order_id']} 没收到，我要投诉并找人工，顺便核对优惠金额",
     "must_intents": ["delivery_exception", "complaint", "human_handoff", "price_breakdown"],
     "must_tools": ["get_order_facts", "get_delivery_timeline", "get_price_breakdown"],
     "must_escalate": True}
    for i, item in enumerate(DATA_ORDER_CASES, 1)
] + [
    {"id": "graph-edge-001", "message": "SKU a234e08c57 的属性和库存记录",
     "must_intents": ["sku_query", "inventory_query"], "must_tools": ["get_sku", "get_inventory"],
     "must_escalate": False},
    {"id": "graph-edge-002", "message": "退款和取消规则分别是什么",
     "must_intents": ["refund_policy", "cancel_policy"], "must_tools": ["search_knowledge"],
     "must_escalate": False},
    {"id": "graph-edge-003", "message": "如果没收到就想退款；先查订单 81a6fa818d 的物流",
     "must_intents": ["delivery_status", "refund_policy"],
     "must_tools": ["get_order_facts", "get_delivery_timeline", "search_knowledge"],
     "must_escalate": False},
    {"id": "graph-edge-004", "message": "订单 ffffffffff 的物流和价格是多少",
     "must_intents": ["delivery_status", "price_breakdown"],
     "must_tools": ["get_order_facts", "get_delivery_timeline", "get_price_breakdown"],
     "must_escalate": False, "must_not_ground": True},
]
