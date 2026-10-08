"""Broad, reproducible commerce and multi-intent regression matrix.

The source order identifiers are anonymous fixed fixtures.  Product personas,
categories, stock states and prices below are explicitly synthetic.  A case
is one concrete input/expectation pair, not a claim of independent annotation.
"""
from __future__ import annotations

from servemind.evaluation.data_cases import DATA_ORDER_CASES


INTENT_PATTERNS = (
    ("订单 {order} 的配送进度是什么", "delivery_status"),
    ("帮我追一下 {order} 这单的物流", "delivery_status"),
    ("包裹 {order} 到哪一步了", "delivery_status"),
    ("查询订单 {order} 什么时候送达", "delivery_status"),
    ("订单 {order} 原价是多少", "price_breakdown"),
    ("帮我核对 {order} 的优惠金额", "price_breakdown"),
    ("订单 {order} 实付多少钱", "price_breakdown"),
    ("{order} 这单折扣怎么算", "price_breakdown"),
    ("SKU {sku} 的品牌和属性是什么", "sku_query"),
    ("帮我看 SKU {sku} 的规格", "sku_query"),
    ("SKU {sku} 现在有货吗", "inventory_query"),
    ("订单 {order} 的退款规则是什么", "refund_policy"),
    ("我想取消订单 {order}", "cancel_policy"),
    ("我要投诉订单 {order}", "complaint"),
    ("订单 {order} 请帮我找人工", "human_handoff"),
)

EXPANDED_INTENT_CASES = [
    {"id": f"expanded-intent-{order_index:02d}-{pattern_index:02d}",
     "message": pattern.format(order=item["order_id"], sku=item["sku_id"]),
     "expected": expected}
    for order_index, item in enumerate(DATA_ORDER_CASES, 1)
    for pattern_index, (pattern, expected) in enumerate(INTENT_PATTERNS, 1)
]


GRAPH_PATTERNS = (
    ("帮我看订单 {order} 的配送进度，还要核对价格",
     ("delivery_status", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), False),
    ("订单 {order} 的发货情况和优惠金额一起核对",
     ("delivery_status", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), False),
    ("订单 {order} 的包裹到哪里了，同时看原价",
     ("delivery_status", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), False),
    ("订单 {order} 包裹丢了，投诉并找人工，同时核对价格",
     ("delivery_exception", "complaint", "human_handoff", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), True),
    ("订单 {order} 没收到，先查物流和价格，再找人工",
     ("delivery_exception", "human_handoff", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), True),
    ("订单 {order} 的物流、价格、退款规则都说一下",
     ("delivery_status", "price_breakdown", "refund_policy"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown", "search_knowledge"), False),
    ("订单 {order} 的配送、优惠和取消规则分别是什么",
     ("delivery_status", "price_breakdown", "cancel_policy"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown", "search_knowledge"), False),
    ("我想知道订单 {order} 的物流和优惠，也想看 SKU {sku} 的属性",
     ("delivery_status", "price_breakdown", "sku_query"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown", "get_sku"), False),
    ("订单 {order} 的配送和价格先核对；如果有问题再考虑退款",
     ("delivery_status", "price_breakdown", "refund_policy"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown", "search_knowledge"), False),
    ("订单 {order} 没收到，我要投诉并请人工核对优惠",
     ("delivery_exception", "complaint", "human_handoff", "price_breakdown"),
     ("get_order_facts", "get_delivery_timeline", "get_price_breakdown"), True),
)

EXPANDED_TASK_GRAPH_CASES = [
    {"id": f"expanded-graph-{order_index:02d}-{pattern_index:02d}",
     "message": pattern.format(order=item["order_id"], sku=item["sku_id"]),
     "must_intents": list(intents), "must_tools": list(tools), "must_escalate": escalate}
    for order_index, item in enumerate(DATA_ORDER_CASES, 1)
    for pattern_index, (pattern, intents, tools, escalate) in enumerate(GRAPH_PATTERNS, 1)
]


CATEGORIES = ("手机数码", "电脑办公", "家用电器", "家居日用", "母婴童装",
              "食品酒饮", "美妆护肤", "服饰鞋靴", "汽车用品", "图书文具")
STOCK_STATES = ("在售", "库存紧张", "预售", "暂时缺货", "已下架")
ORDER_STATES = ("待发货", "运输中", "配送异常", "已签收")


def _commerce_case(product_index: int, variant: int) -> dict:
    category = CATEGORIES[(product_index - 1) % len(CATEGORIES)]
    status = STOCK_STATES[(product_index - 1) % len(STOCK_STATES)]
    title = f"{category}{product_index}"
    price = f"{39 + product_index * 7}.90"
    product = {
        "id": f"eval-product-{product_index:02d}", "title": title,
        "category": category, "description": f"{title}，商品详情请向商家咨询。",
        "catalog_status": status, "display_price": price,
        "price_basis": "merchant_declared" if product_index % 2 else "simulated_catalog_price",
        "merchant_id": f"eval-merchant-{(product_index - 1) % 4 + 1}",
    }
    order = {"order_alias": f"我的订单 · {product_index:03d}",
             "fulfillment_status": ORDER_STATES[(product_index - 1) % len(ORDER_STATES)]}
    patterns = (
        (f"这件{title}现在多少钱", [f"¥{price}"], False, 1, None),
        (f"{title}有货吗，多少钱", [status, f"¥{price}"], False, 2, None),
        (f"这件{title}是现货吗", [status], False, 1, None),
        (f"{title}的价格和库存能一起说说吗", [status, f"¥{price}"], False, 2, None),
        (f"这件{title}能开发票吗", ["发票抬头", "商家"], True, 0, None),
        (f"我要投诉这件{title}，它还有货吗", [status, "商家"], True, 1, None),
        (f"请把{title}的价格交给店家确认", [f"¥{price}", "商家"], True, 1, None),
        (f"这件{title}如果想退款，先讲规则，暂时别通知店家", ["售后", "现在不会通知商家"], False, 1, None),
        (f"这件{title}的物流和价格都要查", [f"¥{price}", order["fulfillment_status"]], False, 1, order),
        (f"这件{title}的品牌和价格是什么", [f"¥{price}", "不能替商家确认"], False, 2, None),
        (f"我的订单没收到，请立即退款，商品是{title}", ["商家", "退款"], True, 1, None),
        (f"这件{title}的优惠金额和物流到哪了", ["物流进度", "优惠金额"], False, 0, None),
    )
    message, required, handoff, min_evidence, purchase = patterns[variant]
    case = {"id": f"expanded-commerce-{product_index:02d}-{variant + 1:02d}",
            "message": message, "product": product, "required": required,
            "handoff": handoff, "min_evidence": min_evidence}
    if purchase:
        case["purchase"] = purchase
    return case


EXPANDED_COMMERCE_CASES = [
    _commerce_case(product_index, variant)
    for product_index in range(1, 21)
    for variant in range(12)
]

assert len(EXPANDED_INTENT_CASES) == 300
assert len(EXPANDED_TASK_GRAPH_CASES) == 200
assert len(EXPANDED_COMMERCE_CASES) == 240
