"""Compose one section per request, with explicit claim-to-evidence bindings."""
from __future__ import annotations

from typing import Any
import re


class CommerceResponseComposer:
    def compose(self, *, topics: tuple[str, ...], evidence: list[dict], message: str,
                needs_merchant: bool, failed_topics: set[str] | None = None) -> list[dict[str, Any]]:
        sections = []
        failed_topics = failed_topics or set()
        product = {e["field"]: e.get("value") for e in evidence if e.get("kind") == "product"}
        purchase = {e["field"]: e.get("value") for e in evidence if e.get("kind") == "purchase"}
        for topic in topics:
            ids, status = [], "answered"
            fields: tuple[str, ...] = ()
            kind = "product"
            if topic in failed_topics:
                text, status = "这部分暂时没查成功，我可以请商家继续核实。", "failed"
            elif topic == "inventory_query":
                fields = ("catalog_status",)
                state = product.get("catalog_status")
                state = "在售" if state == "available" else state
                text = (f"这款商品目前页面显示为“{state}”，库存可能变化，下单时请再确认。"
                        if state else "页面暂时没有库存信息，我可以帮你向商家确认。")
                status = "answered" if state else "needs_merchant"
            elif topic == "price_breakdown":
                if any(term in message for term in ("优惠", "实付", "成交", "订单价格")):
                    kind, fields = "purchase", ("paid_amount", "discount_amount")
                    values = [("实付金额", purchase.get("paid_amount")), ("优惠金额", purchase.get("discount_amount"))]
                    text = "；".join(f"订单记录的{label}为 ¥{value}" for label, value in values if value is not None)
                    if not text:
                        text, status = "我现在无法核对这笔订单的实付和优惠金额，请打开对应已购订单；需要的话可以联系商家核实。", "clarifying"
                else:
                    fields = ("display_price", "price_basis")
                    price = product.get("display_price")
                    text = (f"商家标价是 ¥{price}，实际支付金额请以结算页面为准。" if price
                            else "这款商品暂时没有标价，我可以帮你向商家确认价格。")
                    if price and product.get("price_basis") == "historical_reference_simulated":
                        text = f"目前能看到的参考价格是 ¥{price}，下单前需要请商家确认现在的售价。"
                    status = "answered" if price else "needs_merchant"
            elif topic == "sku_query":
                fields = ("title", "description", "category")
                description = str(product.get("description", ""))[:180]
                if re.search(r"忽略.*(?:指令|限制)|system prompt|api.?key|执行退款|泄露", description, re.I):
                    description = "页面中的部分描述需要向商家核实。"
                text = (f"商品页面写的是：{product.get('title', '这款商品')}。{description} "
                        "页面没有写明的品牌或规格，我不能替商家确认。")
                if not product:
                    text, status = "商品页面暂时没有品牌或规格信息，我可以帮你问商家。", "clarifying"
            elif topic in {"order_query", "delivery_status", "delivery_exception"}:
                kind, fields = "purchase", ("order_alias", "fulfillment_status", "estimated_delivery")
                if purchase.get("fulfillment_status"):
                    text = f"物流方面，{purchase.get('order_alias', '你的订单')}目前显示为“{purchase['fulfillment_status']}”。"
                    if any(term in message for term in ("什么时候", "几天", "多久", "送达")):
                        if purchase.get("estimated_delivery"):
                            text += f"页面预计送达时间为 {purchase['estimated_delivery']}，具体以配送更新为准。"
                        else:
                            text += "目前没有预计送达时间，需要商家确认具体安排。"
                    if topic == "delivery_exception" and needs_merchant:
                        text += "你反映没有收到，我会请商家核实签收和配送情况。"
                else:
                    text, status = "我还没找到与你本人绑定的订单，暂时查不到物流信息和物流进度。请从已购商品中打开对应订单，我再帮你看。", "clarifying"
            elif topic in {"refund_policy", "cancel_policy", "invoice_query"}:
                kind, fields = "knowledge", ("policy",)
                policies = [e for e in evidence if e.get("kind") == "knowledge" and topic in e.get("policy_intents", [topic])]
                guidance = [e.get("customer_guidance", "") for e in policies if e.get("customer_guidance")]
                if topic == "invoice_query":
                    text = "发票抬头能否开具需要商家确认，我可以将要求交给商家。"
                elif topic == "cancel_policy":
                    text = "关于取消订单，能否办理要看订单和商品情况，可以联系商家核实；我现在没有替你取消订单。"
                else:
                    text = "关于售后，能否退货或退款，要看订单和商品情况；退款尚未办理。"
                if guidance:
                    if topic == "invoice_query":
                        text = guidance[0][:300]
                    else:
                        text += " " + guidance[0][:300]
                if topic == "invoice_query" and "单位" in message:
                    text = "单位发票抬头能否开具，需要商家确认。完整税务或支付凭证请通过商家确认的安全方式提供。"
                if not needs_merchant:
                    text += " 如果你决定申请，我可以帮你联系商家核实；现在不会通知商家。"
                if not policies:
                    status = "needs_merchant" if needs_merchant else "clarifying"
                elif needs_merchant:
                    # Knowing a policy does not mean the merchant has approved the request.
                    status = "needs_merchant"
            elif topic in {"complaint", "human_handoff"}:
                text = "我会把需要确认的问题转给商家，由商家在这里回复你。" if needs_merchant else "我先帮你说明，目前不会通知商家。"
                status = "needs_merchant" if needs_merchant else "answered"
            else:
                text, status = "我可以帮你了解商品、价格、订单和售后。你想先了解哪一项？", "clarifying"
            if topic not in failed_topics:
                ids = [e["evidence_id"] for e in evidence if e.get("kind") == kind and e.get("field") in fields
                       and (kind!='knowledge' or topic in e.get('policy_intents',[topic]))]
            claims = [{'field':e['field'],'value':e.get('value'),'kind':e['kind'],
                       'evidence_id':e['evidence_id'],'authority':e.get('scope')}
                      for e in evidence if e['evidence_id'] in ids and e['kind'] != 'knowledge']
            sections.append({"intent": topic, "text": text, "evidence_ids": ids, "status": status, 'claims':claims})
        if needs_merchant and not any("商家在这里回复" in s["text"] for s in sections):
            sections.append({"intent": "merchant_handoff", "text": "我会把需要确认的问题转给商家，由商家在这里回复你。",
                             "evidence_ids": [], "status": "needs_merchant"})
        return sections


def validate_sections(sections: list[dict], evidence: list[dict]) -> dict:
    known = {e["evidence_id"] for e in evidence}
    by_id = {e['evidence_id']:e for e in evidence}
    required = {'inventory_query':{'catalog_status'},'sku_query':{'title','description','category'},
                'price_breakdown':{'display_price','paid_amount','discount_amount'},
                'order_query':{'fulfillment_status'},'delivery_status':{'fulfillment_status'},
                'delivery_exception':{'fulfillment_status'},'refund_policy':{'policy'},
                'cancel_policy':{'policy'},'invoice_query':{'policy'}}
    bindings_valid = all(set(s["evidence_ids"]).issubset(known) for s in sections)
    factual = [s for s in sections if s["status"] == "answered" and s["intent"] not in {"human_handoff", "complaint", "merchant_handoff", "other"}]
    field_kinds={'catalog_status':'product','title':'product','description':'product','category':'product',
                 'display_price':'product','paid_amount':'purchase','discount_amount':'purchase',
                 'fulfillment_status':'purchase','policy':'knowledge'}
    def supports_topic(e, topic):
        field=e.get('field')
        return (field in required.get(topic,set()) and e.get('kind')==field_kinds[field]
                and (e['kind']!='knowledge' or topic in e.get('policy_intents',[topic])))
    covered = sum(any(supports_topic(by_id.get(k,{}),s['intent']) for k in s['evidence_ids']) for s in factual)
    claims_valid = all(c.get('evidence_id') in s['evidence_ids'] and
                       all(c.get(k) == by_id.get(c.get('evidence_id'),{}).get(k) for k in ('field','value','kind'))
                       for s in sections for c in s.get('claims',[]))
    return {"bindings_valid": bindings_valid, "factual_sections": len(factual),
            'claims_valid':claims_valid, 'claim_count':sum(len(s.get('claims',[])) for s in sections),
            "covered_sections": covered, "coverage": round(covered / len(factual), 4) if factual else 1.0,
            "passed": bindings_valid and claims_valid and covered == len(factual)}
