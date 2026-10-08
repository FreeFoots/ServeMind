from __future__ import annotations


class ResponseGrounder:
    def answer(self, *, escalated: bool, evidence: list[dict] | None = None) -> str:
        if escalated:
            return "当前数据可以确认配送节点，但无法确认签收人或投递位置。我已整理已核验信息，建议转人工进一步核验。"
        items = evidence or []
        order_facts = {fact["field"]: fact["value"] for item in items if item.get("kind") == "order" for fact in item.get("facts", [])}
        delivery_facts = {fact["field"]: fact["value"] for item in items if item.get("kind") == "delivery" for fact in item.get("facts", [])}
        if delivery_facts:
            return f"我核对到该订单包含 {order_facts.get('sku_count', '当前记录')} 个 SKU，由 {delivery_facts.get('package_count', '当前记录')} 个包裹配送。记录显示包裹于 {delivery_facts.get('ship_out_time')} 出库、{delivery_facts.get('arr_station_time')} 到达配送站，并于 {delivery_facts.get('arr_time')} 完成送达。"
        return f"我核对到该订单包含 {order_facts.get('sku_count', '当前记录')} 个 SKU，但当前数据没有对应配送记录，无法据此判断是否已发货。"
