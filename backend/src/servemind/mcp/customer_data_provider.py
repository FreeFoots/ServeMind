from __future__ import annotations

import csv
from typing import Any

from servemind.config.settings import DATA_ROOT, DATA_VERSION, DELIVERY_DATA_PATH, ORDER_DATA_PATH
from servemind.core.domain_models import EvidenceItem

DEFAULT_ORDER_ID = "81a6fa818d"


class MSOMCustomerDataProvider:
    """从项目内 MSOM 表读取订单和配送事实，并按订单号建立内存索引。"""

    def __init__(self) -> None:
        self.orders: dict[str, list[dict[str, str]]] = {}
        self.deliveries: dict[str, list[dict[str, str]]] = {}
        self._load_orders()
        self._load_deliveries()
        self.skus: dict[str, dict[str, str]] = {}
        self.inventory: set[tuple[str, str, str]] = set()
        self.users: dict[str, dict[str, str]] = {}
        self.clicks: dict[str, list[dict[str, str]]] = {}
        self.click_path = DATA_ROOT / "JD_click_data.csv"
        self._load_optional_tables()

    def _load_optional_tables(self) -> None:
        for name, target, key_name in (("JD_sku_data.csv", self.skus, "sku_ID"), ("JD_user_data.csv", self.users, "user_ID")):
            path = DATA_ROOT / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if row.get(key_name):
                            target[row[key_name]] = row
        path = DATA_ROOT / "JD_inventory_data.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.inventory.add((row.get("dc_ID", ""), row.get("sku_ID", ""), row.get("date", "")))

    def _load_orders(self) -> None:
        with ORDER_DATA_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.orders.setdefault(row["order_ID"], []).append(row)

    def _load_deliveries(self) -> None:
        with DELIVERY_DATA_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.deliveries.setdefault(row["order_ID"], []).append(row)

    def list_evidence(self, order_id: str = DEFAULT_ORDER_ID) -> list[EvidenceItem]:
        order_rows = self.orders.get(order_id, [])
        delivery_rows = self.deliveries.get(order_id, [])
        if not order_rows:
            return []

        first = order_rows[0]
        order_time = first["order_time"].replace(".0", "")
        facts: list[dict[str, Any]] = [
            {"field": "order_id", "value": order_id, "confidence": 1.0},
            {"field": "user_id", "value": first["user_ID"], "confidence": 1.0},
            {"field": "sku_count", "value": len(order_rows), "confidence": 1.0},
            {"field": "package_id", "value": delivery_rows[0]["package_ID"] if delivery_rows else "无配送记录", "confidence": 1.0},
            {"field": "fulfillment", "value": "1P · 平台履约" if first["type"] == "1" else "3P · 第三方履约", "confidence": 1.0},
            {"field": "promise_days", "value": first["promise"], "confidence": 1.0},
            {"field": "final_unit_price", "value": first["final_unit_price"], "confidence": 1.0},
        ]
        items = [EvidenceItem(
            evidence_id="ev_order_01", kind="order", display_label="订单事实",
            retrieved_at=order_time, source="data/MSOM_Data_Driven_Challenge_2020/JD_order_data.csv", facts=facts,
        )]
        if delivery_rows:
            delivery = delivery_rows[0]
            items.append(EvidenceItem(
                evidence_id="ev_delivery_01", kind="delivery", display_label="配送时间线",
                retrieved_at=delivery["arr_time"], source="data/MSOM_Data_Driven_Challenge_2020/JD_delivery_data.csv",
                facts=[
                    {"field": "ship_out_time", "value": delivery["ship_out_time"], "confidence": 1.0},
                    {"field": "arr_station_time", "value": delivery["arr_station_time"], "confidence": 1.0},
                    {"field": "arr_time", "value": delivery["arr_time"], "confidence": 1.0},
                    {"field": "package_count", "value": len(delivery_rows), "confidence": 1.0},
                ],
            ))
        return items

    def current_focus(self, order_id: str = DEFAULT_ORDER_ID) -> dict[str, str] | None:
        rows = self.orders.get(order_id, [])
        if not rows:
            return None
        delivery_rows = self.deliveries.get(order_id, [])
        return {"order_id": order_id, "user_id": rows[0]["user_ID"], "package_id": delivery_rows[0]["package_ID"] if delivery_rows else ""}

    def order_facts(self, order_id: str, user_id: str | None = None) -> list[EvidenceItem]:
        rows = self.orders.get(order_id, [])
        if not rows or (user_id and rows[0].get("user_ID") != user_id):
            return []
        return self.list_evidence(order_id)

    def price_breakdown(self, order_id: str, user_id: str | None = None) -> list[EvidenceItem]:
        rows = self.orders.get(order_id, [])
        if not rows or (user_id and rows[0].get("user_ID") != user_id):
            return []
        facts = [{"field": f"line_{i + 1}", "value": {"sku_id": row["sku_ID"], "quantity": row["quantity"], "original_unit_price": row["original_unit_price"], "final_unit_price": row["final_unit_price"], "gift_item": row["gift_item"]}, "confidence": 1.0} for i, row in enumerate(rows)]
        return [EvidenceItem("ev_price_01", "price", "价格与优惠", rows[0]["order_time"].replace(".0", ""), facts, "data/MSOM_Data_Driven_Challenge_2020/JD_order_data.csv", version=DATA_VERSION)]

    def sku_facts(self, sku_id: str) -> list[EvidenceItem]:
        row = self.skus.get(sku_id)
        if not row:
            return []
        facts = [{"field": k, "value": v, "confidence": 1.0} for k, v in row.items() if k != "sku_ID"]
        return [EvidenceItem("ev_sku_01", "sku", "SKU 信息", row.get("activate_date", ""), facts, "data/MSOM_Data_Driven_Challenge_2020/JD_sku_data.csv", version=DATA_VERSION)]

    def inventory_facts(self, sku_id: str, dc_id: str | None = None, date: str | None = None) -> list[EvidenceItem]:
        matches = [(dc, sku, day) for dc, sku, day in self.inventory if sku == sku_id and (not dc_id or dc == dc_id) and (not date or day == date)]
        facts = [{"field": "record", "value": {"dc_id": dc, "sku_id": sku, "date": day}, "confidence": 1.0} for dc, sku, day in sorted(matches)]
        return [EvidenceItem("ev_inventory_01", "inventory", "库存记录", date or "", facts, "data/MSOM_Data_Driven_Challenge_2020/JD_inventory_data.csv", version=DATA_VERSION)] if facts else []

    def browsing_facts(self, user_id: str, limit: int = 20) -> list[EvidenceItem]:
        rows = list(self.clicks.get(user_id, []))
        if not rows and self.click_path.exists():
            with self.click_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("user_ID") == user_id:
                        rows.append(row)
                        if len(rows) >= limit:
                            break
        rows = sorted(rows, key=lambda row: row.get("request_time", ""), reverse=True)[:limit]
        facts = [{"field": "click", "value": {"sku_id": row.get("sku_ID"), "request_time": row.get("request_time"), "channel": row.get("channel")}, "confidence": 1.0} for row in rows]
        return [EvidenceItem("ev_click_01", "browsing", "浏览记录", rows[0].get("request_time", ""), facts, "data/MSOM_Data_Driven_Challenge_2020/JD_click_data.csv", version=DATA_VERSION)] if facts else []


def evidence_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id, "kind": item.kind, "display_label": item.display_label,
        "retrieved_at": item.retrieved_at, "source": item.source, "facts": item.facts, "scope": item.scope, "version": item.version,
    }
