from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import psycopg
from psycopg.rows import dict_row

from servemind.core.domain_models import EvidenceItem
from servemind.mcp.customer_data_provider import DATA_VERSION, MSOMCustomerDataProvider


class PostgresMSOMProvider(MSOMCustomerDataProvider):
    """Read-only indexed projection of the seven imported MSOM tables.

    This provider is deliberately separate from the self-registered commerce
    accounts: anonymous historical user_IDs are not authenticated buyers.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind")
        self.orders: dict[str, list[dict[str, str]]] = {}
        self.deliveries: dict[str, list[dict[str, str]]] = {}
        self.skus: dict[str, dict[str, str]] = {}
        self.inventory: set[tuple[str, str, str]] = set()
        self.users: dict[str, dict[str, str]] = {}
        self.clicks: dict[str, list[dict[str, str]]] = {}
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) AS n FROM msom.import_batches")
                if cursor.fetchone()["n"] < 7:
                    raise RuntimeError("msom_import_incomplete")

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)

    def _rows(self, query: str, params: tuple[Any, ...]) -> list[dict[str, str]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

    def _ensure_order(self, order_id: str) -> None:
        if order_id in self.orders:
            return
        self.orders[order_id] = self._rows(
            """SELECT order_id AS "order_ID", user_id AS "user_ID", sku_id AS "sku_ID",
                      order_date, order_time, quantity, type, promise,
                      original_unit_price, final_unit_price, direct_discount_per_unit,
                      quantity_discount_per_unit, bundle_discount_per_unit,
                      coupon_discount_per_unit, gift_item, dc_ori, dc_des
               FROM msom.orders WHERE order_id = %s""", (order_id,),
        )
        self.deliveries[order_id] = self._rows(
            """SELECT package_id AS "package_ID", order_id AS "order_ID", type,
                      ship_out_time, arr_station_time, arr_time
               FROM msom.deliveries WHERE order_id = %s""", (order_id,),
        )

    def list_evidence(self, order_id: str = "81a6fa818d") -> list[EvidenceItem]:
        self._ensure_order(order_id)
        return [replace(item, source=f"database/msom.{'deliveries' if item.kind == 'delivery' else 'orders'}")
                for item in super().list_evidence(order_id)]

    def current_focus(self, order_id: str = "81a6fa818d") -> dict[str, str] | None:
        self._ensure_order(order_id)
        return super().current_focus(order_id)

    def order_facts(self, order_id: str, user_id: str | None = None) -> list[EvidenceItem]:
        self._ensure_order(order_id)
        return super().order_facts(order_id, user_id)

    def price_breakdown(self, order_id: str, user_id: str | None = None) -> list[EvidenceItem]:
        self._ensure_order(order_id)
        return [replace(item, source="database/msom.orders")
                for item in super().price_breakdown(order_id, user_id)]

    def sku_facts(self, sku_id: str) -> list[EvidenceItem]:
        if sku_id not in self.skus:
            rows = self._rows(
                """SELECT sku_id AS "sku_ID", type, brand_id AS "brand_ID",
                          attribute1, attribute2, activate_date, deactivate_date
                   FROM msom.skus WHERE sku_id = %s LIMIT 1""", (sku_id,),
            )
            self.skus[sku_id] = rows[0] if rows else {}
        return [replace(item, source="database/msom.skus") for item in super().sku_facts(sku_id)]

    def inventory_facts(self, sku_id: str, dc_id: str | None = None,
                        date: str | None = None) -> list[EvidenceItem]:
        query = "SELECT dc_id, sku_id, date FROM msom.inventory WHERE sku_id = %s"
        params: list[str] = [sku_id]
        if dc_id is not None:
            query += " AND dc_id = %s"
            params.append(dc_id)
        if date is not None:
            query += " AND date = %s"
            params.append(date)
        query += " ORDER BY date DESC LIMIT 100"
        rows = self._rows(query, tuple(params))
        facts = [{"field": "record", "value": row, "confidence": 1.0} for row in rows]
        return [EvidenceItem("ev_inventory_01", "inventory", "库存记录", date or "", facts,
                             "database/msom.inventory", version=DATA_VERSION)] if facts else []

    def browsing_facts(self, user_id: str, limit: int = 20) -> list[EvidenceItem]:
        if not user_id:
            return []
        rows = self._rows(
            """SELECT sku_id, user_id, request_time, channel FROM msom.clicks
               WHERE user_id = %s ORDER BY request_time DESC LIMIT %s""", (user_id, min(limit, 20)),
        )
        facts = [{"field": "click", "value": {"sku_id": row["sku_id"],
                  "request_time": row["request_time"], "channel": row["channel"]},
                  "confidence": 1.0} for row in rows]
        return [EvidenceItem("ev_click_01", "browsing", "浏览记录", rows[0]["request_time"],
                             facts, "database/msom.clicks", version=DATA_VERSION)] if facts else []
