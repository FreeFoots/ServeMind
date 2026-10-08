from __future__ import annotations

from servemind.evaluation.commerce_cases import PRODUCT
from servemind.service.commerce_store import CommerceStore
from servemind.service.commerce_support import CommerceSupport


def test_catalog_completion_gives_every_listing_owner_name_category_and_price(tmp_path):
    store = CommerceStore(tmp_path / "commerce.sqlite3")
    merchant = store.register(username="merchant-demo", password="test-password-123",
                              display_name="演示店铺", role="merchant")["account"]
    with store._db() as db, db:
        db.execute(
            """INSERT INTO catalog_assignments
               (sku_id, merchant_id, display_title, catalog_status, assignment_type,
                source_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("test-sku-001", merchant["id"], "旧商品名", "在售", "simulated", "test", "now"),
        )
        store._complete_demo_catalog(db)
        store._complete_demo_catalog(db)
        rows = db.execute("SELECT merchant_id, title, category, display_price_cents FROM products").fetchall()
        assert len(rows) == 1
        assert rows[0]["merchant_id"] == merchant["id"]
        assert rows[0]["title"] and rows[0]["category"]
        assert rows[0]["display_price_cents"] > 0


def test_buyer_reply_uses_customer_language_and_respects_handoff():
    support = CommerceSupport(use_llm=False)
    product = {**PRODUCT, "catalog_status": "暂时缺货",
               "display_price": "67.90", "price_basis": "simulated_catalog_price"}
    answer, metadata = support.reply("商品还有货吗，价格是多少", product)
    assert "暂时缺货" in answer and "¥67.90" in answer
    assert all(term not in answer for term in ("历史订单样本", "演示目录状态", "sku_ID", "模型"))
    assert metadata["needs_merchant"] is False

    answer, metadata = support.reply("我不想继续聊机器人，请把库存和价格一起交给店家", product)
    assert metadata["needs_merchant"] is True
    assert "商家" in answer

    answer, metadata = support.reply("如果我想退货，先告诉我规则，暂时别通知店家", product)
    assert metadata["needs_merchant"] is False
    assert "不会通知商家" in answer
