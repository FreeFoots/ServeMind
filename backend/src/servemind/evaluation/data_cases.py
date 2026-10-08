"""从项目内匿名 CSV 固定抽取的多状态评测样本。"""

DATA_ORDER_CASES = [
    {"id": "data-order-001", "order_id": "d0cf5cc6db", "user_id": "0abe9ef2ce", "sku_id": "581d5b54c1", "merchant": "数码生活馆", "fulfillment_type": "2", "promise": "-", "has_delivery": False, "quantity": "1", "discounted": True},
    {"id": "data-order-002", "order_id": "7444318d01", "user_id": "33a9e56257", "sku_id": "067b673f2b", "merchant": "安心家居店", "fulfillment_type": "1", "promise": "2", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-003", "order_id": "d43a33c38a", "user_id": "4829223b6f", "sku_id": "623d0a582a", "fulfillment_type": "1", "promise": "1", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-004", "order_id": "89286e5fd9", "user_id": "79154d0001", "sku_id": "6717b7c979", "fulfillment_type": "1", "promise": "1", "has_delivery": True, "quantity": "1", "discounted": False},
    {"id": "data-order-005", "order_id": "72585b87a6", "user_id": "d5e8910932", "sku_id": "d829f03a28", "fulfillment_type": "1", "promise": "1", "has_delivery": True, "quantity": "2", "discounted": True},
    {"id": "data-order-006", "order_id": "9c65b6264b", "user_id": "2021a86702", "sku_id": "d3e31fdd6e", "fulfillment_type": "1", "promise": "1", "has_delivery": True, "quantity": "2", "discounted": False},
    {"id": "data-order-007", "order_id": "11bb010740", "user_id": "e5b7eff416", "sku_id": "6e18cb9666", "fulfillment_type": "1", "promise": "3", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-008", "order_id": "ebfaf945eb", "user_id": "088b19b63b", "sku_id": "c76a9ca5a8", "fulfillment_type": "1", "promise": "4", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-009", "order_id": "319e4c6a41", "user_id": "2e9a1761a2", "sku_id": "b85cab3552", "fulfillment_type": "2", "promise": "4", "has_delivery": True, "quantity": "2", "discounted": True},
    {"id": "data-order-010", "order_id": "6762f1214d", "user_id": "0be89ca802", "sku_id": "867b4f2363", "fulfillment_type": "2", "promise": "-", "has_delivery": False, "quantity": "1", "discounted": False},
    {"id": "data-order-011", "order_id": "9e836f588e", "user_id": "5b41dcb3bf", "sku_id": "9c71a03bd3", "fulfillment_type": "2", "promise": "2", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-012", "order_id": "8c1c9037d4", "user_id": "2bda9508a0", "sku_id": "422e6bcde4", "fulfillment_type": "2", "promise": "2", "has_delivery": True, "quantity": "1", "discounted": False},
    {"id": "data-order-013", "order_id": "fa2ef930c2", "user_id": "85f4753169", "sku_id": "14e1926805", "fulfillment_type": "2", "promise": "4", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-014", "order_id": "4d91f1ff1a", "user_id": "3f399e9323", "sku_id": "d3e31fdd6e", "fulfillment_type": "1", "promise": "2", "has_delivery": True, "quantity": "2", "discounted": False},
    {"id": "data-order-015", "order_id": "2aef071560", "user_id": "7a6489d562", "sku_id": "945094e307", "fulfillment_type": "1", "promise": "2", "has_delivery": True, "quantity": "2", "discounted": True},
    {"id": "data-order-016", "order_id": "97fb9e470b", "user_id": "fcb3481cef", "sku_id": "d7b669cfe3", "fulfillment_type": "1", "promise": "5", "has_delivery": True, "quantity": "1", "discounted": True},
    {"id": "data-order-017", "order_id": "a14af46828", "user_id": "22a328a1c1", "sku_id": "f52ec33325", "fulfillment_type": "1", "promise": "3", "has_delivery": False, "quantity": "1", "discounted": True},
    {"id": "data-order-018", "order_id": "413bf6aa13", "user_id": "282d5a34d5", "sku_id": "05db087f6b", "fulfillment_type": "2", "promise": "3", "has_delivery": False, "quantity": "1", "discounted": False},
    {"id": "data-order-019", "order_id": "d7a0519669", "user_id": "209f80ca07", "sku_id": "84cf7e36da", "fulfillment_type": "2", "promise": "-", "has_delivery": False, "quantity": "2", "discounted": False},
    {"id": "data-order-020", "order_id": "307102357a", "user_id": "47c64c5bc7", "sku_id": "fe4a7861d8", "fulfillment_type": "2", "promise": "5", "has_delivery": True, "quantity": "1", "discounted": True},
]
