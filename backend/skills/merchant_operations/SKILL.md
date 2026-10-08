---
name: 商家履约与库存支持
audience: merchant
keywords: 商家,店铺,库存,备货,发货量,仓库,SKU,履约
intents: sku_query,inventory_query,delivery_status,order_query
priority: 50
required_permissions: merchant_read
valid_from: 2026-09-23
---

# 商家支持规范

- 先识别订单、SKU、库存、仓库或履约问题。
- 库存工具只能证明指定日期存在记录，不能编造库存数量或实时可用性。
- 匿名 SKU 只能说明数据中的属性，不生成商品真实名称。
- 涉及批量经营决策时返回聚合结果和数据边界，不暴露无关用户字段。
