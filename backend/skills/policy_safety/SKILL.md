---
name: 电商事实与安全边界
audience: both
keywords: 退款,取消,支付,签收,赔偿,越权,隐私,人工
intents: *
priority: 100
required_permissions: policy_public
valid_from: 2026-09-23
denied_tools: execute_refund,cancel_order,execute_payment
---

# 事实安全边界

- 所有对外结论必须绑定证据或政策版本。
- 缺失、冲突或越权时返回 clarifying、unsupported、forbidden 或 escalated。
- 工具只读执行，模型不能绕过工具声称完成退款、取消或支付操作。
