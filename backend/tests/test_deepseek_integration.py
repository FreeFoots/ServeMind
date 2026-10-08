from __future__ import annotations

from servemind.llm.deepseek_client import Completion, redact_identifiers
from servemind.service.commerce_support import CommerceSupport


class FakeDeepSeek:
    configured = True

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 256) -> Completion:
        self.calls.append({"system": system, "user": user})
        return Completion("我可以先说明一般商品信息；商品尚未核验。", "deepseek-flash", 30, 16)


def test_provider_composes_private_answer_without_raw_identifiers():
    llm = FakeDeepSeek()
    support = CommerceSupport(llm_client=llm)
    answer, metadata = support.reply("你好，订单 81a6fa818d 怎么看", {"provenance": "merchant_declared_unverified"})
    assert answer.startswith("我可以先说明")
    assert metadata["model_use"]["mode"] == "deepseek_provider"
    assert metadata["model_use"]["prompt_tokens"] == 30
    assert "81a6fa818d" not in llm.calls[0]["user"]


def test_provider_never_decides_merchant_handoff():
    llm = FakeDeepSeek()
    support = CommerceSupport(llm_client=llm)
    _, metadata = support.reply("我要退款，请商家处理", {"provenance": "merchant_declared_unverified"})
    assert metadata["needs_merchant"] is True
    assert metadata["model_use"]["mode"] == "deterministic_local"
    assert llm.calls == []


def test_identifier_redaction():
    assert "81a6fa818d" not in redact_identifiers("订单 81a6fa818d")
    assert "13800138000" not in redact_identifiers("电话 13800138000")
