"""Transparent DeepSeek Flash cost estimate, not a provider invoice."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def estimate_deepseek_flash_cost(*, input_tokens: int, output_tokens: int,
                                 cache_hit_tokens: int, cache_miss_tokens: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    peak = now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)
    hit_rate, miss_rate, output_rate = ((0.006, 0.3, 1.2) if peak else (0.003, 0.15, 0.6))
    if cache_hit_tokens + cache_miss_tokens == 0:
        cache_miss_tokens = input_tokens
        basis = "all_input_assumed_cache_miss"
    else:
        basis = "provider_reported_cache_split"
    value = (cache_hit_tokens * hit_rate + cache_miss_tokens * miss_rate + output_tokens * output_rate) / 1_000_000
    return {"currency": "USD", "estimated_value": round(value, 8),
            "tariff": "peak" if peak else "off_peak", "basis": basis,
            "pricing_as_of": "2026-09-30",
            "source": "https://api-docs.deepseek.com/quick_start/pricing/"}
