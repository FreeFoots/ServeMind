from __future__ import annotations

import os

from servemind.config import settings as _settings  # Load local configuration and aliases first.

MODEL_NAME = os.getenv("SERVEMIND_MODEL", "deepseek-flash")
MODEL_VERSION = "DeepSeek-V4.1-Flash"
MODEL_PROVIDER = os.getenv("SERVEMIND_MODEL_PROVIDER", "deepseek")
MODEL_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def model_metadata() -> dict[str, str | bool]:
    return {
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION if MODEL_NAME == "deepseek-flash" else "provider_defined",
        "base_url": MODEL_BASE_URL,
        "configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "runtime_mode": "deepseek_provider" if os.getenv("DEEPSEEK_API_KEY") else "deterministic_local",
    }
