from __future__ import annotations

import json
import os
from typing import Any

import redis


class RedisWorkingMemory:
    """Disposable, conversation-scoped working state; SQLite messages remain durable."""

    def __init__(self, url: str | None = None, ttl_seconds: int = 86400) -> None:
        self.url = url or os.getenv("SERVEMIND_REDIS_URL", "redis://127.0.0.1:6379/0")
        self.ttl_seconds = ttl_seconds
        self.client = redis.Redis.from_url(self.url, socket_connect_timeout=1, socket_timeout=1)

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"servemind:working:{conversation_id}"

    def get(self, conversation_id: str) -> dict[str, Any]:
        try:
            value = self.client.get(self._key(conversation_id))
            return json.loads(value) if value else {}
        except (redis.RedisError, ValueError):
            return {}

    def put(self, conversation_id: str, state: dict[str, Any]) -> bool:
        try:
            self.client.set(self._key(conversation_id), json.dumps(state, ensure_ascii=False),
                            ex=self.ttl_seconds)
            return True
        except redis.RedisError:
            return False
