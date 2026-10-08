from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import redis


class RedisQueryCache:
    """Versioned cache for public knowledge results; raw queries are not keys."""

    def __init__(self, url: str | None = None, ttl_seconds: int = 3600) -> None:
        self.client = redis.Redis.from_url(
            url or os.getenv("SERVEMIND_REDIS_URL", "redis://127.0.0.1:6379/0"),
            socket_connect_timeout=0.5, socket_timeout=0.5,
        )
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(query: str, *, top_k: int, version: str, backend: str) -> str:
        digest = hashlib.sha256(
            json.dumps([query.strip().lower(), top_k, version, backend], ensure_ascii=False).encode()
        ).hexdigest()
        return f"servemind:knowledge:{digest}"

    def get(self, query: str, *, top_k: int, version: str, backend: str) -> list[dict[str, Any]] | None:
        try:
            value = self.client.get(self._key(query, top_k=top_k, version=version, backend=backend))
            return json.loads(value) if value else None
        except (redis.RedisError, ValueError):
            return None

    def put(self, query: str, items: list[dict[str, Any]], *, top_k: int,
            version: str, backend: str) -> bool:
        try:
            self.client.set(
                self._key(query, top_k=top_k, version=version, backend=backend),
                json.dumps(items, ensure_ascii=False), ex=self.ttl_seconds,
            )
            return True
        except redis.RedisError:
            return False
