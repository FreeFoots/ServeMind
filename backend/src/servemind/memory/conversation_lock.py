from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

import redis


_RELEASE_IF_OWNER = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class ConversationLock:
    """Redis cross-process lock, with a process-local fallback on outage."""

    def __init__(self, url: str | None = None) -> None:
        self.client = redis.Redis.from_url(
            url or os.getenv("SERVEMIND_REDIS_URL", "redis://127.0.0.1:6379/0"),
            socket_connect_timeout=0.5, socket_timeout=0.5,
        )
        self._guard = threading.Lock()
        self._local: dict[str, threading.Lock] = {}

    @contextmanager
    def hold(self, conversation_id: str, *, wait_seconds: float = 5.0) -> Iterator[str]:
        with self._guard:
            local = self._local.setdefault(conversation_id, threading.Lock())
        if not local.acquire(timeout=wait_seconds):
            raise TimeoutError("conversation_busy")
        key = f"servemind:lock:{conversation_id}"
        token = uuid.uuid4().hex
        acquired_remote = False
        backend = "local_fallback"
        try:
            try:
                deadline = time.monotonic() + wait_seconds
                while time.monotonic() < deadline:
                    if self.client.set(key, token, nx=True, ex=120):
                        acquired_remote = True
                        backend = "redis"
                        break
                    time.sleep(0.05)
                if not acquired_remote:
                    raise TimeoutError("conversation_busy")
            except redis.RedisError:
                # SQLite's unique client_message_id still protects persisted
                # messages; the local lock avoids duplicate provider calls here.
                pass
            yield backend
        finally:
            if acquired_remote:
                try:
                    self.client.eval(_RELEASE_IF_OWNER, 1, key, token)
                except redis.RedisError:
                    pass
            local.release()
