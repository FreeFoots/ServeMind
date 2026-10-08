from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


_EXECUTOR = ThreadPoolExecutor(max_workers=12, thread_name_prefix="servemind-tool")
_SLOTS = threading.BoundedSemaphore(24)


class CircuitBreaker:
    """Operational failures only; permission denials cannot poison a tool."""
    def __init__(self, threshold: int = 3, recovery_seconds: float = 30) -> None:
        self.threshold, self.recovery_seconds = threshold, recovery_seconds
        self.failures = 0
        self.opened_at: float | None = None
        self.probing = False
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at < self.recovery_seconds or self.probing:
                return False
            self.probing = True
            return True

    def record(self, success: bool) -> None:
        with self.lock:
            self.probing = False
            if success:
                self.failures, self.opened_at = 0, None
            else:
                self.failures += 1
                if self.failures >= self.threshold:
                    self.opened_at = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": "half_open" if self.probing else "open" if self.opened_at is not None else "closed",
                    "consecutive_failures": self.failures}


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    tool_name: str = ""
    error: str | None = None
    degraded: bool = False
    latency_ms: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[dict[str, Any], dict[str, Any]], Any]
    required: tuple[str, ...] = ()
    read_only: bool = True
    input_model: type[BaseModel] | None = None
    required_permissions: tuple[str, ...] = ()
    timeout_seconds: float = 5.0
    max_retries: int = 0
    retryable_exceptions: tuple[type[Exception], ...] = ()
    stats: dict[str, float] = field(default_factory=lambda: {"total": 0, "success": 0, "failed": 0, 'total_latency_ms':0})
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class ToolManager:
    """Deterministic tool registry with schema-like validation and audit traces."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.traces: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _count(self, tool: Tool, key: str) -> None:
        with self._lock:
            tool.stats[key] += 1

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description,
                 "required": list(t.required), "read_only": t.read_only,
                 "input_schema": t.input_model.model_json_schema() if t.input_model else
                 {"type": "object", "required": list(t.required)},
                 "required_permissions": list(t.required_permissions),
                 "timeout_seconds": t.timeout_seconds, "max_retries": t.max_retries}
                for t in self.tools.values()]

    def call(self, name: str, params: dict[str, Any], context: dict[str, Any] | None = None) -> ToolResult:
        started = time.perf_counter()
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(False, tool_name=name, error="tool_not_found")
        self._count(tool, "total")
        context = context or {}
        allowed = set(context.get("allowed_tools") or ())
        granted = set(context.get("permissions") or ())
        if (name not in allowed or
            not set(tool.required_permissions).issubset(granted) or
            (not tool.read_only and not context.get("allow_writes", False))):
            self._count(tool, "failed")
            result = ToolResult(False, tool_name=name, error="forbidden:tool_scope")
        else:
            missing = [field for field in tool.required if not params.get(field)]
            if missing:
                self._count(tool, "failed")
                result = ToolResult(False, tool_name=name, error=f"missing_fields:{','.join(missing)}")
            else:
                try:
                    validated = (tool.input_model.model_validate(params).model_dump(exclude_none=True)
                                 if tool.input_model else params)
                except ValidationError:
                    self._count(tool, "failed")
                    result = ToolResult(False, tool_name=name, error="invalid_arguments")
                else:
                    if not tool.breaker.allow():
                        result = ToolResult(False, tool_name=tool.name, error="tool_circuit_open", degraded=True)
                    else:
                        result = self._execute(tool, validated, context)
                        if result.error not in {"PermissionError", "tool_overloaded"}:
                            tool.breaker.record(result.success)
                        else:
                            # Finish a probe without attributing a scope error or
                            # host capacity limit to the downstream service.
                            with tool.breaker.lock:
                                tool.breaker.probing = False
                    if result.success:
                        self._count(tool, "success")
                    else:
                        self._count(tool, "failed")
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        with self._lock:
            tool.stats['total_latency_ms'] += result.latency_ms
            self.traces.append({"name": name, "role": context.get("agent_role"), "param_keys": sorted(params), "success": result.success,
                                "error": result.error, "latency_ms": result.latency_ms, "degraded": result.degraded})
            if len(self.traces) > 1000:
                del self.traces[:len(self.traces) - 1000]
        return result

    @staticmethod
    def _execute(tool: Tool, params: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        attempts = max(0, tool.max_retries) + 1 if tool.read_only else 1
        for attempt in range(attempts):
            if not _SLOTS.acquire(timeout=0.05):
                return ToolResult(False, tool_name=tool.name, error="tool_overloaded", degraded=True)
            future = _EXECUTOR.submit(tool.handler, params, context)
            future.add_done_callback(lambda _future: _SLOTS.release())
            try:
                return ToolResult(True, data=future.result(timeout=tool.timeout_seconds),
                                  tool_name=tool.name, degraded=attempt > 0)
            except FutureTimeout:
                # Do not retry a still-running handler: that could double-load
                # PostgreSQL and race a response after its request timed out.
                future.cancel()
                return ToolResult(False, tool_name=tool.name, error="tool_timeout", degraded=True)
            except Exception as exc:
                if (attempt + 1 < attempts and tool.retryable_exceptions and
                    isinstance(exc, tool.retryable_exceptions)):
                    time.sleep(0.05 * (attempt + 1))
                    continue
                return ToolResult(False, tool_name=tool.name,
                                  error=type(exc).__name__, degraded=attempt > 0)
        return ToolResult(False, tool_name=tool.name, error="tool_retry_exhausted", degraded=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {name: {**tool.stats, 'avg_latency_ms':round(tool.stats['total_latency_ms']/tool.stats['total'],2) if tool.stats['total'] else 0,
                "circuit": tool.breaker.snapshot(), "success_rate": round(tool.stats["success"] / tool.stats["total"], 3) if tool.stats["total"] else 1.0} for name, tool in self.tools.items()}
