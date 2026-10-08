from __future__ import annotations

import time
import uuid
import threading
from collections import deque
from collections import defaultdict
from statistics import quantiles


class Trace:
    def __init__(self) -> None:
        self.request_id = f"req_{uuid.uuid4().hex[:8]}"
        self.started = time.perf_counter()

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)


class OnlineMonitor:
    """在线监控与路由降权：按 Agent/意图维护成功率、延迟和惩罚。"""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.request_events: list[dict] = []
        self.penalties: dict[str, float] = defaultdict(float)

    def record(self, *, key: str, success: bool, latency_ms: float, status: str) -> None:
        self.events.append({"key": key, "success": success, "latency_ms": latency_ms, "status": status})
        if not success or status in {"failed", "unsupported"}:
            self.penalties[key] = min(0.9, self.penalties[key] + 0.05)
        else:
            self.penalties[key] = max(0.0, self.penalties[key] - 0.01)

    def record_request(self, *, latency_ms: float, status: str, route_success: bool,
                       evidence_count: int, grounded: bool, tools: list[dict],
                       handoff_recommended: bool = False, handoff_expected: bool = False,
                       input_chars: int = 0, output_chars: int = 0,
                       unauthorized_blocked: bool = False, cache_hit: bool = False,
                       rule_conflict: bool = False, rule_expired: bool = False,
                       grounding_required: bool = False) -> None:
        """Record request-level SLO signals in a provider-neutral form.

        Token counts are estimates in deterministic-local mode; a provider adapter can
        replace them with usage returned by the model without changing the report shape.
        """
        input_tokens = max(1, round(input_chars / 4)) if input_chars else 0
        output_tokens = max(1, round(output_chars / 4)) if output_chars else 0
        self.request_events.append({
            "latency_ms": round(latency_ms, 2), "status": status,
            "route_success": route_success, "evidence_count": evidence_count,
            "grounded": grounded, "tool_calls": len(tools),
            "grounding_required": grounding_required,
            "tool_successes": sum(item.get("status") == "ok" for item in tools),
            "tool_errors": sum(item.get("status") != "ok" for item in tools),
            "handoff_recommended": handoff_recommended, "handoff_expected": handoff_expected,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "unauthorized_blocked": unauthorized_blocked, "cache_hit": cache_hit,
            "rule_conflict": rule_conflict, "rule_expired": rule_expired,
        })

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return round(values[0], 2)
        return round(quantiles(values, n=100, method="inclusive")[94], 2)

    def runtime_metrics(self) -> dict:
        events = self.request_events
        tool_calls = sum(item["tool_calls"] for item in events)
        tool_successes = sum(item["tool_successes"] for item in events)
        tool_errors = sum(item["tool_errors"] for item in events)
        escalated = sum(item["status"] == "escalated" for item in events)
        total_tokens = sum(item["input_tokens"] + item["output_tokens"] for item in events)
        # No provider key/pricing is assumed in deterministic-local mode. This is an
        # explicit zero estimate, rather than pretending local execution incurred API cost.
        return {
            "requests": len(events),
            "tool_success_rate": self._rate(tool_successes, tool_calls),
            "tool_error_rate": self._rate(tool_errors, tool_calls),
            "agent_route_success_rate": self._rate(sum(item["route_success"] for item in events), len(events)),
            "unauthorized_block_count": sum(item["unauthorized_blocked"] for item in events),
            "handoff": {
                "recommended": escalated,
                "correct_rate_proxy": self._rate(sum(item["handoff_recommended"] for item in events), escalated),
            },
            "evidence_coverage_rate": self._rate(sum(item["evidence_count"] > 0 for item in events), len(events)),
            "grounding_failure_rate": self._rate(sum(item.get("grounding_required", False) and not item["grounded"] for item in events),
                                                 sum(item.get("grounding_required", False) for item in events)),
            "p95_latency_ms": self._p95([item["latency_ms"] for item in events]),
            "token_usage": {"estimated_input": sum(item["input_tokens"] for item in events), "estimated_output": sum(item["output_tokens"] for item in events), "estimated_total": total_tokens},
            "estimated_cost": {"currency": "USD", "value": 0.0, "basis": "deterministic_local_no_provider_billing"},
            "cache_hit_rate": self._rate(sum(item["cache_hit"] for item in events), len(events)),
            "rule_conflict_count": sum(item["rule_conflict"] for item in events),
            "rule_expired_count": sum(item["rule_expired"] for item in events),
        }

    def summary(self) -> dict:
        recent = self.events[-100:]
        return {"events": len(recent), "penalties": dict(self.penalties), "recent": recent[-20:], "runtime_metrics": self.runtime_metrics()}


class CommerceMonitor:
    """Bounded buyer-path traces; provider usage and estimates are kept separate."""

    def __init__(self, capacity: int = 2000) -> None:
        self._events = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.observer = None

    def record_commerce(self, *, metadata: dict, latency_ms: float) -> None:
        event = {"request_id": metadata["request_id"], "latency_ms": latency_ms,
                 "status": metadata["status"], "tools": metadata["tools_used"],
                 "graph": metadata["task_graph"], "grounding": metadata["grounding"],
                 "handoff": metadata["needs_merchant"], "model_use": metadata["model_use"],
                 "cache_hit": any(e.get("cache_hit") for e in metadata["evidence"]),
                 "rule_conflicts": len(metadata["skill_conflicts"]),
                 "rule_expired": len(metadata["skill_expired"]),
                 "agent_mode": metadata.get("agent_use"),
                 "agent_contributions": [{k: c.get(k) for k in ("agent", "instance", "status")}
                                         for c in metadata.get("agent_contributions", [])],
                 "planner_mode": metadata.get("intent_three_way", {}).get("mode"),
                 "model_calls": len(metadata.get("agent_runtime", {}).get("calls", []))}
        with self._lock:
            self._events.append(event)
        if self.observer:
            try:
                self.observer(event)
            except Exception:
                pass

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)
        tools = [tool for event in events for tool in event["tools"]]
        usage = [e["model_use"] for e in events]
        model_events = [e for e in events if e["agent_mode"] in {
            "model_intent_routing_and_specialist_tool_loops", "model_planning_and_specialist_tool_loops"}]
        contributions = [c for e in model_events for c in e["agent_contributions"]]
        return {"requests": len(events), "window": "last_2000_requests_process_local",
                "model_agent_requests": len(model_events),
                "specialist_model_success_rate": OnlineMonitor._rate(sum(c["status"] == "model_analyzed" for c in contributions), len(contributions)),
                "specialist_fallback_count": sum(c["status"] == "deterministic_fallback" for c in contributions),
                "supervisor_fallback_count": sum(e["planner_mode"] == "supervisor_failed_rule_plan_fallback" for e in model_events),
                "intent_recognition_fallback_count": sum(e["planner_mode"] == "intent_recognition_failed_rule_fallback" for e in model_events),
                "model_calls": sum(e["model_calls"] for e in model_events),
                "tool_success_rate": OnlineMonitor._rate(sum(t["status"] == "ok" for t in tools), len(tools)),
                "tool_errors": sum(t["status"] != "ok" for t in tools),
                "unauthorized_block_count": sum(str(t.get("error", "")).startswith("forbidden:")
                                                or t.get("error") == "PermissionError" for t in tools),
                "p95_latency_ms": OnlineMonitor._p95([e["latency_ms"] for e in events]),
                "task_completion_rate": OnlineMonitor._rate(sum(e["graph"]["completed"] for e in events), sum(e["graph"]["total"] for e in events)),
                "grounding_failure_count": sum(not e["grounding"]["passed"] for e in events),
                "handoff_count": sum(e["handoff"] for e in events),
                "cache_hit_rate": OnlineMonitor._rate(sum(e["cache_hit"] for e in events), len(events)),
                "provider_usage": {"input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                                   "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
                                   "cache_hit_tokens": sum(u.get("prompt_cache_hit_tokens", 0) for u in usage),
                                   "output_rejected": sum(bool(u.get("output_rejected")) for u in usage)},
                "estimated_cost_usd": round(sum((u.get("estimated_cost") or {}).get("estimated_value", 0) for u in usage), 8),
                "rule_conflict_count": sum(e["rule_conflicts"] for e in events),
                "rule_expired_count": sum(e["rule_expired"] for e in events),
                "recent": events[-20:]}
