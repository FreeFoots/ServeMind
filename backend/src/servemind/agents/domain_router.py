"""Deterministic main/support routing over the existing commerce role scopes."""
from __future__ import annotations

from dataclasses import dataclass

from servemind.agents.task_graph import ROLE_FOR_INTENT, SIGNALS, TaskGraph, TaskNode, active_escalation
from servemind.core.intent_recognizer import Intent, IntentResult
from servemind.mcp.commerce_tools import commerce_graph


@dataclass(frozen=True)
class RoutingDecision:
    primary: str
    supporting: tuple[str, ...]
    scores: dict[str, float]
    reason: str
    confidence: float
    clarification: bool = False

    @property
    def roles(self) -> tuple[str, ...]:
        return (self.primary, *self.supporting)

    def as_dict(self) -> dict:
        return {"mode": "domain_scored_primary_supporting", "primary": self.primary,
                "supporting": list(self.supporting), "scores": dict(self.scores),
                "reason": self.reason, "confidence": self.confidence,
                "multi_agent": bool(self.supporting), "clarification": self.clarification,
                "score_is_probability": False}


class CommerceDomainRouter:
    def route(self, message: str, intent: IntentResult, topics: tuple[str, ...], *,
              available_roles: set[str]) -> RoutingDecision:
        # Only recognized/verified topics can introduce a domain. Raw negated
        # keywords must not resurrect a task deliberately excluded by the LLM.
        topic_roles = {t: ROLE_FOR_INTENT.get(Intent(t), "general") for t in topics}
        represented = set(topic_roles.values())
        missing = represented - available_roles
        if missing:
            raise ValueError("commerce_role_unavailable")
        clarify = (intent.intent == Intent.OTHER and intent.confidence < .5
                   and set(topics).issubset({"other", "clarification"}))
        scores = {role: (.1 if role == "general" else 0.0)
                  for role in sorted(available_roles)}
        primary_role = ROLE_FOR_INTENT.get(intent.intent, "general")
        for role in represented:
            scores[role] += .75 if role == primary_role else .55
            markers = {marker for topic, topic_role in topic_roles.items() if topic_role == role
                       for marker in SIGNALS.get(Intent(topic), ())}
            scores[role] += min(.45, sum(marker in message.lower() for marker in markers) * .18)
        # Entities are recognition hints, never authority to read another order.
        if "order" in represented and intent.entities.get("order_id"):
            scores["order"] += .1
        if "billing" in represented and intent.entities.get("amount"):
            scores["billing"] += .15
        scores = {role: round(score, 3) for role, score in scores.items()}
        ordered = sorted(represented, key=lambda role: (-scores[role], role))
        primary = "general" if clarify else (ordered[0] if ordered else "general")
        # Match EchoMind's escalation-first route while retaining independently
        # answerable requested topics under the existing private-chat boundary.
        escalation = ("escalation" in represented and
                      active_escalation(message, tuple(t for t in topics
                                                      if topic_roles[t] == "escalation")))
        if escalation:
            primary = "escalation"
        supporting = tuple(role for role in ordered if role != primary and
                           scores[role] >= .45 and scores[role] >= scores[primary] * .55)
        # Explicit multi-domain questions cannot lose a requested topic solely
        # because the relative-score threshold was exceeded by another domain.
        retained = tuple(role for role in ordered if role != primary and role not in supporting)
        supporting += retained
        reason = (f"intent={intent.intent.value}; primary={primary}; "
                  f"supporting={','.join(supporting) or 'none'}; "
                  f"scores={','.join(f'{role}:{scores[role]:.3f}' for role in ordered)}")
        if escalation:
            reason = "explicit_escalation_priority; " + reason
        if retained:
            reason += "; retained_requested_domains=" + ",".join(retained)
        if clarify:
            reason = "low_confidence_other_clarification; " + reason
        return RoutingDecision(primary, supporting, scores, reason,
                               round(min(scores.get(primary, 0), 1), 3), clarify)

    def tasks(self, decision: RoutingDecision, topics: tuple[str, ...], *,
              product_id: str, message: str) -> TaskGraph:
        if decision.clarification:
            return TaskGraph(("clarification",), (), deadline_seconds=60)
        base = commerce_graph(topics, product_id, message)
        nodes = list(base.nodes)
        for topic in topics:
            if not any(node.intent == topic for node in nodes):
                role = ROLE_FOR_INTENT.get(Intent(topic), "general")
                nodes.append(TaskNode(f"task-{len(nodes)+1}", topic, role, "", {}))
        priority = {role: index for index, role in enumerate(decision.roles)}
        nodes.sort(key=lambda node: priority[node.role])
        # TaskGraph remains a compatible trace container; commercial dispatch
        # has no dependency edges and runs once per selected role.
        graph = TaskGraph(tuple(node.intent for node in nodes), tuple(nodes), deadline_seconds=60)
        graph.validate()
        return graph
