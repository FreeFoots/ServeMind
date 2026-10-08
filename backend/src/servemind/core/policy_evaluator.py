from __future__ import annotations

from servemind.config.policy_registry import policy_metadata


def evaluate(*, escalated: bool, status: str | None = None) -> dict[str, str]:
    decision = "policy_review" if escalated else (status or "policy_match")
    return {**policy_metadata(), "decision": decision}
