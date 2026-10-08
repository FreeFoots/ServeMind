from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    display_label: str
    retrieved_at: str
    facts: list[dict[str, Any]]
    source: str
    scope: str = "business_fact"
    version: str = ""


@dataclass
class Session:
    conversation_id: str
    channel: str = "web"
    locale: str = "zh-CN"
    messages: list[dict[str, str]] = field(default_factory=list)
    current_focus: dict[str, str] | None = None
    user_id: str | None = None
    actor_type: str = "buyer"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    pending_slots: list[str] = field(default_factory=list)
    # 三层记忆：当前工作状态、会话事件、受控用户画像。
    working_memory: dict[str, Any] = field(default_factory=dict)
    episodic_memory: list[dict[str, Any]] = field(default_factory=list)
    profile_memory: dict[str, Any] = field(default_factory=dict)

    def recent_context(self, limit: int = 6) -> list[dict[str, str]]:
        return self.messages[-max(1, limit):]
