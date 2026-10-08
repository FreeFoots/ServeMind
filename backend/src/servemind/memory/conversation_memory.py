from __future__ import annotations

from servemind.core.domain_models import Session


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, conversation_id: str, *, channel: str, locale: str, actor_type: str = "buyer", user_id: str | None = None) -> Session:
        session = Session(conversation_id=conversation_id, channel=channel, locale=locale, actor_type=actor_type, user_id=user_id)
        self._sessions[conversation_id] = session
        return session

    def get(self, conversation_id: str) -> Session | None:
        return self._sessions.get(conversation_id)

    def ensure(self, conversation_id: str) -> Session:
        return self.get(conversation_id) or self.create(conversation_id, channel="web", locale="zh-CN")

    def context(self, conversation_id: str) -> dict:
        session = self.get(conversation_id)
        if not session:
            return {}
        return {
            "working": dict(session.working_memory),
            "episodic": list(session.episodic_memory[-6:]),
            # 仅保留受控字段，不把完整用户画像送入回答上下文。
            "profile": {key: value for key, value in session.profile_memory.items() if key in {"actor_type", "user_id"}},
        }

    def record_turn(self, conversation_id: str, *, message: str, answer: str, intent: str, evidence_ids: list[str]) -> None:
        session = self.ensure(conversation_id)
        session.episodic_memory.append({"message": message, "answer": answer, "intent": intent, "evidence_ids": list(evidence_ids)})
        session.working_memory.update({"last_intent": intent, "last_evidence_ids": list(evidence_ids)})
