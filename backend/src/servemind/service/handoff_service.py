from __future__ import annotations

import uuid


class HandoffOutbox:
    def enqueue(self, *, conversation_id: str, reason: str, summary: str, consent: bool) -> dict[str, str | bool]:
        return {
            "handoff_id": f"ho_{uuid.uuid4().hex[:8]}", "outbox_id": f"out_{uuid.uuid4().hex[:8]}",
            "conversation_id": conversation_id, "status": "queued_local", "reason": reason,
            "summary": summary, "consent": consent,
        }
