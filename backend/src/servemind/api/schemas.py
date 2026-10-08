from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    channel: str = "web"
    locale: str = "zh-CN"
    actor_type: str = "buyer"
    user_id: str | None = None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    client_message_id: str | None = None
    channel: str = "web"
    locale: str = "zh-CN"


class HandoffRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=3000)
    consent: bool = False
