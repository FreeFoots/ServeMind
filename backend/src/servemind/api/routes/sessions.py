from fastapi import APIRouter, Depends, HTTPException, Request

from servemind.api.legacy import legacy_disabled
from servemind.api.schemas import MessageRequest, SessionCreate

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(legacy_disabled)])


@router.post("")
def create_session(payload: SessionCreate, request: Request):
    return request.app.state.runtime.create_session(channel=payload.channel, locale=payload.locale, actor_type=payload.actor_type, user_id=payload.user_id)


@router.get("/{conversation_id}")
def get_session(conversation_id: str, request: Request):
    result = request.app.state.runtime.get_session(conversation_id)
    if not result:
        raise HTTPException(status_code=404, detail="session not found")
    return result


@router.get("/{conversation_id}/memory")
def get_memory(conversation_id: str, request: Request):
    if not request.app.state.runtime.get_session(conversation_id):
        raise HTTPException(status_code=404, detail="session not found")
    return request.app.state.runtime.sessions.context(conversation_id)


@router.post("/{conversation_id}/messages")
def send_message(conversation_id: str, payload: MessageRequest, request: Request):
    if not request.app.state.runtime.get_session(conversation_id):
        raise HTTPException(status_code=404, detail="session not found")
    return request.app.state.runtime.respond(conversation_id, payload.message)
