from fastapi import APIRouter, Depends, HTTPException, Request

from servemind.api.legacy import legacy_disabled
from servemind.api.schemas import HandoffRequest

router = APIRouter(prefix="/sessions", tags=["handoffs"], dependencies=[Depends(legacy_disabled)])


@router.post("/{conversation_id}/handoffs")
def create_handoff(conversation_id: str, payload: HandoffRequest, request: Request):
    runtime = request.app.state.runtime
    if not runtime.get_session(conversation_id):
        raise HTTPException(status_code=404, detail="session not found")
    return runtime.outbox.enqueue(conversation_id=conversation_id, reason=payload.reason,
                                  summary=payload.summary, consent=payload.consent)
