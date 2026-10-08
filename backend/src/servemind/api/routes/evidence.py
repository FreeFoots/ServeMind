from fastapi import APIRouter, Depends, HTTPException, Request

from servemind.api.legacy import legacy_disabled

router = APIRouter(prefix="/sessions", tags=["evidence"], dependencies=[Depends(legacy_disabled)])


@router.get("/{conversation_id}/evidence")
def get_evidence(conversation_id: str, request: Request):
    runtime = request.app.state.runtime
    if not runtime.get_session(conversation_id):
        raise HTTPException(status_code=404, detail="session not found")
    session = runtime.get_session(conversation_id) or {}
    focus = session.get("current_focus") or {}
    order_id = focus.get("order_id")
    return {"conversation_id": conversation_id, "items": runtime.evidence_for_order(order_id) if order_id else session.get("evidence", [])}
