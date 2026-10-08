from fastapi import APIRouter, Depends, Request

from servemind.api.legacy import legacy_disabled

router = APIRouter(tags=["runtime"], dependencies=[Depends(legacy_disabled)])


@router.get("/skills")
def skills(request: Request):
    return request.app.state.runtime.skills.summary()


@router.post("/skills/reload")
def reload_skills(request: Request):
    request.app.state.runtime.skills.load()
    return request.app.state.runtime.skills.summary()


@router.get("/tools")
def tools(request: Request):
    runtime = request.app.state.runtime
    return {"tools": runtime.tools.list_tools(), "stats": runtime.tools.stats()}


@router.get("/agents")
def agents(request: Request):
    return {"agents": request.app.state.runtime.agent_summary()}


@router.get("/monitor")
def monitor(request: Request):
    return request.app.state.runtime.monitor.summary()


@router.get("/memory/{conversation_id}")
def memory(conversation_id: str, request: Request):
    return request.app.state.runtime.sessions.context(conversation_id)


@router.get("/knowledge")
def knowledge(request: Request):
    return request.app.state.runtime.knowledge.summary()


@router.get("/traces")
def traces(request: Request):
    return {"items": request.app.state.runtime.tools.traces[-50:]}


@router.post("/eval/run")
def eval_run(request: Request):
    from servemind.evaluation.evaluator import evaluate_all
    return evaluate_all(request.app.state.runtime)
