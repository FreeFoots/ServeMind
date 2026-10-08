from fastapi import APIRouter, HTTPException, Request

from servemind.evaluation.evaluator import evaluate_all
from servemind.evaluation.test_cases import ANSWER_CASES, CONTEXT_CASES, DATA_ANSWER_CASES, DATA_CONTEXT_CASES, DATA_INTENT_CASES, DATA_MULTI_INTENT_CASES, END_TO_END_CASES, INTENT_CASES, JUDGE_CASES, MULTI_INTENT_CASES, RAG_CASES, TASK_GRAPH_CASES
from servemind.evaluation.commerce_cases import COMMERCE_CASES
from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES
from servemind.evaluation.expanded_cases import (EXPANDED_INTENT_CASES,
                                                    EXPANDED_TASK_GRAPH_CASES,
                                                    EXPANDED_COMMERCE_CASES)


router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/cases")
def cases():
    return {"intent": [*INTENT_CASES, *DATA_INTENT_CASES, *EXPANDED_INTENT_CASES], "context": [*CONTEXT_CASES, *DATA_CONTEXT_CASES], "rag": RAG_CASES, "answers": [*ANSWER_CASES, *DATA_ANSWER_CASES], "judge": JUDGE_CASES, "multi_intent": [*MULTI_INTENT_CASES, *DATA_MULTI_INTENT_CASES], "task_graph": [*TASK_GRAPH_CASES, *EXPANDED_TASK_GRAPH_CASES], "commerce_conversation": [*COMMERCE_CASES, *EXPANDED_COMMERCE_CASES], "curated_commerce": CURATED_COMMERCE_CASES, "end_to_end": [{**case, "must_use": sorted(case["must_use"])} for case in END_TO_END_CASES]}


@router.post("/run")
def run(request: Request):
    # The public local endpoint must not allow unauthenticated paid model runs.
    if "live_judge" in request.query_params:
        raise HTTPException(status_code=403, detail="live_judge_cli_only")
    return evaluate_all(request.app.state.runtime)
