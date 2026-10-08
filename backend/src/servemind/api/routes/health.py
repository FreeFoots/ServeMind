import os

from fastapi import APIRouter, Request

from servemind.config.policy_registry import policy_metadata
from servemind.mcp.customer_data_provider import DATA_VERSION
from servemind.config.model_settings import model_metadata

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "service": "servemind-api", "model": model_metadata(), "data_version": DATA_VERSION,
            "policy": policy_metadata(), "dependencies": {
                "provider": "merchant-declared-products",
                "commerce_storage": request.app.state.commerce_store.backend,
                "historical_provider": os.getenv("SERVEMIND_DATA_BACKEND", "csv"),
                "historical_api": "disabled_until_verified_identity",
                "working_memory": "redis",
                "durable_memory": "postgres" if request.app.state.commerce_support.durable_memory else "disabled",
                "semantic_memory_enabled": request.app.state.commerce_support.semantic_memory,
                "knowledge": os.getenv("SERVEMIND_RAG_BACKEND", "keyword"),
                "skills": "dynamic",
                "agent_runtime": "model_intent_domain_routing_parallel_roles" if request.app.state.commerce_support.model_runtime else "deterministic_fallback",
            }}


@router.get("/model")
def model():
    return model_metadata()
