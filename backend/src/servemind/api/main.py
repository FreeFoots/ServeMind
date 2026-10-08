from __future__ import annotations

from pathlib import Path
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from servemind.agents.agent_orchestrator import ServeMindAgent
from servemind.api.routes import admin, commerce, evaluations, evidence, handoffs, health, sessions
from servemind.config.settings import HOST, PORT, configured_cors_origins
from servemind.service.commerce_store import CommerceStore, DEFAULT_COMMERCE_DB
from servemind.service.commerce_support import CommerceSupport
from servemind.service.postgres_commerce_store import PostgresCommerceStore
from servemind.monitor.operations import PerformanceMonitor
from servemind.api.routes import operations

APP_NAME = "ServeMind API"

def create_app(*, commerce_db_path: Path | str | None = None, runtime: ServeMindAgent | None = None,
               use_llm: bool | None = None, commerce_store: CommerceStore | None = None) -> FastAPI:
    """Create an API with injectable commerce storage for isolated tests."""
    @asynccontextmanager
    async def lifespan(app):
        await app.state.operations.start()
        try:
            yield
        finally:
            await app.state.operations.stop()
    application = FastAPI(title=APP_NAME, version="0.4.0", docs_url="/docs", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.state.runtime = runtime if runtime is not None else ServeMindAgent()
    if os.getenv('SERVEMIND_COMMERCE_BACKEND','sqlite') not in {'sqlite','postgres'}:
        raise ValueError('invalid_commerce_backend')
    application.state.commerce_store = (commerce_store if commerce_store is not None else
        CommerceStore(commerce_db_path) if commerce_db_path is not None else
        PostgresCommerceStore() if os.getenv('SERVEMIND_COMMERCE_BACKEND','sqlite') == 'postgres' else
        CommerceStore(DEFAULT_COMMERCE_DB))
    application.state.commerce_support = CommerceSupport(
        use_llm=use_llm, use_durable_memory=commerce_db_path is None and commerce_store is None,
    )
    application.state.operations = PerformanceMonitor(application.state.commerce_support)
    @application.middleware('http')
    async def http_metrics(request, call_next):
        started=time.monotonic()
        status=500
        try:
            response=await call_next(request)
            status=response.status_code
            return response
        finally:
            route=getattr(request.scope.get('route'),'path','unmatched')
            monitor=application.state.operations
            monitor.http_requests.labels(route,request.method,str(status)).inc()
            monitor.http_latency.labels(route).observe(time.monotonic()-started)
    application.include_router(operations.router)
    application.include_router(health.router, prefix="/v1")
    application.include_router(commerce.router, prefix="/v1")
    application.include_router(evaluations.router, prefix="/v1")
    # Legacy endpoints retain their paths but are disabled until historical
    # buyer/order ownership can be verified against a trusted identity source.
    application.include_router(sessions.router, prefix="/v1")
    application.include_router(evidence.router, prefix="/v1")
    application.include_router(handoffs.router, prefix="/v1")
    application.include_router(admin.router, prefix="/v1")
    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("servemind.api.main:app", host=HOST, port=PORT, reload=False)
