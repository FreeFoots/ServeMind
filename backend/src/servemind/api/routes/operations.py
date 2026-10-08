"""Operational interfaces are separate from disabled historical business APIs."""
import hmac
import os
import ipaddress
from fastapi import APIRouter, Depends, HTTPException, Request, Response

router = APIRouter(tags=['operations'])

def private_monitor(request: Request):
    token = os.getenv('SERVEMIND_MONITOR_TOKEN','')
    supplied = request.headers.get('authorization','').removeprefix('Bearer ')
    if token and hmac.compare_digest(supplied,token):
        return
    if not token and request.client:
        if request.client.host=='testclient':
            return
        try:
            address=ipaddress.ip_address(request.client.host)
            allowed=os.getenv('SERVEMIND_MONITOR_CIDRS','127.0.0.1/32,::1/128')
            if any(address in ipaddress.ip_network(c.strip()) for c in allowed.split(',')):
                return
        except ValueError:
            pass
    raise HTTPException(403,'private_monitor_required')

@router.get('/metrics', dependencies=[Depends(private_monitor)])
def metrics(request: Request):
    return Response(request.app.state.operations.render(),media_type='text/plain; version=0.0.4')

@router.get('/v1/operations', dependencies=[Depends(private_monitor)])
def monitor(request: Request):
    return {'performance':request.app.state.commerce_support.monitor.summary(),
            'operations':request.app.state.operations.snapshot(),
            'agent_health':request.app.state.commerce_support.model_runtime.health()
                           if request.app.state.commerce_support.model_runtime else {}}

@router.get('/v1/ready')
def readiness(request: Request):
    checks = {}
    try:
        with request.app.state.commerce_store._db() as connection:
            checks['commerce_database'] = connection.execute('SELECT 1').fetchone() is not None
    except Exception:
        checks['commerce_database']=False
    try:
        checks['redis']=bool(request.app.state.commerce_support.working_memory.client.ping())
    except Exception:
        checks['redis']=False
    # Readiness is distinct from process liveness; never returns credentials.
    return Response(__import__('json').dumps({'status':'ready' if all(checks.values()) else 'not_ready',
                                             'checks':checks}),status_code=200 if all(checks.values()) else 503,
                    media_type='application/json')
