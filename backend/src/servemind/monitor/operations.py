"""Periodic performance collection, private Prometheus, alerts and route feedback."""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from servemind.config.settings import PROJECT_ROOT


class AnomalyDetector:
    def __init__(self, window=60, sensitivity=2.5):
        self.history = defaultdict(lambda: deque(maxlen=window))
        self.sensitivity = sensitivity

    def record(self, key, value):
        prior = self.history[key]
        score = None
        if len(prior) >= 20:
            mean, deviation = statistics.mean(prior), statistics.stdev(prior)
            if deviation and abs(value-mean)/deviation > self.sensitivity:
                score = round(abs(value-mean)/deviation, 3)
        prior.append(value)
        return score


class PerformanceMonitor:
    def __init__(self, support, *, log_path: Path | None = None, webhook: str | None = None):
        self.support = support
        self.registry = CollectorRegistry()
        self.requests = Counter('servemind_requests_total','Commercial generations', ['status'], registry=self.registry)
        self.latency = Histogram('servemind_request_latency_seconds','Observed request latency',
                                buckets=(.1,.5,1,3,5,10,20,40,60,90), registry=self.registry)
        self.http_requests=Counter('servemind_http_requests_total','API results',['route','method','status'],registry=self.registry)
        self.http_latency=Histogram('servemind_http_latency_seconds','API observed latency',['route'],registry=self.registry)
        self.tools = Counter('servemind_tool_calls_total','Tool calls',['tool','status'],registry=self.registry)
        self.tokens = Counter('servemind_model_tokens_total','Provider reported usage',['direction'],registry=self.registry)
        self.cost = Counter('servemind_estimated_cost_usd_total','Estimated, not invoiced, cost',registry=self.registry)
        self.agents = Gauge('servemind_agent_success_ratio','Logical instance success',['agent'],registry=self.registry)
        self.penalties = Gauge('servemind_agent_routing_penalty','Monitor routing penalty',['agent'],registry=self.registry)
        self.task_counts = Counter('servemind_tasks_total','Task graph nodes',['status'],registry=self.registry)
        self.safety = Counter('servemind_safety_events_total','Governance failures',['kind'],registry=self.registry)
        self.detector = AnomalyDetector()
        self.alerts = deque(maxlen=200)
        self.active = {}
        self._last = {}
        self._lock = threading.RLock()
        self.task = None
        self.log_path = log_path or PROJECT_ROOT/'backend/runtime/monitor/alerts.jsonl'
        self.webhook = webhook if webhook is not None else os.getenv('SERVEMIND_ALERT_WEBHOOK','')
        support.monitor.observer = self.observe

    def observe(self, event):
        # Finite domain labels only: never request, account or product IDs.
        self.requests.labels(event['status']).inc()
        self.latency.observe(event['latency_ms']/1000)
        for tool in event['tools']:
            self.tools.labels(tool['name'],tool['status']).inc()
            if str(tool.get('error','')).startswith('forbidden:') or tool.get('error')=='PermissionError':
                self.safety.labels('unauthorized').inc()
        self.task_counts.labels('completed').inc(event['graph']['completed'])
        self.task_counts.labels('failed').inc(event['graph']['failed'])
        if not event['grounding']['passed']:
            self.safety.labels('grounding').inc()
        if event['model_use'].get('output_rejected'):
            self.safety.labels('output_rejected').inc()
        usage = event['model_use']
        self.tokens.labels('input').inc(usage.get('prompt_tokens',0))
        self.tokens.labels('output').inc(usage.get('completion_tokens',0))
        self.cost.inc((usage.get('estimated_cost') or {}).get('estimated_value',0))

    @staticmethod
    def routing_penalty(success_rate, avg_ms):
        return min(.9,max(0,(.9-success_rate)*2)+min(.4,max(0,(avg_ms-3000)/10000)))

    def _check(self, key, value, threshold, breached):
        now = time.monotonic()
        with self._lock:
            if not breached and key not in self.active:
                return
            if breached and key in self.active and now-self._last.get(key,0)<300:
                return
            alert = {'timestamp':datetime.now(timezone.utc).isoformat(),'metric':key,
                     'value':round(value,4),'threshold':threshold,'state':'firing' if breached else 'resolved'}
            if breached:
                self.active[key]=alert
            else:
                self.active.pop(key,None)
            self._last[key]=now
            self.alerts.append(alert)
            try:
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                # Daily files, no destructive rotation or user content.
                dated = self.log_path.with_name(self.log_path.stem+'-'+datetime.now(timezone.utc).strftime('%Y-%m-%d')+'.jsonl')
                with dated.open('a',encoding='utf-8') as out:
                    out.write(json.dumps(alert,ensure_ascii=False)+'\n')
                dated.chmod(0o600)
            except OSError:
                pass  # Observability failure must not fail a customer reply.
        if self.webhook:
            try:
                httpx.post(self.webhook,json=alert,timeout=3,follow_redirects=False)
            except httpx.HTTPError:
                pass

    def collect(self):
        runtime = self.support.model_runtime
        penalties = {}
        for name, stats in (runtime.health() if runtime else {}).items():
            rate = stats['success']/stats['total'] if stats['total'] else 1
            penalty = self.routing_penalty(rate,stats['avg_ms'])
            penalties[name]=penalty
            self.agents.labels(name).set(rate)
            self.penalties.labels(name).set(penalty)
            if stats['total'] >= 5:
                self._check('agent_success:'+name,rate,.9,rate<.9)
                self._check('agent_latency:'+name,stats['avg_ms'],3000,stats['avg_ms']>3000)
                z = self.detector.record('agent_latency:'+name,stats['avg_ms'])
                self._check('latency_anomaly:'+name,z or 0,2.5,bool(z))
        if runtime:
            runtime.update_routing_penalties(penalties)
        for name, stats in self.support.tools.stats().items():
            if stats['total'] >= 5:
                self._check('tool_success:'+name,stats['success_rate'],.95,stats['success_rate']<.95)
                self._check('tool_latency:'+name,stats['avg_latency_ms'],5000,stats['avg_latency_ms']>5000)
            self._check('tool_circuit:'+name,1 if stats['circuit']['state']=='open' else 0,0,
                        stats['circuit']['state']=='open')

    async def start(self):
        if self.task is None and os.getenv('SERVEMIND_MONITOR_ENABLED','true').lower()=='true':
            self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            try:
                await asyncio.to_thread(self.collect)
            except Exception:
                # No exception bodies: DSNs/webhook credentials must not be logged.
                pass
            await asyncio.sleep(max(1,float(os.getenv('SERVEMIND_MONITOR_INTERVAL','10'))))

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task=None

    def snapshot(self):
        with self._lock:
            return {'active_alerts':list(self.active.values()),'recent_alerts':list(self.alerts)[-20:],
                    'collection_running':self.task is not None,'webhook_configured':bool(self.webhook),
                    'metrics_scope':'single_process_counters_aggregate_via_prometheus',
                    'suggestions':['检查依赖、熔断状态和同角色实例，禁止扩大权限来绕过故障'] if self.active else []}

    def render(self):
        return generate_latest(self.registry)
