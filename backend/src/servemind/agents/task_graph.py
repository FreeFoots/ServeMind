"""Bounded, auditable multi-intent task execution for the research Agent.

The graph never grants tools. Each node is executed through ToolManager with
the caller's already-scoped whitelist and permissions.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from servemind.core.intent_recognizer import Intent, IntentResult
from servemind.mcp.customer_data_provider import evidence_to_dict
from servemind.mcp.tool_manager import ToolResult


INTENT_TOOL = {
    Intent.ORDER_QUERY: ("get_order_facts",),
    Intent.DELIVERY_STATUS: ("get_order_facts", "get_delivery_timeline"),
    Intent.DELIVERY_EXCEPTION: ("get_order_facts", "get_delivery_timeline"),
    Intent.PRICE_BREAKDOWN: ("get_price_breakdown",),
    Intent.SKU_QUERY: ("get_sku",),
    Intent.INVENTORY_QUERY: ("get_inventory",),
    Intent.BROWSING_HISTORY: ("get_browsing_history",),
    Intent.REFUND_POLICY: ("search_knowledge",),
    Intent.CANCEL_POLICY: ("search_knowledge",),
}
ROLE_FOR_INTENT = {
    Intent.ORDER_QUERY: "order", Intent.DELIVERY_STATUS: "fulfillment",
    Intent.DELIVERY_EXCEPTION: "fulfillment", Intent.PRICE_BREAKDOWN: "billing",
    Intent.SKU_QUERY: "catalog", Intent.INVENTORY_QUERY: "catalog",
    Intent.BROWSING_HISTORY: "catalog", Intent.REFUND_POLICY: "billing",
    Intent.CANCEL_POLICY: "billing", Intent.COMPLAINT: "escalation",
    Intent.HUMAN_HANDOFF: "escalation",
    Intent.INVOICE_QUERY: "billing",
}
SIGNALS = {
    Intent.DELIVERY_EXCEPTION: ("没收到", "未收到", "没拿到", "丢件", "丢了", "延误", "超时", "物流没更新"),
    Intent.DELIVERY_STATUS: ("配送", "物流", "包裹", "送达", "发货", "到哪里", "到哪", "什么时候到", "几天到", "多久能到"),
    Intent.PRICE_BREAKDOWN: ("价格", "价钱", "优惠", "金额", "实付", "多少钱", "原价", "折扣"),
    Intent.SKU_QUERY: ("sku", "商品", "属性", "规格", "品牌"),
    Intent.INVENTORY_QUERY: ("库存", "有货", "缺货", "备货", "现货"),
    Intent.REFUND_POLICY: ("退款", "退货", "退钱", "怎么退"),
    Intent.CANCEL_POLICY: ("取消", "撤单"),
    Intent.COMPLAINT: ("投诉", "不满", "赔偿"),
    Intent.HUMAN_HANDOFF: ("人工", "找商家", "转商家", "交给商家", "交给店家", "联系店家", "客服"),
    Intent.INVOICE_QUERY: ("发票", "开票", "抬头"),
    Intent.ORDER_QUERY: ("订单", "买了什么"),
}


@dataclass(frozen=True)
class TaskNode:
    id: str
    intent: str
    role: str
    tool: str
    params: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskGraph:
    intents: tuple[str, ...]
    nodes: tuple[TaskNode, ...]
    deadline_seconds: float = 12.0

    def validate(self) -> None:
        if len(self.nodes) > 12:
            raise ValueError("task_graph_too_large")
        lookup = {node.id: node for node in self.nodes}
        if len(lookup) != len(self.nodes):
            raise ValueError("task_graph_duplicate_node")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("task_graph_cycle")
            if node_id in visited:
                return
            if node_id not in lookup:
                raise ValueError("task_graph_missing_dependency")
            visiting.add(node_id)
            for dependency in lookup[node_id].depends_on:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node in self.nodes:
            visit(node.id)


class IntentDecomposer:
    def decompose(self, message: str, intent: IntentResult, *, order_id: str | None,
                  sku_id: str | None) -> TaskGraph:
        text = message.lower()
        # Conditional future exceptions are not reports of an actual lost parcel.
        text = re.sub(r"如果[^，。；！？\n]{0,16}(?:没收到|未收到|没拿到|丢件|丢了)", "", text)
        found = [item for item, markers in SIGNALS.items() if any(marker in text for marker in markers)]
        if intent.intent not in found and not (intent.intent == Intent.DELIVERY_EXCEPTION and text != message.lower()):
            found.insert(0, intent.intent)
        if not found:
            found = [Intent.OTHER]
        # A bare order ID or a conditional future wish is not itself a verified
        # order action. Keep all explicit topics, but only escalate active ones.
        found = list(dict.fromkeys(found))[:7]
        nodes: list[TaskNode] = []
        for topic in found:
            for tool in INTENT_TOOL.get(topic, ()):
                if tool in {"get_order_facts", "get_delivery_timeline", "get_price_breakdown"}:
                    if not order_id:
                        continue
                    params = {"order_id": order_id}
                elif tool in {"get_sku", "get_inventory"}:
                    if not sku_id:
                        continue
                    params = {"sku_id": sku_id}
                elif tool == "search_knowledge":
                    params = {"query": "退款" if topic == Intent.REFUND_POLICY else "取消"}
                else:
                    params = {}
                if any(node.tool == tool and node.params == params for node in nodes):
                    continue
                nodes.append(TaskNode(f"task-{len(nodes) + 1}", topic.value,
                                      ROLE_FOR_INTENT.get(topic, "general"), tool, params))
        graph = TaskGraph(tuple(item.value for item in found), tuple(nodes))
        graph.validate()
        return graph


class ParallelAgentRunner:
    def __init__(self, tools: Any) -> None:
        self.tools = tools

    def run(self, graph: TaskGraph, context: dict[str, Any]) -> tuple[list[tuple[TaskNode, Any]], dict[str, Any]]:
        graph.validate()
        started = time.monotonic()
        results: list[tuple[TaskNode, Any]] = []
        if not graph.nodes:
            return results, {"total": 0, "completed": 0, "failed": 0, "timed_out": False,
                             "completion_rate": 1.0}
        executor = ThreadPoolExecutor(max_workers=min(8, len(graph.nodes)), thread_name_prefix="task-graph")
        pending = {node.id: node for node in graph.nodes}
        completed: dict[str, bool] = {}
        timed_out = False
        try:
            while pending:
                ready = [node for node in pending.values()
                         if all(dep in completed for dep in node.depends_on)]
                for node in ready:
                    pending.pop(node.id)
                runnable = []
                for node in ready:
                    if any(not completed[dep] for dep in node.depends_on):
                        results.append((node, ToolResult(False, tool_name=node.tool,
                                                         error="dependency_failed")))
                        completed[node.id] = False
                    else:
                        runnable.append(node)
                if not runnable:
                    continue
                remaining = graph.deadline_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                def node_context(node: TaskNode) -> dict[str, Any]:
                    scopes = context.get("role_scopes")
                    if scopes is None:
                        return context
                    return {**context, "agent_role": node.role,
                            "allowed_tools": set(context.get("allowed_tools", ())).intersection(scopes.get(node.role, ()))}
                futures = {executor.submit(self.tools.call, node.tool, node.params, node_context(node)): node
                           for node in runnable}
                try:
                    for future in as_completed(futures, timeout=remaining):
                        node = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = ToolResult(False, tool_name=node.tool,
                                                error=type(exc).__name__)
                        results.append((node, result))
                        completed[node.id] = result.success
                except FuturesTimeout:
                    timed_out = True
                    for future, node in futures.items():
                        if node.id not in completed:
                            future.cancel()
                            results.append((node, ToolResult(False, tool_name=node.tool,
                                                             error="task_timeout", degraded=True)))
                            completed[node.id] = False
                    break
        except FuturesTimeout:
            timed_out = True
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if timed_out:
            for node in pending.values():
                results.append((node, ToolResult(False, tool_name=node.tool,
                                                 error="task_timeout", degraded=True)))
        results.sort(key=lambda pair: int(pair[0].id.split("-")[-1]))
        successful = sum(bool(getattr(result, "success", False)) for _, result in results)
        return results, {"total": len(graph.nodes), "completed": successful,
                         "failed": len(graph.nodes) - successful, "timed_out": timed_out,
                         "completion_rate": round(successful / len(graph.nodes), 4),
                         "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}


class ParallelContributionRunner:
    """Run role-limited live-commerce Agent contributions with bounded retries."""

    @staticmethod
    def run(agents: dict[Any, Any], roles: list[Any], *, intent: Any,
            evidence: list[dict[str, Any]], message: str,
            timeout_seconds: float = 4.0) -> tuple[list[Any], dict[str, Any]]:
        if not roles:
            return [], {"total": 0, "completed": 0, "failed": 0,
                        "timed_out": False, "completion_rate": 1.0}

        def contribute(role: Any) -> Any:
            for attempt in range(2):
                try:
                    return agents[role].contribute(intent=intent, evidence=evidence, message=message)
                except Exception:
                    if attempt:
                        raise
            raise RuntimeError("agent_retry_exhausted")

        executor = ThreadPoolExecutor(max_workers=min(6, len(roles)), thread_name_prefix="commerce-agent")
        futures = {executor.submit(contribute, role): role for role in roles}
        completed: dict[Any, Any] = {}
        timed_out = False
        try:
            for future in as_completed(futures, timeout=timeout_seconds):
                try:
                    completed[futures[future]] = future.result()
                except Exception:
                    pass
        except FuturesTimeout:
            timed_out = True
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        contributions = [completed[role] for role in roles if role in completed]
        return contributions, {"total": len(roles), "completed": len(contributions),
                               "failed": len(roles) - len(contributions),
                               "timed_out": timed_out,
                               "completion_rate": round(len(contributions) / len(roles), 4)}


class EvidenceMerger:
    @staticmethod
    def merge(results: list[tuple[TaskNode, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for _node, result in results:
            if not getattr(result, "success", False) or not isinstance(result.data, list):
                continue
            for item in result.data:
                if hasattr(item, "evidence_id"):
                    evidence = evidence_to_dict(item)
                elif isinstance(item, dict) and "evidence_id" in item:
                    evidence = item
                elif isinstance(item, dict):
                    evidence = {
                        "evidence_id": f"ev_knowledge_{item.get('chunk_id') or item.get('title', 'unknown')}",
                        "kind": "knowledge", "display_label": item.get("display_title") or item.get("title"),
                        "title": item.get("title"), "retrieved_at": "", "source": item.get("source", "knowledge"),
                        "facts": [{"field": "content", "value": item.get("content", ""),
                                   "confidence": item.get("score", 0.0)}],
                        "scope": "policy", "version": item.get("document_version"),
                        "valid_from": item.get("valid_from"), "valid_until": item.get("valid_until"),
                    }
                else:
                    continue
                merged[evidence["evidence_id"]] = evidence
        return list(merged.values())


class ResponseComposer:
    @staticmethod
    def compose(intents: tuple[str, ...], evidence: list[dict[str, Any]], *,
                escalated: bool, base_answer: str) -> str:
        if len(intents) < 2:
            return base_answer
        kinds = {item.get("kind") for item in evidence}
        parts: list[str] = []
        if "delivery_status" in intents or "delivery_exception" in intents:
            delivery = next((item for item in evidence if item.get("kind") == "delivery"), None)
            facts = {fact["field"]: fact["value"] for fact in delivery.get("facts", [])} if delivery else {}
            parts.append("配送：" + (f"记录显示 {facts.get('arr_time')} 到达；这不能核验实际签收人。"
                                  if facts.get("arr_time") else "目前没有可核验的配送节点。"))
        if "price_breakdown" in intents:
            price = next((item for item in evidence if item.get("kind") == "price"), None)
            facts = price.get("facts", []) if price else []
            line = next((fact.get("value") for fact in facts if isinstance(fact.get("value"), dict)), None)
            parts.append("价格：" + (f"订单记录的原单价 {line.get('original_unit_price')}，成交单价 {line.get('final_unit_price')}；不代表支付成功。"
                                if line else "缺少可核验的订单价格记录。"))
        if "sku_query" in intents:
            parts.append("商品属性：" + ("已找到匿名 SKU 记录；它不证明真实商品名称或商家归属。"
                                   if "sku" in kinds else "缺少可核验的 SKU 记录。"))
        if "inventory_query" in intents:
            parts.append("库存：" + ("找到历史库存记录，但不能据此判断实时可售件数。"
                               if "inventory" in kinds else "没有可核验的历史库存记录，不能推断实时库存。"))
        if "refund_policy" in intents or "cancel_policy" in intents:
            parts.append("售后规则：" + ("已检索公共政策；实际退款或取消仍需订单核验及商家处理。"
                                 if "knowledge" in kinds else "当前没有可核验的政策材料，需人工确认。"))
        if escalated:
            parts.append("人工处理：已建议交由商家核实；AI 没有执行退款、取消或赔偿。")
        return "\n".join(parts) if parts else base_answer


def active_escalation(message: str, intents: tuple[str, ...]) -> bool:
    text = message.lower()
    if any(marker in text for marker in ("不需要转人工", "不要转人工", "不用转人工", "不想转人工", "先了解规则", "别通知店家", "别通知商家", "不要通知店家", "暂时不要联系商家")):
        return False
    if "如果" in text and not any(marker in text for marker in ("我要投诉", "要投诉", "找人工", "转人工", "请找商家", "找商家")):
        return False
    if any(marker in text for marker in ("没收到", "未收到", "没拿到", "丢件", "丢了", "投诉", "找人工", "要人工", "转人工", "找商家", "给商家", "交给商家", "交给店家", "通知店家", "联系店家", "不想继续聊机器人", "赔偿")):
        return True
    if "refund_policy" in intents and re.search(r"(?:我要|帮我|申请|想要|想|请立即|请马上|立即|请)退款", text):
        return True
    if "cancel_policy" in intents and re.search(r"(?:我要|帮我|想要|请立即|请马上|立即)取消", text):
        return True
    return False
