from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from servemind.agents.task_graph import (IntentDecomposer, ParallelAgentRunner,
                                            ParallelContributionRunner, TaskGraph, TaskNode)
from servemind.core.intent_recognizer import classify_message
from servemind.evaluation.evaluator import _judge_cost_estimate
from servemind.mcp.tool_manager import Tool, ToolManager
from servemind.service.commerce_support import CommerceSupport
from servemind.api.main import create_app


def test_decomposer_keeps_delivery_and_price_as_separate_tasks():
    message = "订单 81a6fa818d 的物流和价格优惠都查一下"
    graph = IntentDecomposer().decompose(message, classify_message(message),
                                         order_id="81a6fa818d", sku_id=None)
    assert {"delivery_status", "price_breakdown"}.issubset(graph.intents)
    assert {"get_order_facts", "get_delivery_timeline", "get_price_breakdown"} == {
        node.tool for node in graph.nodes}


def test_graph_cycle_and_size_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph((), (TaskNode("a", "x", "general", "x", {}, ("b",)),
                       TaskNode("b", "x", "general", "x", {}, ("a",)))).validate()
    with pytest.raises(ValueError, match="too_large"):
        TaskGraph((), tuple(TaskNode(str(i), "x", "general", "x", {})
                            for i in range(13))).validate()


def test_parallel_runner_completes_independent_tools_and_fails_closed():
    manager = ToolManager()
    manager.register(Tool("a", "read", lambda p, c: ["a"],
                          required_permissions=("policy_public",)))
    manager.register(Tool("b", "read", lambda p, c: ["b"],
                          required_permissions=("policy_public",)))
    graph = TaskGraph(("x",), (TaskNode("task-1", "x", "general", "a", {}),
                                TaskNode("task-2", "x", "general", "b", {})))
    results, status = ParallelAgentRunner(manager).run(graph, {
        "allowed_tools": {"a", "b"}, "permissions": {"policy_public"}})
    assert [result.data for _, result in results] == [["a"], ["b"]]
    assert status["completion_rate"] == 1.0
    _, denied = ParallelAgentRunner(manager).run(graph, {"allowed_tools": {"a", "b"}})
    assert denied["completion_rate"] == 0.0


def test_task_dependency_runs_after_parent_and_is_blocked_on_failure():
    calls = []
    manager = ToolManager()
    manager.register(Tool("first", "read", lambda p, c: calls.append("first") or []))
    manager.register(Tool("second", "read", lambda p, c: calls.append("second") or []))
    graph = TaskGraph(("x",), (TaskNode("task-1", "x", "general", "first", {}),
                                TaskNode("task-2", "x", "general", "second", {}, ("task-1",))))
    _, status = ParallelAgentRunner(manager).run(graph, {"allowed_tools": {"first", "second"}})
    assert status["completion_rate"] == 1.0
    assert calls == ["first", "second"]
    calls.clear()
    results, status = ParallelAgentRunner(manager).run(graph, {"allowed_tools": {"second"}})
    assert status["completion_rate"] == 0.0
    assert calls == []
    assert results[1][1].error == "dependency_failed"


def test_live_commerce_routes_multiple_agent_roles():
    product = {"id": "prod_demo", "catalog_status": "暂时缺货",
               "display_price": "67.90", "price_basis": "historical_reference_simulated",
               "provenance": "merchant_declared_unverified"}
    answer, metadata = CommerceSupport(use_llm=False).reply("商品还有货吗，价格是多少", product)
    assert {"catalog", "billing"}.issubset(metadata["agent_roles"])
    assert metadata["task_graph"]["completion_rate"] == 1.0
    assert "暂时缺货" in answer and "67.90" in answer


def test_provider_usage_cost_is_explicitly_estimated():
    cost = _judge_cost_estimate(input_tokens=100, output_tokens=20,
                                cache_hit_tokens=0, cache_miss_tokens=100)
    assert cost["estimated_value"] > 0
    assert cost["basis"] == "provider_reported_cache_split"
    assert cost["source"].startswith("https://api-docs.deepseek.com/")


def test_live_api_multi_intent_stays_private_until_explicit_handoff(tmp_path):
    app = create_app(commerce_db_path=tmp_path / "commerce.sqlite3", use_llm=False)
    with TestClient(app) as client:
        def register(name: str, role: str):
            response = client.post("/v1/commerce/accounts/register", json={
                "username": name, "password": "test-password-123", "display_name": name,
                "role": role})
            assert response.status_code == 200
            return {"Authorization": f"Bearer {response.json()['token']}"}

        merchant = register("merchant-multi", "merchant")
        buyer = register("buyer-multi", "buyer")
        product_response = client.post("/v1/commerce/products", json={
            "title": "演示耳机", "description": "商家自填蓝牙耳机", "price": "99.00"},
            headers=merchant)
        assert product_response.status_code == 200, product_response.text
        product_id = product_response.json()["id"]
        conversation = client.post("/v1/commerce/conversations",
                                   json={"product_id": product_id}, headers=buyer).json()
        path = f"/v1/commerce/conversations/{conversation['id']}"
        first = client.post(path + "/messages", json={
            "content": "商品还有货吗，价格是多少", "client_message_id": "multi-1"},
            headers=buyer)
        assert first.status_code == 200, first.text
        ai = first.json()["ai_message"]
        assert "¥99.00" in ai["content"]
        assert ai["metadata"]["needs_merchant"] is False
        assert client.get(path, headers=merchant).status_code == 404
        second = client.post(path + "/messages", json={
            "content": "请找人工确认商品价格", "client_message_id": "multi-2"},
            headers=buyer)
        assert second.status_code == 200, second.text
        assert second.json()["ai_message"]["metadata"]["needs_merchant"] is True
        assert client.get(path, headers=merchant).status_code == 200
