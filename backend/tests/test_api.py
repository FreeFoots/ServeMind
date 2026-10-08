from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from servemind.api.main import app, create_app


@pytest.fixture
def client(tmp_path):
    # Tests never touch the local demo's accounts or conversations.
    with TestClient(create_app(commerce_db_path=tmp_path / "commerce.sqlite3", runtime=app.state.runtime, use_llm=False)) as test_client:
        yield test_client


def register(client: TestClient, username: str, role: str) -> dict:
    response = client.post(
        "/v1/commerce/accounts/register",
        json={"username": username, "password": "test-password-123", "display_name": username, "role": role},
    )
    assert response.status_code == 200, response.text
    return response.json()


def auth(identity: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {identity['token']}"}


def test_health_and_legacy_order_routes_are_closed(client: TestClient):
    assert client.get("/v1/health").status_code == 200
    assert client.post("/v1/evaluations/run?live_judge=true").status_code == 403
    for method, path in (
        ("post", "/v1/sessions"),
        ("get", "/v1/sessions/old/evidence"),
        ("get", "/v1/memory/old"),
        ("get", "/v1/traces"),
        ("post", "/v1/eval/run"),
    ):
        assert getattr(client, method)(path).status_code == 410


def test_registration_login_and_token_auth(client: TestClient):
    buyer = register(client, "buyer-one", "buyer")
    assert buyer["account"]["role"] == "buyer"
    assert "password_hash" not in buyer["account"]
    assert client.get("/v1/commerce/me").status_code == 401
    assert client.get("/v1/commerce/me", headers=auth(buyer)).json()["account"]["id"] == buyer["account"]["id"]
    assert client.post(
        "/v1/commerce/accounts/register",
        json={"username": "BUYER-ONE", "password": "test-password-123", "display_name": "duplicate", "role": "buyer"},
    ).status_code == 409
    assert client.post(
        "/v1/commerce/accounts/login", json={"username": "buyer-one", "password": "wrong"}
    ).status_code == 401
    login = client.post(
        "/v1/commerce/accounts/login", json={"username": "buyer-one", "password": "test-password-123"}
    )
    assert login.status_code == 200
    assert login.json()["account"]["id"] == buyer["account"]["id"]


def test_buyer_product_merchant_shared_thread_and_isolation(client: TestClient):
    merchant = register(client, "merchant-one", "merchant")
    buyer = register(client, "buyer-two", "buyer")
    stranger = register(client, "buyer-other", "buyer")
    other_merchant = register(client, "merchant-other", "merchant")

    assert client.post(
        "/v1/commerce/products", json={"title": "No", "description": "No"}, headers=auth(buyer)
    ).status_code == 403
    product_response = client.post(
        "/v1/commerce/products",
        json={"title": "测试耳机", "description": "商家自行填写的描述", "sku_id": "81a6fa818d"},
        headers=auth(merchant),
    )
    assert product_response.status_code == 200
    product = product_response.json()
    assert product["merchant_id"] == merchant["account"]["id"]
    assert product["provenance"] == "merchant_declared_unverified"
    assert product["verification_status"] == "unverified"
    assert client.get("/v1/commerce/products", headers=auth(buyer)).json()["items"][0]["id"] == product["id"]

    assert client.post(
        "/v1/commerce/conversations", json={"product_id": "prod_missing"}, headers=auth(buyer)
    ).status_code == 404
    assert client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(merchant)
    ).status_code == 403
    created = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    )
    assert created.status_code == 200
    conversation = created.json()
    conversation_id = conversation["id"]
    assert conversation["buyer"]["id"] == buyer["account"]["id"]
    assert conversation["merchant"]["id"] == merchant["account"]["id"]
    assert "username" not in conversation["buyer"]
    assert "username" not in conversation["merchant"]
    assert "username" not in conversation["product"]["merchant"]
    assert conversation["product"]["provenance"] == "merchant_declared_unverified"
    assert conversation["messages"] == []

    path = f"/v1/commerce/conversations/{conversation_id}"
    assert client.get(path, headers=auth(stranger)).status_code == 404
    assert client.get(path, headers=auth(other_merchant)).status_code == 404
    assert client.post(path + "/messages", json={"content": "偷看"}, headers=auth(stranger)).status_code == 404

    sent = client.post(
        path + "/messages",
        json={"content": "订单 81a6fa818d 物流在哪里？", "client_message_id": "buyer-1"},
        headers=auth(buyer),
    )
    assert sent.status_code == 200
    first = sent.json()
    assert first["message"]["sender_type"] == "buyer"
    assert first["ai_message"]["sender_type"] == "ai"
    assert first["ai_message"]["reply_to_id"] == first["message"]["id"]
    assert "81a6fa818d" not in first["ai_message"]["content"]
    assert "81a6fa818d" not in json.dumps(first["ai_message"]["metadata"])
    assert first["ai_message"]["metadata"]["evidence_ids"] == []
    assert first["ai_message"]["metadata"]["grounded"] is False
    assert first["ai_message"]["metadata"]["routing"]["primary"] == "fulfillment"

    assert client.get(path, headers=auth(merchant)).status_code == 404
    assert client.get("/v1/commerce/conversations", headers=auth(merchant)).json()["items"] == []
    handoff = client.post(
        path + "/messages", json={"content": "我要退款，请商家处理"}, headers=auth(buyer)
    )
    assert handoff.status_code == 200
    assert handoff.json()["ai_message"]["metadata"]["needs_merchant"] is True
    visible_to_merchant = client.get(path, headers=auth(merchant)).json()["messages"]
    assert [item["sender_type"] for item in visible_to_merchant] == ["buyer", "ai", "buyer", "ai"]
    merchant_reply = client.post(
        path + "/messages", json={"content": "您好，我会核实您的问题。"}, headers=auth(merchant)
    )
    assert merchant_reply.status_code == 200
    assert merchant_reply.json()["ai_message"] is None
    visible_to_buyer = client.get(path, headers=auth(buyer)).json()["messages"]
    assert [item["sender_type"] for item in visible_to_buyer] == ["buyer", "ai", "buyer", "ai", "merchant"]
    assert client.get("/v1/commerce/conversations", headers=auth(merchant)).json()["items"][0]["id"] == conversation_id
    assert client.get("/v1/commerce/conversations", headers=auth(stranger)).json()["items"] == []


def test_client_message_id_is_idempotent(client: TestClient):
    merchant = register(client, "merchant-repeat", "merchant")
    buyer = register(client, "buyer-repeat", "buyer")
    product = client.post(
        "/v1/commerce/products", json={"title": "商品", "description": "演示"}, headers=auth(merchant)
    ).json()
    conversation_id = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    ).json()["id"]
    path = f"/v1/commerce/conversations/{conversation_id}/messages"
    payload = {"content": "你好", "client_message_id": "retry-key"}
    first = client.post(path, json=payload, headers=auth(buyer)).json()
    second = client.post(path, json=payload, headers=auth(buyer)).json()
    assert second == first
    assert client.post(
        path, json={"content": "不同内容", "client_message_id": "retry-key"}, headers=auth(buyer)
    ).status_code == 409
    detail = client.get(f"/v1/commerce/conversations/{conversation_id}", headers=auth(buyer)).json()
    assert len(detail["messages"]) == 2


def test_client_cannot_spoof_merchant_or_message_sender(client: TestClient):
    merchant = register(client, "merchant-owner", "merchant")
    another_merchant = register(client, "merchant-spoof", "merchant")
    buyer = register(client, "buyer-spoof", "buyer")
    assert client.post(
        "/v1/commerce/products",
        json={"title": "伪造商品", "merchant_id": another_merchant["account"]["id"]},
        headers=auth(merchant),
    ).status_code == 422
    product = client.post(
        "/v1/commerce/products", json={"title": "真实发布商品"}, headers=auth(merchant)
    ).json()
    assert product["merchant_id"] == merchant["account"]["id"]
    assert client.post(
        "/v1/commerce/conversations",
        json={"product_id": product["id"], "merchant_id": another_merchant["account"]["id"]},
        headers=auth(buyer),
    ).status_code == 422
    conversation = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    ).json()
    assert conversation["merchant"]["id"] == merchant["account"]["id"]
    path = f"/v1/commerce/conversations/{conversation['id']}/messages"
    assert client.post(
        path, json={"content": "冒充商家", "sender_type": "merchant"}, headers=auth(buyer)
    ).status_code == 422
    assert client.post(path, json={"content": "我是买家"}, headers=auth(buyer)).json()["message"]["sender_type"] == "buyer"


def test_conversation_survives_app_reinitialization(client: TestClient):
    merchant = register(client, "merchant-persist", "merchant")
    buyer = register(client, "buyer-persist", "buyer")
    product = client.post(
        "/v1/commerce/products", json={"title": "持久化商品"}, headers=auth(merchant)
    ).json()
    conversation_id = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    ).json()["id"]
    client.post(
        f"/v1/commerce/conversations/{conversation_id}/messages",
        json={"content": "我要退款，重新打开后还能看到吗？"}, headers=auth(buyer),
    )
    db_path = client.app.state.commerce_store.path
    with TestClient(create_app(commerce_db_path=db_path, runtime=app.state.runtime, use_llm=False)) as reloaded:
        assert reloaded.get("/v1/commerce/me", headers=auth(buyer)).json()["account"]["id"] == buyer["account"]["id"]
        detail = reloaded.get(f"/v1/commerce/conversations/{conversation_id}", headers=auth(merchant)).json()
        assert detail["product"]["id"] == product["id"]
        assert [message["sender_type"] for message in detail["messages"]] == ["buyer", "ai"]


def test_inbox_orders_by_latest_message(client: TestClient):
    merchant = register(client, "merchant-inbox", "merchant")
    buyer = register(client, "buyer-inbox", "buyer")
    product = client.post(
        "/v1/commerce/products", json={"title": "收件箱商品"}, headers=auth(merchant)
    ).json()
    first = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    ).json()
    second = client.post(
        "/v1/commerce/conversations", json={"product_id": product["id"]}, headers=auth(buyer)
    ).json()
    before = client.get("/v1/commerce/conversations", headers=auth(merchant)).json()["items"]
    assert before == []
    client.post(f"/v1/commerce/conversations/{second['id']}/messages", json={"content": "我要退款"}, headers=auth(buyer))
    client.post(f"/v1/commerce/conversations/{first['id']}/messages", json={"content": "我要退款"}, headers=auth(buyer))
    before = client.get("/v1/commerce/conversations", headers=auth(merchant)).json()["items"]
    assert [item["id"] for item in before] == [first["id"], second["id"]]
    client.post(
        f"/v1/commerce/conversations/{first['id']}/messages",
        json={"content": "新消息"}, headers=auth(buyer),
    )
    after = client.get("/v1/commerce/conversations", headers=auth(merchant)).json()["items"]
    assert [item["id"] for item in after] == [first["id"], second["id"]]
    assert after[0]["updated_at"] > after[0]["created_at"]
