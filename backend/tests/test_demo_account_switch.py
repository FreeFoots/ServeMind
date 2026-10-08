from fastapi.testclient import TestClient

from servemind.api.main import create_app
from servemind.api.routes import commerce


def test_local_demo_account_selection_preserves_role_and_requires_no_password(tmp_path, monkeypatch):
    monkeypatch.setattr(commerce, "APP_ENV", "development")
    monkeypatch.setattr(commerce, "DEMO_ACCOUNT_SWITCH", True)
    app = create_app(commerce_db_path=tmp_path / "demo.sqlite3", use_llm=False)
    with TestClient(app) as client:
        created = client.post("/v1/commerce/accounts/register", json={
            "username": "sample-merchant", "password": "test-password-123",
            "display_name": "样例商家", "role": "merchant",
        }).json()["account"]
        response = client.get("/v1/commerce/demo/accounts")
        assert response.status_code == 200
        assert response.json()["items"] == [{
            "id": created["id"], "display_name": "样例商家", "role": "merchant",
        }]
        selected = client.post("/v1/commerce/demo/select", json={
            "account_id": created["id"],
        })
        assert selected.status_code == 200
        assert selected.json()["account"]["role"] == "merchant"
        token = selected.json()["token"]
        assert client.get("/v1/commerce/me", headers={
            "Authorization": "Bearer " + token,
        }).json()["account"]["id"] == created["id"]
        assert client.post("/v1/commerce/demo/select", json={
            "account_id": "missing",
        }).status_code == 404

    monkeypatch.setattr(commerce, "APP_ENV", "production")
    with TestClient(app) as client:
        assert client.get("/v1/commerce/demo/accounts").status_code == 404
