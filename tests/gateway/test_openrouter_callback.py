"""The gateway relays a hosted OpenRouter callback without retaining its authorization code."""
from __future__ import annotations

import httpx
from fastmcp.server.auth.providers.google import GoogleProvider
from starlette.testclient import TestClient

from gateway.app import build_app
from gateway.children import Child


class _Process:
    pid = 1
    returncode = None

    def terminate(self):
        self.returncode = 0

    async def wait(self):
        return 0


def _app(tmp_path, monkeypatch):
    async def no_token(self, token):
        return None

    monkeypatch.setattr(GoogleProvider, "verify_token", no_token)
    return build_app(base_url="https://gw.example.com", client_id="cid", client_secret="secret",
                     homes_root=tmp_path, idle_seconds=9999)


def test_openrouter_callback_is_one_use_and_relays_only_to_its_child(tmp_path, monkeypatch):
    received = {}

    async def relay(self, url, **kwargs):
        received.update(url=url, json=kwargs["json"], headers=kwargs["headers"])
        return httpx.Response(200, json={"status": "stored"})

    monkeypatch.setattr(httpx.AsyncClient, "post", relay)
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        pool = app.state.pool
        child = Child(subject="42", home=tmp_path / "42", port=5432,
                      proc=_Process(), last_seen=0, interaction_key="child-key")  # type: ignore[arg-type]
        pool._children[child.subject] = child

        registered = client.post("/_internal/hosted/register",
                                 headers={"X-Opyt-Hosted-Interaction-Key": "child-key"},
                                 json={"kind": "openrouter", "nonce": "nonce-a"})
        assert registered.status_code == 204

        callback = client.get("/login/openrouter/nonce-a?code=authorization-code")
        assert callback.status_code == 200
        assert "OpenRouter is connected" in callback.text
        assert "authorization-code" not in callback.text
        assert callback.headers["cache-control"] == "no-store"

        assert received == {
            "url": "http://127.0.0.1:5432/_hosted-openrouter/callback/nonce-a",
            "json": {"code": "authorization-code"},
            "headers": {"X-Opyt-Hosted-Interaction-Key": "child-key"},
        }
        assert client.get("/login/openrouter/nonce-a?code=authorization-code").status_code == 404


def test_openrouter_callback_never_calls_a_child_without_a_code(tmp_path, monkeypatch):
    called = False

    async def relay(self, url, **kwargs):
        nonlocal called
        called = True
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "post", relay)
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/login/openrouter/not-a-real-route")

    assert response.status_code == 400
    assert called is False
