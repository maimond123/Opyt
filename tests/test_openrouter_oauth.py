"""
tests/test_openrouter_oauth.py

The zero-paste key acquisition. Tests the PURE pieces (PKCE derivation, URL shape, the exchange
body) and stubs the network — nothing here talks to OpenRouter.

The last test is the important one: a timeout must degrade to a URL the user can open by hand,
and the returned payload must never contain a key value.
"""

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import pytest

from opyt_core import openrouter_oauth as oo


def test_challenge_is_s256_of_the_verifier():
    v, c = oo._pkce_pair()
    want = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    assert c == want and "=" not in c


def test_auth_url_carries_callback_and_challenge():
    url = oo._auth_url("http://localhost:5000/cb/n", "CHAL")
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "openrouter.ai"
    assert q["callback_url"] == ["http://localhost:5000/cb/n"]
    assert q["code_challenge"] == ["CHAL"] and q["code_challenge_method"] == ["S256"]


def test_exchange_posts_the_verifier_and_returns_the_key(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, body=json)
        return {"key": "sk-or-v1-abc"}

    monkeypatch.setattr(oo, "_post_json", fake_post)
    assert oo._exchange("CODE", "VERIFIER") == "sk-or-v1-abc"
    assert seen["url"].endswith("/api/v1/auth/keys")
    assert seen["body"] == {"code": "CODE", "code_verifier": "VERIFIER",
                            "code_challenge_method": "S256"}


def test_exchange_without_a_key_field_raises(monkeypatch):
    monkeypatch.setattr(oo, "_post_json", lambda *a, **k: {"error": "bad code"})
    with pytest.raises(oo.OAuthError):
        oo._exchange("CODE", "VERIFIER")


def test_timeout_degrades_to_a_manual_url_never_a_dead_end(monkeypatch):
    monkeypatch.setattr(oo, "_open_browser", lambda url: False)

    class _Cap:
        url = "http://localhost:1/cb/n"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def wait(self):
            return None

    monkeypatch.setattr(oo.local_auth, "Capture", lambda **kw: _Cap())
    out = oo.acquire(timeout=0.01)
    assert out["status"] == "waiting"
    assert out["open_this_url"].startswith("https://openrouter.ai/auth")
    assert "key" not in out          # ⚠️ a value must never appear in a returned payload


def test_env_name_comes_from_the_registry_not_a_literal():
    """⚠️ The `credential-registry-is-the-one-list` guard caught a hardcoded
    OPENROUTER_API_KEY here. Deriving it keeps this from becoming a sixth copy of
    "which variable holds which credential"."""
    from opyt_core.credentials_registry import by_service
    assert oo.env_name() == by_service("openrouter").env
    assert "OPENROUTER_API_KEY" not in open(oo.__file__).read()
