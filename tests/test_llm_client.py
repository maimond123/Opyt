"""
tests/test_llm_client.py

Exercises pipeline/llm_client.py without making real network calls. Uses the
`_set_backend_for_tests` seam so backends can be replaced with deterministic fakes.

⚠️ EVERY `role="..."` HERE MUST BE DECLARED IN config/settings.example.yaml. `call()` reads the
ACTIVE settings.yaml, so a role name that exists only in the author's ~/.opyt/settings.yaml makes
this file pass on his machine and fail on a fresh install. Measured once: `chat_extract`,
`note_classify` and `synthesis_verify` had all outlived their callers and been dropped from the
template, and 5 of these 11 tests failed under a template-seeded OPYT_HOME with
`ValueError: role not declared`. The template is the contract; check a role against it, not
against whatever your own file happens to carry.

`vision` is the one role the template declares, and it declares no response_format. The test
that needs a role declaring one adds it to the config it reads (`_with_json_role`).
"""

from __future__ import annotations

import pytest

from pipeline import llm_client


# ── Fakes ────────────────────────────────────────────────────────────────────


def _fake_backend_text(text: str, in_tok: int = 100, out_tok: int = 50):
    """Build a backend fn that returns deterministic content. Accepts the optional
    response_format / images kwargs so it mirrors the real backend contract — a fake
    must tolerate whatever `call` threads, independent of which roles happen to declare
    response_format in settings.yaml (else adding json_object to a role breaks it)."""
    def _fn(model: str, system: str, user: str, max_tokens: int,
            response_format=None, images=None):
        return text, in_tok, out_tok, 0.42, {"raw": "mock"}
    return _fn


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _with_json_role(monkeypatch, name: str = "json_role") -> str:
    """The active config plus one role that declares `response_format: json_object`."""
    real = llm_client.load_yaml_config

    def config():
        cfg = real()
        backends = dict(cfg.get("llm_backends") or {})
        backends["roles"] = {**(backends.get("roles") or {}),
                             name: {"provider": "openrouter", "model": "test/json-model",
                                    "response_format": "json_object"}}
        return {**cfg, "llm_backends": backends}

    monkeypatch.setattr(llm_client, "load_yaml_config", config)
    return name


@pytest.fixture
def fake_openrouter():
    """Substitute a fake openrouter backend; restore afterward."""
    real = llm_client._BACKENDS["openrouter"]
    llm_client._set_backend_for_tests("openrouter", _fake_backend_text("hello"))
    yield
    llm_client._set_backend_for_tests("openrouter", real)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_the_backend_table_has_one_entry():
    """One backend: the direct Anthropic transport is retired.

    Asserted as an exact set, not `"openrouter" in ...`, on purpose: this is the test that fails
    if a second backend is registered without a matching credential-registry row. The dispatch
    table surviving at size one is deliberate — it is the registration seam for the next
    provider, not dead generality."""
    assert set(llm_client._BACKENDS) == {"openrouter"}


class TestRoleResolution:
    def test_unknown_role_raises(self, fake_openrouter):
        with pytest.raises(ValueError, match="not declared"):
            llm_client.call(role="totally_made_up", system="x", user="y")

    def test_known_role_dispatches_to_correct_backend(self, fake_openrouter):
        # vision is configured to openrouter in the shipped template
        resp = llm_client.call(role="vision", system="s", user="u")
        assert resp.text == "hello"
        assert resp.provider == "openrouter"

    def test_response_format_threaded_only_when_role_declares_it(self, monkeypatch):
        """A role with `response_format` threads it to the backend; a role without it
        calls the backend with no such kwarg (so 4-arg backends stay compatible)."""
        seen = {}

        def capturing(model, system, user, max_tokens, response_format=None):
            seen["response_format"] = response_format
            return "ok", 1, 1, 0.0, {}

        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", capturing)
        try:
            llm_client.call(role=_with_json_role(monkeypatch), system="s", user="u")
            assert seen["response_format"] == "json_object"
            # vision declares none → backend sees None (kwarg simply not passed)
            seen.clear()
            llm_client.call(role="vision", system="s", user="u")
            assert seen.get("response_format") is None
        finally:
            llm_client._set_backend_for_tests("openrouter", real)

    def test_images_threaded_only_when_supplied(self):
        """`images` reaches the backend only when passed (4-arg fakes stay valid)."""
        seen = {}

        def capturing(model, system, user, max_tokens, response_format=None, images=None):
            seen["images"] = images
            return "ok", 1, 1, 0.0, {}

        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", capturing)
        try:
            llm_client.call(role="vision", system="s", user="u", images=["http://x/i.png"])
            assert seen["images"] == ["http://x/i.png"]
            seen.clear()
            llm_client.call(role="vision", system="s", user="u")
            assert seen.get("images") is None
        finally:
            llm_client._set_backend_for_tests("openrouter", real)


class TestPerCallOverrides:
    def test_caller_can_override_model(self, fake_openrouter):
        resp = llm_client.call(
            role="vision",
            system="x", user="y",
            model="openai/gpt-4o-mini",  # override the configured model
        )
        assert resp.model == "openai/gpt-4o-mini"

    def test_caller_can_override_max_tokens(self, fake_openrouter):
        # We can't see max_tokens in the fake's output, but we can verify
        # the call completes without error when overriding.
        resp = llm_client.call(
            role="vision",
            system="x", user="y",
            max_tokens=512,
        )
        assert resp.text == "hello"
