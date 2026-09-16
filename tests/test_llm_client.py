"""
tests/test_llm_client.py

Exercises pipeline/llm_client.py without making real network calls. Uses the
`_set_backend_for_tests` seam so backends can be replaced with deterministic fakes.

⚠️ EVERY `role="..."` HERE MUST BE DECLARED IN config/settings.example.yaml. `call()` reads the
ACTIVE settings.yaml, so a role name that exists only in the author's ~/.opyt/settings.yaml makes
this file pass on his machine and fail on a fresh install. Measured 2026-08-30: `chat_extract`,
`note_classify` and `synthesis_verify` had all outlived their callers and been dropped from the
template, and 5 of these 11 tests failed under a template-seeded OPYT_HOME with
`ValueError: role not declared`. The template is the contract; check a role against it, not
against whatever your own file happens to carry.

The three used below are chosen for the properties the assertions need, not for their meaning:
`entity_classify`, `content_quality` (declares response_format), and `vision` (declares none).
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


@pytest.fixture
def fake_openrouter():
    """Substitute a fake openrouter backend; restore afterward."""
    real = llm_client._BACKENDS["openrouter"]
    llm_client._set_backend_for_tests("openrouter", _fake_backend_text("hello"))
    yield
    llm_client._set_backend_for_tests("openrouter", real)


# The `fake_anthropic` fixture was DELETED here 2026-08-13 with the direct Anthropic backend.
# Nothing replaced it: the one test that took it asserted `provider == "openrouter"` on both of
# its calls, so the fixture was substituting a backend the assertions never reached.


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRoleResolution:
    def test_unknown_role_raises(self, fake_openrouter):
        with pytest.raises(ValueError, match="not declared"):
            llm_client.call(role="totally_made_up", system="x", user="y")

    def test_known_role_dispatches_to_correct_backend(self, fake_openrouter):
        # entity_classify is configured to openrouter in the shipped template
        resp = llm_client.call(role="entity_classify", system="s", user="u")
        assert resp.text == "hello"
        assert resp.provider == "openrouter"
        # vision is now configured to openrouter (Gemini multimodal)
        resp2 = llm_client.call(role="vision", system="s", user="u")
        assert resp2.text == "hello"
        assert resp2.provider == "openrouter"

    def test_response_format_threaded_only_when_role_declares_it(self):
        """A role with `response_format` threads it to the backend; a role without it
        calls the backend with no such kwarg (so 4-arg backends stay compatible)."""
        seen = {}

        def capturing(model, system, user, max_tokens, response_format=None):
            seen["response_format"] = response_format
            return "ok", 1, 1, 0.0, {}

        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", capturing)
        try:
            # content_quality declares response_format: json_object in the shipped template
            llm_client.call(role="content_quality", system="s", user="u")
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
            role="entity_classify",
            system="x", user="y",
            model="openai/gpt-4o-mini",  # override the configured model
        )
        assert resp.model == "openai/gpt-4o-mini"

    def test_caller_can_override_max_tokens(self, fake_openrouter):
        # We can't see max_tokens in the fake's output, but we can verify
        # the call completes without error when overriding.
        resp = llm_client.call(
            role="entity_classify",
            system="x", user="y",
            max_tokens=512,
        )
        assert resp.text == "hello"
