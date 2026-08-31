"""
tests/test_llm_client.py

Exercises pipeline/llm_client.py without making real network calls. Uses the
test seams (`_set_backend_for_tests`, `_override_stats_file_for_tests`) so
backends can be replaced with deterministic fakes.

⚠️ EVERY `role="..."` HERE MUST BE DECLARED IN config/settings.example.yaml. `call()` reads the
ACTIVE settings.yaml, so a role name that exists only in the author's ~/.opyt/settings.yaml makes
this file pass on his machine and fail on a fresh install. Measured 2026-08-30: `chat_extract`,
`note_classify` and `synthesis_verify` had all outlived their callers and been dropped from the
template, and 5 of these 11 tests failed under a template-seeded OPYT_HOME with
`ValueError: role not declared`. The template is the contract; check a role against it, not
against whatever your own file happens to carry.

The three used below are chosen for the properties the assertions need, not for their meaning:
`entity_classify` (llama, so the pricing rows resolve), `content_quality` (declares
response_format), `vision` (gemini, and declares none).
"""

from __future__ import annotations

import json
from pathlib import Path

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
def isolated_stats(tmp_path: Path):
    stats_path = tmp_path / "api_stats.json"
    llm_client._override_stats_file_for_tests(stats_path)
    yield stats_path
    llm_client._override_stats_file_for_tests(None)


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


class TestCostMath:
    def test_known_model_pricing(self):
        # Exactly 1M of each token kind, so the cost IS the row: proves the per-million
        # arithmetic without hardcoding a price. A hardcoded figure here broke on 2026-08-28
        # when the llama row was corrected from a 5.5x-low estimate to the catalog's number —
        # the test was pinning the stale price, not the math.
        from pipeline import llm_spend
        pin, pout = llm_spend._PRICING["meta-llama/llama-3.3-70b-instruct"]
        cost = llm_client.cost_for("meta-llama/llama-3.3-70b-instruct", 1_000_000, 1_000_000)
        assert cost == pytest.approx(pin + pout, rel=1e-6)

    def test_unknown_model_zero_cost(self):
        # Unknown models shouldn't raise — they should report $0 so the rest
        # of the pipeline keeps working. We just lose cost visibility.
        assert llm_client.cost_for("never-heard-of-this", 1000, 1000) == 0.0


class TestRoleResolution:
    def test_unknown_role_raises(self, fake_openrouter):
        with pytest.raises(ValueError, match="not declared"):
            llm_client.call(role="totally_made_up", system="x", user="y")

    def test_known_role_dispatches_to_correct_backend(self, fake_openrouter, isolated_stats):
        # entity_classify is configured to openrouter in the shipped template
        resp = llm_client.call(role="entity_classify", system="s", user="u")
        assert resp.text == "hello"
        assert resp.provider == "openrouter"
        # vision is now configured to openrouter (Gemini multimodal)
        resp2 = llm_client.call(role="vision", system="s", user="u")
        assert resp2.text == "hello"
        assert resp2.provider == "openrouter"

    def test_response_format_threaded_only_when_role_declares_it(self, isolated_stats):
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

    def test_images_threaded_only_when_supplied(self, isolated_stats):
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


class TestStatsAccumulation:
    def test_lifetime_totals_grow(self, fake_openrouter, isolated_stats):
        llm_client.call(role="entity_classify", system="x", user="y")
        llm_client.call(role="entity_classify", system="x", user="y")
        llm_client.flush_stats()
        stats = json.loads(isolated_stats.read_text())
        assert stats["lifetime"]["calls"] == 2
        assert stats["lifetime"]["input_tokens"] == 200
        assert stats["lifetime"]["output_tokens"] == 100
        # The two calls' tokens priced off the live table — derived, not hardcoded, so a price
        # correction cannot fail a test that is really about the totals accumulating.
        expected = llm_client.cost_for("meta-llama/llama-3.3-70b-instruct", 200, 100)
        assert stats["lifetime"]["cost_usd"] == pytest.approx(expected, rel=1e-3)

    def test_by_model_and_by_role_partitioned(self, fake_openrouter, isolated_stats):
        llm_client.call(role="entity_classify", system="x", user="y")
        llm_client.call(role="vision", system="x", user="y")  # vision → openrouter now
        llm_client.flush_stats()
        stats = json.loads(isolated_stats.read_text())
        assert "meta-llama/llama-3.3-70b-instruct" in stats["by_model"]
        assert "google/gemini-2.5-flash" in stats["by_model"]
        assert stats["by_role"]["entity_classify"]["calls"] == 1
        assert stats["by_role"]["vision"]["calls"] == 1

    def test_pricing_version_persisted(self, fake_openrouter, isolated_stats):
        llm_client.call(role="entity_classify", system="x", user="y")
        llm_client.flush_stats()
        stats = json.loads(isolated_stats.read_text())
        assert stats["pricing_version"] == llm_client.PRICING_TABLE_VERSION


class TestPerCallOverrides:
    def test_caller_can_override_model(self, fake_openrouter, isolated_stats):
        resp = llm_client.call(
            role="entity_classify",
            system="x", user="y",
            model="openai/gpt-4o-mini",  # override the configured model
        )
        assert resp.model == "openai/gpt-4o-mini"

    def test_caller_can_override_max_tokens(self, fake_openrouter, isolated_stats):
        # We can't see max_tokens in the fake's output, but we can verify
        # the call completes without error when overriding.
        resp = llm_client.call(
            role="entity_classify",
            system="x", user="y",
            max_tokens=512,
        )
        assert resp.text == "hello"
