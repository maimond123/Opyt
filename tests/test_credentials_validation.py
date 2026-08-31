"""
tests/test_credentials_validation.py

Credential validation is now LLM-provider-AGNOSTIC: it validates every provider the
configured roles reference, via the llm_client backend dispatch — no hardcoded vendor
and no direct Anthropic SDK. These tests pin that contract with fake backends (no
network), so they hold regardless of which provider settings.yaml points at.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from pipeline import credentials, llm_client, llm_providers


# ── Fakes ────────────────────────────────────────────────────────────────────


def _ok_backend(model, system, user, max_tokens, api_key=None):
    return "pong", 1, 1, 0.0, {"raw": "mock"}


def _bad_backend(model, system, user, max_tokens, api_key=None):
    raise RuntimeError("HTTP 401: invalid key")


@pytest.fixture
def fake_or_ok():
    real = llm_client._BACKENDS["openrouter"]
    llm_client._set_backend_for_tests("openrouter", _ok_backend)
    yield
    llm_client._set_backend_for_tests("openrouter", real)


# ── validate_provider (the agnostic core) ────────────────────────────────────


class TestValidateProvider:
    def test_liveness_success(self, fake_or_ok):
        ok, msg = llm_client.validate_provider("openrouter", "sk-test")
        assert ok and "valid" in msg

    def test_liveness_failure_surfaces_backend_error(self):
        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", _bad_backend)
        try:
            ok, msg = llm_client.validate_provider("openrouter", "sk-bad")
            assert not ok and "401" in msg
        finally:
            llm_client._set_backend_for_tests("openrouter", real)

    def test_unknown_provider_is_a_clean_failure(self):
        ok, _ = llm_client.validate_provider("not-a-provider", "x")
        assert not ok

    def test_passed_key_is_threaded_to_backend_not_into_env(self, monkeypatch):
        # Fix A: the key under test reaches the backend as a PARAMETER, and
        # os.environ is never touched — so a concurrent call() can't see it.
        seen = {}

        def capturing(model, system, user, max_tokens, api_key=None):
            seen["api_key"] = api_key
            return "pong", 1, 1, 0.0, {}

        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", capturing)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-original")
        try:
            ok, _ = llm_client.validate_provider("openrouter", "sk-candidate")
            assert ok
            assert seen["api_key"] == "sk-candidate"                  # threaded as a param
            assert os.environ["OPENROUTER_API_KEY"] == "sk-original"  # env untouched
        finally:
            llm_client._set_backend_for_tests("openrouter", real)

    def test_no_key_falls_back_to_env(self, fake_or_ok):
        # key=None → backend gets no api_key override and reads the env itself.
        seen = {}

        def capturing(model, system, user, max_tokens, api_key=None):
            seen["api_key"] = api_key
            return "pong", 1, 1, 0.0, {}

        real = llm_client._BACKENDS["openrouter"]
        llm_client._set_backend_for_tests("openrouter", capturing)
        try:
            llm_client.validate_provider("openrouter", None)
            assert seen["api_key"] is None        # no override passed
        finally:
            llm_client._set_backend_for_tests("openrouter", real)


# ── provider discovery from settings.yaml ────────────────────────────────────


class TestProviderDiscovery:
    def test_openrouter_is_configured(self):
        assert "openrouter" in llm_client.configured_providers()

    def test_default_provider_is_openrouter(self):
        assert llm_client.default_provider() == "openrouter"

    def test_known_providers_covers_the_backends(self):
        """ONE backend since the direct Anthropic transport was retired 2026-08-13.

        Asserted as an exact set, not `"openrouter" in ...`, on purpose: this is the test that
        fails if a second backend is registered without a matching `_PROVIDER_ENV` entry and a
        credential-registry row. The dispatch table surviving at size one is deliberate — it is
        the registration seam for the next provider, not dead generality."""
        assert llm_client.known_providers() == {"openrouter"}


# ── credentials routing ──────────────────────────────────────────────────────


class TestValidateCredentialAgnostic:
    def test_provider_service_routes_to_liveness(self, fake_or_ok):
        ok, _ = credentials.validate_credential("openrouter", "sk-test")
        assert ok

    def test_vendor_specific_validator_is_gone(self):
        # The hardcoded Anthropic validator was deleted under the OpenRouter migration.
        assert not hasattr(credentials, "_validate_anthropic")

    def test_no_direct_anthropic_or_haiku_in_source(self):
        # Migration completeness: credentials.py must not resurrect the SDK/model ref.
        src = pathlib.Path(credentials.__file__).read_text()
        assert "claude-haiku" not in src
        assert "anthropic.Anthropic" not in src
        assert "import anthropic" not in src


# ── validate_provider_status — 401 and 402 must stay apart ───────────────────

def test_validate_provider_status_surfaces_402(monkeypatch):
    """⚠️ THE STATUS IS THE WHOLE POINT. A dead key wants a NEW key; an unfunded account wants
    MONEY. Folding them sends an unfunded user back through OAuth to mint a second unfunded
    key, forever."""
    # ⚠️ Patches `llm_providers`, not `llm_client`. Both `validate_provider` and
    # `validate_provider_status` moved TOGETHER into `pipeline/llm_providers.py` (step 7 split,
    # 2026-08-16); `validate_provider_status` calls `validate_provider` by bare name, resolved in
    # `llm_providers`'s own namespace, so a patch applied only to `llm_client`'s forwarding
    # re-export would never be seen by that internal call — the real (network-hitting)
    # `validate_provider` would run instead.
    monkeypatch.setattr(llm_providers, "validate_provider",
                        lambda p, k=None, **kw: (False, "openrouter key validation failed: "
                                                        "HTTP 402 insufficient credits"))
    ok, msg, status = llm_providers.validate_provider_status("openrouter", "sk-x")
    assert ok is False and status == 402


def test_validate_provider_status_is_none_when_nothing_to_go_on(monkeypatch):
    """None is HONEST, not a failure — the caller must then name both possibilities."""
    monkeypatch.setattr(llm_providers, "validate_provider",
                        lambda p, k=None, **kw: (False, "connection reset by peer"))
    ok, msg, status = llm_providers.validate_provider_status("openrouter", "sk-x")
    assert ok is False and status is None


def test_validate_provider_status_passes_success_through(monkeypatch):
    monkeypatch.setattr(llm_providers, "validate_provider",
                        lambda p, k=None, **kw: (True, "openrouter key is valid"))
    assert llm_providers.validate_provider_status("openrouter", "sk-x") == (
        True, "openrouter key is valid", None)
