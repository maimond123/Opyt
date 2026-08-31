"""Model routing — provider redundancy as an enforced invariant, in three layers.

WHY (2026-08-01): `google/gemma-3-4b-it` was the OCR model, and OpenRouter serves it from exactly
ONE upstream — DeepInfra. The moment DeepInfra hit the deny-list (`7eb36301`, a CHAT-path latency
fix applied globally) every OCR call returned `404 All providers have been ignored`. A 100%-failure
stage, silent for weeks because the cascade degrades fail-safe, and expensive because 18 straight
failures opened the SHARED `openrouter` breaker and stalled `content_quality` behind it.

No offline test could have caught the original: availability is a fact about OpenRouter's LIVE
catalog crossed with the ACTIVE deny-list, and neither is in the repo. What these tests CAN pin is
the machinery that turns that fact into a loud, early failure instead of a silent one:

  layer 1  the registry is complete (guard-enforced elsewhere) and includes ROLE models
  layer 2  preflight distinguishes dead / fragile / unknown, and counts ORGS not endpoints
  layer 3  a "no providers" 404 is TERMINAL — it never trips the shared breaker, and the OCR
           stage disables itself instead of paying a doomed round-trip per image
"""
from __future__ import annotations

import pytest

from pipeline import model_routing as mr


# ── layer 2: orgs, not endpoints ──────────────────────────────────────────────

def test_org_collapses_vendor_endpoint_names():
    """`gemini-2.5-flash-lite` advertises five endpoints that are Google, Google, Google AI
    Studio... — five endpoints and ONE company. Counting endpoints would have scored it the most
    redundant option on the board while being exactly as fragile as the model that broke."""
    assert mr._org_of("Google") == "google"
    assert mr._org_of("Google AI Studio") == "google"
    assert mr._org_of("Google Vertex") == "google"
    assert mr._org_of("Amazon Bedrock") == "amazon"
    assert len({mr._org_of(p) for p in ("Parasail", "Nebius", "Novita", "Phala")}) == 4


def _fake_endpoints(monkeypatch, providers):
    """Stub the OpenRouter endpoints API with a given provider list."""
    import io
    import json as _json

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _open(req, timeout=30):
        body = {"data": {"endpoints": [{"provider_name": p} for p in providers]}}
        return _Resp(_json.dumps(body).encode())
    monkeypatch.setattr(mr.urllib.request, "urlopen", _open)


def test_denied_providers_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _fake_endpoints(monkeypatch, ["DeepInfra", "Nebius", "Novita"])
    assert mr.surviving_orgs("x/y", ["DeepInfra"], use_cache=False) == ["nebius", "novita"]


def test_single_provider_model_reads_as_dead_under_its_deny(monkeypatch, tmp_path):
    """The exact 2026-08-01 shape: one provider, and it is the denied one."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _fake_endpoints(monkeypatch, ["DeepInfra"])
    assert mr.surviving_orgs("google/gemma-3-4b-it", ["DeepInfra"], use_cache=False) == []


def test_unknown_is_not_the_same_as_dead(monkeypatch, tmp_path):
    """None (could not determine) must stay distinct from [] (definitively nothing). Collapsing
    them would let a flaky network or a missing key abort an onboarding — preflight must never be
    the reason a run cannot start."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert mr.surviving_orgs("x/y", ["DeepInfra"], use_cache=False) is None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def _boom(req, timeout=30):
        raise OSError("network down")
    monkeypatch.setattr(mr.urllib.request, "urlopen", _boom)
    assert mr.surviving_orgs("x/y", ["DeepInfra"], use_cache=False) is None


def test_cache_key_includes_the_deny_list():
    """A deny-list edit is precisely what invalidated the previously-true answer. A cache that
    outlived the edit would re-assert the stale verdict at the moment it stopped being true."""
    a = mr._cache_key("m", ["DeepInfra"])
    b = mr._cache_key("m", ["DeepInfra", "Cloudflare"])
    assert a != b
    assert mr._cache_key("m", ["B", "A"]) == mr._cache_key("m", ["A", "B"])   # order-insensitive


# ── layer 2: the verdict ──────────────────────────────────────────────────────

def test_preflight_blocks_on_dead_and_only_warns_on_fragile():
    models = {"dead/model": "ocr", "fragile/model": "chart", "fine/model": "chat"}
    orgs = {"dead/model": [], "fragile/model": ["google"], "fine/model": ["a", "b", "c"]}

    import unittest.mock as m           # inject the lookup rather than hitting the network
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: orgs[mm]):
        rep = mr.preflight(models, deny=["DeepInfra"])

    assert rep["ok"] is False                                  # a dead model blocks
    assert [x[0] for x in rep["dead"]] == ["dead/model"]
    assert [x[0] for x in rep["fragile"]] == ["fragile/model"]  # warns, does not block
    assert rep["unknown"] == []


def test_preflight_unknown_does_not_block():
    import unittest.mock as m
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: None):
        rep = mr.preflight({"a/b": "ocr"}, deny=[])
    assert rep["ok"] is True and [x[0] for x in rep["unknown"]] == ["a/b"]


def test_preflight_dead_ocr_with_live_fallback_does_not_block():
    """§1 and §2 must not fight: when the primary OCR model is dead but a declared fallback
    survives, the cascade substitutes and reports — flipping `ok` would halt every rail on a
    failure the fallback chain exists to absorb. The dead model still shows in `dead`."""
    primary, fallback = mr.OCR_FALLBACKS[0], mr.OCR_FALLBACKS[1]
    models = {primary: "OCR", fallback: "OCR fallback", "embed/x": "emb"}
    orgs = {primary: [], fallback: ["amazon"], "embed/x": ["a", "b"]}

    import unittest.mock as m
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: orgs[mm]):
        rep = mr.preflight(models, deny=["DeepInfra"])
    assert rep["ok"] is True
    assert [x[0] for x in rep["dead"]] == [primary]     # reported, not hidden


def test_preflight_whole_dead_ocr_chain_blocks():
    """With every declared OCR candidate dead there is nothing excusable left — block."""
    primary, fallback = mr.OCR_FALLBACKS[0], mr.OCR_FALLBACKS[1]
    models = {primary: "OCR", fallback: "OCR fallback"}

    import unittest.mock as m
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: []):
        rep = mr.preflight(models, deny=["DeepInfra"])
    assert rep["ok"] is False


def test_fetch_false_answers_from_cache_alone(monkeypatch, tmp_path):
    """`fetch=False` is the interactive-surface contract (the `oracle` screen): a cache miss is
    `unknown`, NEVER a network round-trip inside a tool call."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))          # empty cache
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")    # a key must not tempt a fetch

    def _bomb(req, timeout=30):
        raise AssertionError("fetch=False touched the network")
    monkeypatch.setattr(mr.urllib.request, "urlopen", _bomb)

    assert mr.surviving_orgs("x/y", ["DeepInfra"], fetch=False) is None
    rep = mr.preflight({"x/y": "role:test"}, deny=["DeepInfra"], fetch=False)
    assert rep["ok"] is True and [x[0] for x in rep["unknown"]] == ["x/y"]


def test_registered_models_includes_settings_roles(monkeypatch):
    """Regression: an earlier draft imported the settings accessor lazily inside a broad
    `except Exception: pass`. The name was wrong, so EVERY role model was silently skipped and the
    preflight looked healthy while checking a third of what it claimed to."""
    monkeypatch.setattr(mr, "settings", lambda: {
        "llm_backends": {"roles": {
            "content_quality": {"provider": "openrouter", "model": "vendor/chat-model"},
            "legacy": {"provider": "anthropic", "model": "should-be-ignored"}}}})
    reg = mr.registered_models()
    assert "vendor/chat-model" in reg and reg["vendor/chat-model"] == "role:content_quality"
    assert "should-be-ignored" not in reg          # non-openrouter roles aren't OpenRouter-routed
    assert mr.OCR_MODEL in reg and mr.EMBED_MODEL in reg


# ── layer 2: fallback resolution is REPORTED, never inferred ──────────────────

def test_fallback_is_used_and_announced():
    import unittest.mock as m
    primary, fallback = mr.OCR_FALLBACKS[0], mr.OCR_FALLBACKS[1]
    with m.patch.object(mr, "surviving_orgs",
                        side_effect=lambda mm, d, **kw: [] if mm == primary else ["amazon"]):
        model, reason = mr.resolve_ocr_model(deny=["DeepInfra"])
    assert model == fallback
    assert "FALLBACK" in reason and primary in reason   # the swap must be visible in the report


def test_all_dead_returns_no_model_rather_than_guessing():
    import unittest.mock as m
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: []):
        model, reason = mr.resolve_ocr_model(deny=["DeepInfra"])
    assert model is None                       # refuse to guess an unvetted model
    assert "no declared OCR model is routable" in reason


def test_unknown_availability_resolves_to_the_primary():
    """Preflight must never be the reason a run cannot start."""
    import unittest.mock as m
    with m.patch.object(mr, "surviving_orgs", side_effect=lambda mm, d, **kw: None):
        model, reason = mr.resolve_ocr_model(deny=[])
    assert model == mr.OCR_FALLBACKS[0] and "unknown" in reason


# ── layer 3: a "no providers" 404 is TERMINAL ────────────────────────────────

def test_unroutable_404_raises_its_own_type():
    import urllib.error

    from pipeline import llm_client

    class _E(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 404, "Not Found", {}, None)

        def read(self):
            return b'{"error":{"message":"All providers have been ignored.","code":404}}'

    import unittest.mock as m
    with m.patch.object(llm_client.urllib.request, "urlopen", side_effect=_E()):
        with pytest.raises(llm_client.ModelUnroutableError):
            llm_client._http_json(object())


def test_ordinary_404_stays_a_plain_backend_error():
    """Only the 'no permitted provider' body is terminal — a generic 404 must keep its existing
    transient handling, or the breaker stops protecting against real outages."""
    import urllib.error

    from pipeline import llm_client

    class _E(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 404, "Not Found", {}, None)

        def read(self):
            return b'{"error":{"message":"No endpoints found for model xyz","code":404}}'

    import unittest.mock as m
    with m.patch.object(llm_client.urllib.request, "urlopen", side_effect=_E()):
        with pytest.raises(llm_client._BackendError) as ei:
            llm_client._http_json(object())
    assert not isinstance(ei.value, llm_client.ModelUnroutableError)


def test_unroutable_does_not_trip_the_shared_breaker(tmp_path, monkeypatch):
    """The load-bearing one. 18 of these opened the shared `openrouter` breaker in a single run,
    fail-fasting `content_quality` behind a stage that could never succeed."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    from pipeline.circuit_breaker import CircuitBreaker
    from pipeline.llm_client import ModelUnroutableError

    br = CircuitBreaker("test-openrouter", threshold=3)

    def _boom():
        raise ModelUnroutableError("HTTP 404: All providers have been ignored", status=404)

    for _ in range(10):
        with pytest.raises(ModelUnroutableError):
            br.call(_boom, ignore=(ModelUnroutableError,))
    assert br.allow() is True          # still closed after 10 terminal failures


def test_ocr_stage_disables_itself_instead_of_retrying(monkeypatch):
    """One doomed round-trip, not one per image. The old behavior paid a network call for every
    image in the run and logged an indistinguishable failure each time."""
    from pipeline import llm_client
    from pipeline import ocr_cascade as oc

    oc._reset_stage_for_tests()          # clears the conftest pin → pin resolution explicitly
    monkeypatch.setattr(mr, "resolve_ocr_model", lambda **kw: (mr.OCR_MODEL, "pinned"))
    calls = {"n": 0}

    def _unroutable(**kw):
        calls["n"] += 1
        raise llm_client.ModelUnroutableError("HTTP 404: All providers have been ignored")
    monkeypatch.setattr(oc.llm_client, "call", _unroutable)

    assert oc.read_image("https://img/1.png") is None
    assert oc.read_image("https://img/2.png") is None
    assert oc.read_image("https://img/3.png") is None
    assert calls["n"] == 1                       # only the FIRST image paid a round-trip
    assert "unroutable" in (oc.stage_status() or "")
    oc._reset_stage_for_tests()


def test_ordinary_ocr_failure_does_not_disable_the_stage(monkeypatch):
    """A transient per-image failure must stay per-image — one bad image can't kill the run."""
    from pipeline import ocr_cascade as oc

    oc._reset_stage_for_tests()          # clears the conftest pin → pin resolution explicitly
    monkeypatch.setattr(mr, "resolve_ocr_model", lambda **kw: (mr.OCR_MODEL, "pinned"))
    calls = {"n": 0}

    def _flaky(**kw):
        calls["n"] += 1
        raise RuntimeError("timeout")
    monkeypatch.setattr(oc.llm_client, "call", _flaky)

    assert oc.read_image("https://img/1.png") is None
    assert oc.read_image("https://img/2.png") is None
    assert calls["n"] == 2                       # each image still tried
    assert oc.stage_status() is None
    oc._reset_stage_for_tests()
