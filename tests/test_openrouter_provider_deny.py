"""OpenRouter upstream routing — deny what is broken, then rank what is left by throughput.

The two halves do different jobs and BOTH are needed. `ignore` is a snapshot we maintain by hand;
`sort` is a policy OpenRouter re-evaluates live. Proof that the deny-list alone is insufficient
(2026-07-31, 6 calls per arm, interleaved): banning DeepInfra moved the tail to AkashML (p50 2.28 s,
max 10.49 s) while `sort: "throughput"` collapsed it outright (p50 0.79 s, max 1.20 s). Slowness was
never a property of one provider — it was a property of not selecting on speed at all.

Two rules the tests below pin, because both fail SILENTLY:
  - the sort must yield to a caller's explicit `order` (embed's Nebius-over-SiliconFlow preference
    is about fp8 precision, which a speed ranking has no view on);
  - the sort must not clobber `require_parameters` or `ignore`, the same merge hazard the deny-list
    had.

—— the deny-list's own evidence ——

Why this file exists (the evidence, so a future reader does not re-derive it):

  DeepInfra — denied for LATENCY. Reproduced on two models a week apart, so it is a property of the
  provider, not one bad afternoon:
    2026-07-22  qwen/qwen3-embedding-8b   71–93 s on 5/5 calls, 99.99 % uptime claimed
    2026-07-29  meta-llama/llama-3.3-70b  8.4–23.7 s on 10/10 calls, while every other upstream
                                          answered the identical prompt in under 2.9 s

  Cloudflare — denied for CORRECTNESS, which is the more dangerous failure. 2026-07-29,
  llama-3.3-70b, 8/8 calls: `finish_reason: "tool_calls"` with `message.content == ""` while billing
  70 completion tokens. `llm_client` reads message.content, so the batch looks like an empty answer,
  `content_gate._parse_verdicts("")` returns {}, and the page degrades to keep-all — ungraded
  boilerplate enters the KB, paid for, with no error raised. It also defeats `require_parameters`.

The 2026-07-22 fix used `provider.order` + `allow_fallbacks: true` and its comment claimed DeepInfra
was "excluded". `order` only RANKS; a fallback could still land there. The hard exclusion is `ignore`,
and these tests pin that it is present on both OpenRouter surfaces (chat + embeddings).

The load-bearing test is `test_json_mode_require_parameters_survives`: the chat path already set
`body["provider"] = {"require_parameters": True}`, so a deny-list that ASSIGNS instead of MERGING
would silently drop the JSON-mode guarantee. That failure is invisible — no error, just a slow
return to the malformed-verdict bug `require_parameters` was added to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from opyt_core import config as core_config
from pipeline import llm_client


# ── helpers ──────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cold_routing_cache():
    """The routing policy is memoized per process (it sits in the hot path of every call). Without a
    reset, whichever test ran first would pin the value for the rest of the file and the
    settings-override tests would pass or fail depending on collection order."""
    core_config._reset_routing_cache_for_tests()
    yield
    core_config._reset_routing_cache_for_tests()


class _PassthroughBreaker:
    """The real breaker persists state in opyt.db; tests must not touch it.

    Signature MIRRORS CircuitBreaker.call, `ignore` included: a double that drifts from the real
    interface stops being a stand-in and starts being a second, wrong implementation."""

    def call(self, fn, *, ignore: tuple = ()):
        try:
            return fn()
        except ignore:
            raise


@pytest.fixture
def captured_chat(monkeypatch):
    """Run the REAL _call_openrouter_sync but capture the request instead of sending it, so the
    assertions are about the body that would actually go on the wire."""
    seen: dict = {}

    def fake_http_json(req, timeout=180.0):
        seen["body"] = __import__("json").loads(req.data.decode())
        return ({"choices": [{"message": {"content": "ok"}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}, 0.01)

    monkeypatch.setattr(llm_client, "_http_json", fake_http_json)
    monkeypatch.setattr(llm_client, "_openrouter_breaker", lambda: _PassthroughBreaker())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return seen


# ── the deny-list itself ─────────────────────────────────────────────────────────


def test_default_denies_known_broken_upstreams():
    """DeepInfra is denied for LATENCY, Cloudflare for CORRECTNESS (returns finish_reason
    tool_calls with empty message.content while billing tokens — 8/8 calls, 2026-07-29)."""
    deny = core_config.openrouter_deny_upstreams()
    assert "DeepInfra" in deny
    assert "Cloudflare" in deny


def test_merge_adds_ignore_to_empty_prefs():
    assert core_config.merge_provider_routing({})["ignore"] == ["DeepInfra", "Cloudflare"]
    assert core_config.merge_provider_routing()["ignore"] == ["DeepInfra", "Cloudflare"]


def test_merge_preserves_caller_keys():
    """An `order` preference and `allow_fallbacks` must survive — embed relies on both to pick
    Nebius (full precision, 4x cheaper) over SiliconFlow (fp8)."""
    out = core_config.merge_provider_routing(
        {"order": ["Nebius", "SiliconFlow"], "allow_fallbacks": True})
    assert out["order"] == ["Nebius", "SiliconFlow"]
    assert out["allow_fallbacks"] is True
    assert "DeepInfra" in out["ignore"]


def test_merge_unions_with_existing_ignore_without_duplicating():
    out = core_config.merge_provider_routing({"ignore": ["SomeOther"]})
    assert out["ignore"] == ["SomeOther", "DeepInfra", "Cloudflare"]
    twice = core_config.merge_provider_routing({"ignore": ["DeepInfra"]})
    assert twice["ignore"] == ["DeepInfra", "Cloudflare"], "must not duplicate a denied upstream"


def test_merge_does_not_mutate_callers_dict():
    base = {"require_parameters": True}
    core_config.merge_provider_routing(base)
    assert base == {"require_parameters": True}, "caller's dict must be left alone"


# ── the throughput sort ──────────────────────────────────────────────────────────


def test_sort_defaults_to_throughput():
    """Default routing picks the CHEAPEST upstream, which is what produced the tail. The deny-list
    alone does not fix it — banning DeepInfra just relocated the tail to AkashML (10.49 s max)."""
    assert core_config.merge_provider_routing({})["sort"] == "throughput"


def test_sort_yields_to_an_explicit_order():
    """THE Fork-1 guard. An `order` from the call site states a preference the sort cannot see:
    embed's order picks Nebius over SiliconFlow for full precision, and a throughput ranking would
    happily choose the fp8 upstream — writing quantized vectors into a full-precision index, which
    degrades similarity with no error raised anywhere."""
    out = core_config.merge_provider_routing({"order": ["Nebius", "SiliconFlow"]})
    assert "sort" not in out, "throughput sort overrode an explicit precision preference"
    assert out["order"] == ["Nebius", "SiliconFlow"]
    # An EMPTY order is not a preference — it must not suppress the sort.
    assert core_config.merge_provider_routing({"order": []})["sort"] == "throughput"


def test_sort_does_not_clobber_require_parameters_or_ignore():
    """Same merge hazard the deny-list had: three keys now share one `provider` block, and dropping
    any of them is invisible at runtime."""
    out = core_config.merge_provider_routing({"require_parameters": True, "ignore": ["SomeOther"]})
    assert out["require_parameters"] is True
    assert out["ignore"] == ["SomeOther", "DeepInfra", "Cloudflare"]
    assert out["sort"] == "throughput"


def test_caller_sort_wins():
    """A call site that names its own policy keeps it — the default is a default, not an override."""
    assert core_config.merge_provider_routing({"sort": "latency"})["sort"] == "latency"


def test_sort_survives_a_disabled_denylist(monkeypatch):
    """The two knobs are independent. The original merge returned early when the deny-list was
    empty — which would have skipped the sort entirely for anyone who turned the deny-list off."""
    monkeypatch.setattr(core_config, "settings", lambda: {"openrouter": {"deny_upstreams": []}})
    core_config._reset_routing_cache_for_tests()
    out = core_config.merge_provider_routing({})
    assert out == {"sort": "throughput"}


# ── resolution: overrides, memoization, fail-safe ────────────────────────────────


def test_settings_can_override_and_disable(monkeypatch):
    monkeypatch.setattr(core_config, "settings",
                        lambda: {"openrouter": {"deny_upstreams": ["Foo", "Bar"]}})
    assert core_config.openrouter_deny_upstreams() == ["Foo", "Bar"]
    # An explicit empty list is a real choice (disable), not a missing key.
    monkeypatch.setattr(core_config, "settings", lambda: {"openrouter": {"deny_upstreams": []}})
    # The resolution is memoized per process, so a mid-test config change must invalidate it. This
    # is the intended contract, not a workaround: at runtime config is read once at startup.
    core_config._reset_routing_cache_for_tests()
    assert core_config.openrouter_deny_upstreams() == []
    assert "ignore" not in core_config.merge_provider_routing({})


def test_resolution_is_memoized(monkeypatch):
    """The routing policy sits in the hot path of every LLM call and every embed batch, so it must
    not re-read settings.yaml per request — that was disk I/O + a YAML parse per call, and the GIL
    release mid-batch reordered embed's concurrent slices. ONE read covers both knobs: resolving the
    deny-list and the sort separately would have doubled the I/O this memo exists to remove."""
    calls = {"n": 0}

    def counting_settings():
        calls["n"] += 1
        return {}

    monkeypatch.setattr(core_config, "settings", counting_settings)
    core_config._reset_routing_cache_for_tests()
    for _ in range(25):
        core_config.merge_provider_routing({"require_parameters": True})
    assert calls["n"] == 1, f"settings.yaml re-read {calls['n']}x — the memo is not holding"


def test_unreadable_config_keeps_the_guard(monkeypatch):
    """Fail-safe direction: losing the deny-list silently is the expensive failure, so a broken
    config keeps the built-in default rather than routing everywhere again. Same for the sort —
    dropping it restores the measured tail, and neither loss raises anything."""
    def boom():
        raise OSError("no config")
    monkeypatch.setattr(core_config, "settings", boom)
    assert "DeepInfra" in core_config.openrouter_deny_upstreams()
    assert core_config.merge_provider_routing({})["sort"] == "throughput"


# ── surface 1: chat completions (llm_client) ─────────────────────────────────────


def test_chat_request_routes_on_deny_and_sort(captured_chat):
    """`vision` declares no response_format, so before this change its body carried NO provider
    block at all — this asserts routing reaches non-JSON roles too. The sort is inert for `vision`
    (gemini-2.5-flash is Google-only, so there is no pool to rank) and is sent anyway: a role→policy
    map to suppress a no-op would be upkeep for zero gain."""
    llm_client.call("vision", system="", user="hi")
    prefs = captured_chat["body"]["provider"]
    assert prefs["ignore"] == ["DeepInfra", "Cloudflare"]
    assert prefs["sort"] == "throughput"


def test_json_mode_require_parameters_survives(captured_chat):
    """THE regression guard. `content_quality` sets response_format=json_object, which sets
    require_parameters. All three keys must coexist — a routing merge that replaced the dict would
    drop the JSON-mode guarantee with no error and no log line."""
    llm_client.call("content_quality", system="s", user="u")
    prefs = captured_chat["body"]["provider"]
    assert prefs["require_parameters"] is True, "JSON-mode guarantee was clobbered by routing"
    assert "DeepInfra" in prefs["ignore"]
    assert prefs["sort"] == "throughput"
    assert captured_chat["body"]["response_format"] == {"type": "json_object"}


def test_chat_omits_provider_block_when_routing_disabled(captured_chat, monkeypatch):
    """Negative control: with BOTH knobs off, a non-JSON role sends no provider block — i.e. these
    tests fail if routing is not doing the work, rather than passing either way. Both must be
    disabled now; leaving the sort on would leave a provider block behind and blunt the control."""
    monkeypatch.setattr(core_config, "settings",
                        lambda: {"openrouter": {"deny_upstreams": [], "sort": None}})
    llm_client.call("vision", system="", user="hi")
    assert "provider" not in captured_chat["body"]


# ── surface 2: embeddings (kb.embed) ─────────────────────────────────────────────


def test_embed_request_denies_deepinfra(monkeypatch):
    """The embed path had `order` but no `ignore`, so a fallback could still reach DeepInfra. The
    order preference must survive alongside the new hard exclusion."""
    from pipeline.kb import embed as embed_mod

    seen: dict = {}

    def fake_post(req, timeout):
        seen["body"] = __import__("json").loads(req.data.decode())
        n = len(seen["body"]["input"])
        return {"data": [{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(n)]}

    monkeypatch.setattr(embed_mod, "_http_post_json", fake_post)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda _p: "test-key")

    emb = embed_mod.HostedEmbedder(dict(embed_mod._DEFAULTS), use_breaker=False)
    emb._dim = 3
    emb._embed_batch(["hello"])

    prefs = seen["body"]["provider"]
    assert "DeepInfra" in prefs["ignore"], "embeddings could still fall back to DeepInfra"
    assert prefs["order"] == ["Nebius", "SiliconFlow"], "order preference must survive the merge"
    assert prefs["allow_fallbacks"] is True
    # The Fork-1 rule, asserted on the real surface it protects: a throughput sort here could pick
    # SiliconFlow (fp8, 4x price) over Nebius while Nebius is healthy, quietly mixing quantized
    # vectors into a full-precision index.
    assert "sort" not in prefs, "throughput sort leaked onto the embedding surface"


def test_embed_denylist_survives_a_settings_override(monkeypatch):
    """A user override of `embeddings.provider_routing` replaces the whole routing block. The
    deny-list is merged at REQUEST time so an override cannot silently drop the guard."""
    from pipeline.kb import embed as embed_mod

    seen: dict = {}

    def fake_post(req, timeout):
        seen["body"] = __import__("json").loads(req.data.decode())
        return {"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}

    monkeypatch.setattr(embed_mod, "_http_post_json", fake_post)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda _p: "test-key")

    cfg = dict(embed_mod._DEFAULTS)
    cfg["provider_routing"] = {"order": ["SomethingElse"]}      # user wiped the original block
    emb = embed_mod.HostedEmbedder(cfg, use_breaker=False)
    emb._dim = 3
    emb._embed_batch(["hello"])

    assert "DeepInfra" in seen["body"]["provider"]["ignore"]
    assert seen["body"]["provider"]["order"] == ["SomethingElse"]


def test_embed_vectors_still_returned(monkeypatch):
    """Sanity: the routing change must not disturb the vector assembly contract."""
    from pipeline.kb import embed as embed_mod

    def fake_post(req, timeout):
        body = __import__("json").loads(req.data.decode())
        return {"data": [{"embedding": [1.0, 2.0, 3.0], "index": i}
                         for i in range(len(body["input"]))]}

    monkeypatch.setattr(embed_mod, "_http_post_json", fake_post)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda _p: "test-key")

    emb = embed_mod.HostedEmbedder(dict(embed_mod._DEFAULTS), use_breaker=False)
    emb._dim = 3
    out = emb._embed_batch(["a", "b"])
    assert len(out) == 2
    # _embed_batch L2-normalizes, so compare against the unit vector, not the raw payload.
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(out[0], expected / np.linalg.norm(expected))
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
