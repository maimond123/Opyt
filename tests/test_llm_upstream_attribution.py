"""Which UPSTREAM served each call gets RECORDED — the audit trail for `provider.sort`.

Why this needs pinning at all: `sort: "throughput"` re-ranks against OpenRouter's live
measurements, so the upstream it picks legitimately differs run to run (Groq on 2026-07-31,
SambaNova on 2026-08-01). That makes "routing quietly started choosing something slow, or
something that grades badly" a change with NO other symptom — the same shape as the Cloudflare
failure that billed 70 tokens per call and returned zero parseable verdicts.

`raw["provider"]` is on every OpenRouter response and `raw` is not persisted anywhere, so if the
call does not record it, the information is gone the moment the response is discarded.
"""
from __future__ import annotations

import json

import pytest

from pipeline import llm_client


class _PassthroughBreaker:
    """The real breaker persists state in opyt.db; tests must not touch it.

    Signature MIRRORS CircuitBreaker.call, `ignore` included: a double that drifts from the real
    interface stops being a stand-in and starts being a second, wrong implementation."""

    def call(self, fn, *, ignore: tuple = ()):
        try:
            return fn()
        except ignore:
            raise


@pytest.fixture(autouse=True)
def _isolated_stats(tmp_path, monkeypatch):
    """Stats + latency are process-global accumulators; give each test its own."""
    llm_client._override_stats_file_for_tests(tmp_path / "api_stats.json")
    llm_client.reset_latency()
    yield
    llm_client._override_stats_file_for_tests(None)
    llm_client.reset_latency()


def _fake_openrouter(monkeypatch, *, served_by: str | None, elapsed: float = 0.5):
    """Real `_call_openrouter_sync`, faked transport — so the response shape is the real one."""
    def fake_http_json(req, timeout=180.0):
        body = {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001}}
        if served_by is not None:
            body["provider"] = served_by        # OpenRouter names the serving upstream here
        return (body, elapsed)

    monkeypatch.setattr(llm_client, "_http_json", fake_http_json)
    monkeypatch.setattr(llm_client, "_openrouter_breaker", lambda: _PassthroughBreaker())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_serving_upstream_lands_in_stats_and_latency(monkeypatch):
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=0.8)
    llm_client.call("vision", system="", user="hi")

    assert llm_client._STATS["by_upstream"]["Groq"]["calls"] == 1
    assert llm_client._STATS["by_upstream"]["Groq"]["input_tokens"] == 10
    assert llm_client.upstream_distribution()["Groq"]["count"] == 1
    assert llm_client.upstream_distribution()["Groq"]["p50"] == 0.8


def test_a_routing_shift_is_visible_across_calls(monkeypatch):
    """The point of the whole thing: two upstreams serving the same role must be
    distinguishable. `latency_distribution` alone folds them into one role bucket and CANNOT
    show this — which is why a 10x spread read as 'per-call luck' for weeks."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=0.8)
    llm_client.call("vision", system="", user="hi")
    _fake_openrouter(monkeypatch, served_by="DeepInfra", elapsed=9.0)
    llm_client.call("vision", system="", user="hi")

    by_up = llm_client.upstream_distribution()
    assert set(by_up) == {"Groq", "DeepInfra"}
    assert by_up["Groq"]["p50"] == 0.8 and by_up["DeepInfra"]["p50"] == 9.0
    # The per-ROLE view sees one bucket with a big spread and cannot say why.
    assert llm_client.latency_distribution()["vision"]["count"] == 2


def test_unattributable_call_is_bucketed_not_dropped(monkeypatch):
    """A response with no `provider` field must show up as `unknown`, never vanish. A silently
    absent sample is indistinguishable from no call at all, and 'we cannot see it' is precisely
    the condition this bucket exists to surface."""
    _fake_openrouter(monkeypatch, served_by=None)
    llm_client.call("vision", system="", user="hi")

    assert llm_client._STATS["by_upstream"]["unknown"]["calls"] == 1
    assert "unknown" in llm_client.upstream_distribution()


def test_non_openrouter_provider_is_attributed_to_itself(monkeypatch):
    """Anthropic serves its own models, so the provider name IS the upstream — not `unknown`,
    which would make a healthy Anthropic run look unattributable."""
    monkeypatch.setattr(llm_client, "_load_role_config",
                        lambda role: {"provider": "anthropic", "model": "claude-x",
                                      "max_tokens": 16})
    llm_client._set_backend_for_tests(
        "anthropic", lambda model, system, user, max_tokens, **kw: ("ok", 5, 1, 0.3, {}))
    llm_client.call("summarize", system="", user="hi")

    assert llm_client._STATS["by_upstream"]["anthropic"]["calls"] == 1


def test_reset_latency_clears_upstreams_too(monkeypatch):
    """A 'fresh measurement window' that kept stale upstreams would attribute this run's numbers
    to a provider that never served it."""
    _fake_openrouter(monkeypatch, served_by="Groq")
    llm_client.call("vision", system="", user="hi")
    assert llm_client.upstream_distribution()

    llm_client.reset_latency()
    assert llm_client.upstream_distribution() == {}


def test_stats_shape_has_one_definition():
    """`_override_stats_file_for_tests` used to hand-copy the `_STATS` literal, so adding a
    bucket to production KeyError'd whichever tests reset it. Both must come from one factory."""
    llm_client._override_stats_file_for_tests(None)
    assert set(llm_client._STATS) == set(llm_client._fresh_stats())


def test_run_summary_helper_reports_both_views(monkeypatch):
    """`llm_run_stats` is what adapters spread into their summary; it must carry both keys, and
    must never raise — a diagnostic that breaks a run summary is worse than no diagnostic."""
    from pipeline.kb.ingest_common import llm_run_stats
    _fake_openrouter(monkeypatch, served_by="SambaNova")
    llm_client.call("vision", system="", user="hi")

    out = llm_run_stats()
    assert set(out) == {"llm_call_latency", "llm_upstreams"}
    assert out["llm_upstreams"]["SambaNova"]["count"] == 1
    assert out["llm_call_latency"]["vision"]["count"] == 1


# ── One RUN, not the process lifetime ────────────────────────────────────────
# `_LATENCY`/`_UPSTREAM_LATENCY` are append-only module globals and `reset_latency()` is called from
# tests only, so in the long-lived MCP server the second `add_oracle` of a session reported the
# FIRST one's calls folded into its own distribution. Percentiles cannot be subtracted, so the run
# boundary is a marker over the raw samples — the same snapshot/diff shape `ingest_x` already uses
# for `ocr_cascade.stats_snapshot()`.

def test_a_marker_excludes_the_previous_runs_samples(monkeypatch):
    """THE defect. Run 1's calls must not appear in run 2's numbers."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")          # run 1
    llm_client.call("vision", system="", user="hi")

    mark = llm_client.latency_marker()                       # ← run 2 starts here
    _fake_openrouter(monkeypatch, served_by="DeepInfra", elapsed=5.0)
    llm_client.call("vision", system="", user="hi")

    assert llm_client.latency_distribution(mark)["vision"]["count"] == 1
    assert llm_client.latency_distribution(mark)["vision"]["p50"] == 5.0
    # And the upstream view is bounded the same way — run 1's Groq is gone, not merely outnumbered.
    assert set(llm_client.upstream_distribution(mark)) == {"DeepInfra"}


def test_no_marker_still_reports_the_lifetime(monkeypatch):
    """Pins backwards compatibility: `since=None` must behave exactly as before, so nothing
    outside the wired call sites changes."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")
    llm_client.latency_marker()
    llm_client.call("vision", system="", user="hi")

    assert llm_client.latency_distribution()["vision"]["count"] == 2
    assert llm_client.upstream_distribution()["Groq"]["count"] == 2


def test_a_key_absent_at_mark_time_reads_from_zero(monkeypatch):
    """A role (or upstream) whose FIRST call happens after the mark has no entry in it. That must
    mean 'all of its samples are this run's', not a KeyError — the run that introduces a new role
    is exactly the run you want to see."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")

    mark = llm_client.latency_marker()                       # knows only `vision` / `Groq`
    _fake_openrouter(monkeypatch, served_by="SambaNova", elapsed=2.0)
    llm_client.call("content_quality", system="", user="hi")  # a role the marker never saw

    assert llm_client.latency_distribution(mark)["content_quality"]["count"] == 1
    assert llm_client.upstream_distribution(mark)["SambaNova"]["count"] == 1
    assert "vision" not in llm_client.latency_distribution(mark)   # fully consumed by run 1


def test_an_empty_marker_degrades_to_the_lifetime_view(monkeypatch):
    """`llm_run_marker()` returns `{}` when the import fails (fail-safe). That has to read as
    'no offsets', never as 'this run made no calls' — a diagnostic that silently zeroes is worse
    than one that over-reports."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")
    assert llm_client.latency_distribution({})["vision"]["count"] == 1


def test_run_stats_threads_the_marker_through(monkeypatch):
    """The adapters call `llm_run_stats(llm0)`; both views must honour the same marker."""
    from pipeline.kb.ingest_common import llm_run_marker, llm_run_stats
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")

    llm0 = llm_run_marker()
    _fake_openrouter(monkeypatch, served_by="SambaNova", elapsed=2.0)
    llm_client.call("vision", system="", user="hi")

    out = llm_run_stats(llm0)
    assert out["llm_call_latency"]["vision"]["count"] == 1
    assert set(out["llm_upstreams"]) == {"SambaNova"}
    assert llm_run_stats()["llm_call_latency"]["vision"]["count"] == 2   # lifetime, unchanged


def test_a_role_with_no_new_calls_drops_out_rather_than_reporting_zero(monkeypatch):
    """A role that made no calls this run must be ABSENT, not present with count 0 — `_summarize`
    indexes `xs[-1]` and an empty list would crash the summary it is supposed to describe."""
    _fake_openrouter(monkeypatch, served_by="Groq", elapsed=1.0)
    llm_client.call("vision", system="", user="hi")

    mark = llm_client.latency_marker()
    assert llm_client.latency_distribution(mark) == {}
    assert llm_client.upstream_distribution(mark) == {}
