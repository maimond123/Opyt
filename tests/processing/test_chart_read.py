"""chart_read — one call per image: a chart's extraction, or NOT_CHART. Fully offline.

Proves the verdict (empty → failure, NOT_CHART → a cached no-text read, anything else → a chart),
that every image costs exactly one call on `CHART_MODEL`, the fail-safe contract (failure → None,
never a crash), and how cache entries from before charts-only are read back. Live model behavior is not
measured here.
"""
from __future__ import annotations

import pytest

from pipeline import chart_read


class _Resp:
    """A minimal LLMResponse stand-in — the reader uses only .text/.input_tokens/.output_tokens."""
    def __init__(self, text, i=100, o=40):
        self.text, self.input_tokens, self.output_tokens = text, i, o


# ── the verdict (pure) ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", ["NOT_CHART", " not_chart\n", "`NOT_CHART`", "NOT_CHART.",
                                   "**NOT_CHART**"])
def test_not_chart_is_a_verdict_with_no_text(reply):
    mr = chart_read.verdict(reply)
    assert mr.kind == "not_chart" and mr.text == "" and mr.substance is False


@pytest.mark.parametrize("reply", ["", "   \n", None])
def test_an_empty_reply_is_a_failure(reply):
    assert chart_read.verdict(reply) is None


def test_anything_else_is_the_chart():
    mr = chart_read.verdict("Data Chasm\nx: dataset size (GB)\n~1,000x gap")
    assert mr.kind == "chart" and mr.substance is True and mr.text.startswith("Data Chasm")


# ── read_image (mocked llm_client) ─────────────────────────────────────────────────

def test_one_call_on_the_chart_model_with_context(monkeypatch):
    calls = []

    def fake(role, **kw):
        calls.append(kw)
        return _Resp("Revenue by quarter\nQ1 $2M → Q4 $5M")

    monkeypatch.setattr(chart_read.llm_client, "call", fake)
    mr = chart_read.read_image("http://img", context="our revenue")
    assert [c["model"] for c in calls] == [chart_read.CHART_MODEL]
    assert "Context from the post:\nour revenue" in calls[0]["user"]
    assert "NOT_CHART" in calls[0]["user"] and "table of numbers" in calls[0]["user"]
    assert mr.kind == "chart" and "Q4 $5M" in mr.text


def test_not_chart_reply_comes_back_as_a_read(monkeypatch):
    monkeypatch.setattr(chart_read.llm_client, "call", lambda role, **kw: _Resp("NOT_CHART", o=5))
    mr = chart_read.read_image("http://img")
    assert mr is not None and mr.to_cache() == {"text": "", "kind": "not_chart"}


def test_failure_returns_none_not_crash(monkeypatch):    # poison-value rule: skip, don't cache
    def boom(role, **kw):
        raise RuntimeError("402 out of credits")

    monkeypatch.setattr(chart_read.llm_client, "call", boom)
    assert chart_read.read_image("http://img") is None


def test_empty_reply_is_counted_as_a_failure(monkeypatch):
    monkeypatch.setattr(chart_read.llm_client, "call", lambda role, **kw: _Resp("", o=0))
    before = chart_read.stats_snapshot()["failures"]
    assert chart_read.read_image("http://img") is None
    assert chart_read.stats_snapshot()["failures"] == before + 1


def test_warns_when_output_near_ceiling(monkeypatch):
    monkeypatch.setattr(chart_read.llm_client, "call",
                        lambda role, **kw: _Resp("Title: dense multi-series chart", o=1900))
    logged = []
    monkeypatch.setattr(chart_read, "log", logged.append)
    chart_read.read_image("http://img")
    assert any("possible truncation" in m for m in logged)


def test_no_warning_well_under_ceiling(monkeypatch):
    monkeypatch.setattr(chart_read.llm_client, "call", lambda role, **kw: _Resp("NOT_CHART", o=5))
    logged = []
    monkeypatch.setattr(chart_read, "log", logged.append)
    chart_read.read_image("http://img")
    assert not any("possible truncation" in m for m in logged)


# ── the cache, including entries written before charts-only ────────────────────────

def test_from_cache_round_trips_both_kinds():
    for mr in (chart_read.MediaRead("Title: x", "chart"), chart_read.MediaRead("", "not_chart")):
        assert chart_read.from_cache(mr.to_cache()) == mr


def test_a_legacy_transcription_is_never_replayed():
    """An OCR transcription (`kind: document`) reads as a miss, so the image is read again under
    the chart prompt and the entry overwritten — never rendered as text."""
    assert chart_read.from_cache({"text": "a table", "kind": "document", "substance": True}) is None


def test_a_legacy_photo_is_not_a_chart():
    mr = chart_read.from_cache({"text": "", "kind": "not_chart"})
    assert mr.kind == "not_chart" and mr.text == ""


def test_a_legacy_chart_read_is_kept():
    mr = chart_read.from_cache({"text": "Here's the chart: Title: Data Chasm", "kind": "chart",
                                "substance": True})
    assert mr.kind == "chart" and mr.text == "Title: Data Chasm"        # lead-in stripped too


def test_from_cache_refuses_to_guess_at_a_legacy_string():
    """`image_descriptions.json` once held `describe_image`'s bare prose gloss beside this reader's
    dicts, and inferring a verdict from "is the string non-empty" kept posts a real read would
    drop. None means "cache miss" to every caller, so the entry is re-read and overwritten."""
    assert chart_read.from_cache("an old describe_image gloss") is None
    assert chart_read.from_cache("") is None
    assert chart_read.from_cache(None) is None


# ── the model check (once per process) ──────────────────────────────────────────────

def test_an_unroutable_model_disables_before_any_call(monkeypatch):
    from pipeline import model_routing
    chart_read._reset_stage_for_tests()
    monkeypatch.setattr(model_routing, "chart_model_unroutable",
                        lambda **kw: "google/gemma-4-31b-it has no provider under deny=['x']")

    def bomb(role, **kw):
        raise AssertionError("paid a round-trip with no routable model")
    monkeypatch.setattr(chart_read.llm_client, "call", bomb)

    assert chart_read.read_image("http://img") is None
    assert "has no provider" in (chart_read.stage_status() or "")
    chart_read._reset_stage_for_tests()


def test_a_check_that_throws_proceeds_with_the_model(monkeypatch):
    from pipeline import model_routing
    chart_read._reset_stage_for_tests()

    def broken(**kw):
        raise RuntimeError("catalog down")
    monkeypatch.setattr(model_routing, "chart_model_unroutable", broken)
    monkeypatch.setattr(chart_read.llm_client, "call", lambda role, **kw: _Resp("NOT_CHART"))

    assert chart_read.read_image("http://img") is not None
    chart_read._reset_stage_for_tests()


def test_this_module_holds_no_price_table():
    """Model selection belongs to routing, not a local table in this module."""
    assert not hasattr(chart_read, "cost") and not hasattr(chart_read, "_PRICE")
