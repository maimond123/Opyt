"""Money-safe failure skip for the paid vision path.

The one test here asserts on `chart_read.read_image`, which the atom rail uses
(`pipeline/kb/ingest_x.py` and `pipeline/kb/vision.py` import it directly). It pins an
invariant — a failed external call must SKIP, never write a partial or poison value.
"""

from __future__ import annotations

import pytest

from pipeline import llm_client


def _fake_raise(*, role, system, user, **kw):
    raise RuntimeError("HTTP 402: no credit")       # simulates a paid-call failure


# ── money-safe failure skip ──────────────────────────────────────────────────
def test_failure_returns_none_not_empty(monkeypatch):
    """None is the skip sentinel: never persist a poison value on a failed PAID call.

    The distinction that matters is None vs "" — an empty string is a value, it caches, and it
    then reads as "this image has no text in it" forever. None is not cached, so the next run
    retries. `$0-credit once dumped 2097 garbage rows` is the incident this pins against.
    """
    from pipeline import model_routing, chart_read
    chart_read._reset_stage_for_tests()     # clears the conftest pin → pin resolution explicitly
    monkeypatch.setattr(model_routing, "chart_model_unroutable", lambda **kw: None)
    monkeypatch.setattr(llm_client, "call", _fake_raise)
    assert chart_read.read_image("https://x/y.jpg") is None
