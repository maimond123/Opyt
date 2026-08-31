"""
tests/kb/test_content_gate_concurrency.py — ARC-1: the gate grades a page's batches concurrently.

`content_gate` was measured at 63% (Substack) / 97% (blog) of ingest wall-clock while running its
LLM calls strictly one after another. Parallelizing them is safe ONLY because of one structural
property, which these tests pin:

    each batch votes on a DISJOINT set of unit indices, and a missing index defaults to KEEP

so the verdict cannot depend on the order batches finish in — concurrency changes only the clock.
The other half is fail-safe: per-batch isolation has to survive the move off the calling thread, or
one bad batch would take the page down instead of just its own units.
"""
import json
import re
import threading
import time

import pytest

from pipeline.kb import content_gate as cg

# This module drives the REAL content gate (against a faked llm_client), so it opts out of
# tests/kb/conftest.py's autouse keep-all stub.
pytestmark = pytest.mark.real_gate



class _Resp:
    def __init__(self, text):
        self.text = text


def _page(n_units: int, unit_chars: int = 4000) -> str:
    """A body big enough to force >1 batch (budget: 12000 chars / 40 units per LLM call)."""
    return "\n\n".join(f"Unit {i} " + ("x" * unit_chars) for i in range(n_units))


def _idxs_in(prompt: str) -> list[int]:
    """Recover the indices a batch was asked about — `_prompt` renders `[i] <unit>`."""
    return [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\] ", prompt, re.M)]


@pytest.fixture()
def gate_ready(monkeypatch):
    """Neutralize preflight so `classify_page` reaches the LLM loop with no key configured."""
    monkeypatch.setattr("pipeline.llm_client.preflight", lambda role: None)


@pytest.fixture()
def restore_gate():
    """The gate is a PROCESS-WIDE singleton; a test that halves it must not leak that."""
    before = cg._GATE_SEM._limit
    yield
    cg._GATE_SEM._limit = before


def _install(monkeypatch, fn):
    monkeypatch.setattr("pipeline.llm_client.call", fn)


def _keep(prompt: str) -> _Resp:
    return _Resp(json.dumps({str(i): "keep" for i in _idxs_in(prompt)}))


# ── the win: batches actually overlap ────────────────────────────────────────────

def test_batches_run_concurrently(gate_ready, monkeypatch):
    md = _page(9)
    assert len(cg._batch(cg._split_units(md))) > 1, "fixture must produce multiple batches"

    state = {"inflight": 0, "peak": 0}
    lock = threading.Lock()
    release = threading.Event()

    def _call(role, *, system, user, **k):
        with lock:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        release.wait(timeout=2.0)          # hold calls open so overlap is observable
        with lock:
            state["inflight"] -= 1
        return _keep(user)

    _install(monkeypatch, _call)
    t = threading.Timer(0.3, release.set)
    t.start()
    cg.classify_page(md)
    t.cancel()
    assert state["peak"] > 1, f"batches still serial (peak in-flight={state['peak']})"


# ── the safety argument: disjoint indices ⇒ order cannot matter ──────────────────

def test_verdict_is_independent_of_completion_order(gate_ready, monkeypatch):
    """Grade the same page twice with INVERTED per-batch delays. If batches truly own disjoint
    indices, both runs must keep exactly the same units."""
    md = _page(9)
    drop = {i for i in range(len(cg._split_units(md))) if i % 2 == 0}

    def make(slow_first: bool):
        def _call(role, *, system, user, **k):
            idxs = _idxs_in(user)
            is_first = bool(idxs) and idxs[0] == 0
            time.sleep(0.05 if is_first == slow_first else 0.0)
            return _Resp(json.dumps({str(i): ("drop" if i in drop else "keep") for i in idxs}))
        return _call

    _install(monkeypatch, make(slow_first=True))
    a = cg.classify_page(md)
    _install(monkeypatch, make(slow_first=False))
    b = cg.classify_page(md)

    assert a.keep == b.keep, "completion order changed the verdict — batches are NOT disjoint"
    assert a.kept_text == b.kept_text
    assert any(k is False for k in a.keep), "fixture must actually drop something to be meaningful"


# ── fail-safe survives the move off the calling thread ───────────────────────────

def test_one_failing_batch_keeps_its_units_and_the_page_lives(gate_ready, monkeypatch):
    md = _page(9)
    seen = {"n": 0}
    hit_lock = threading.Lock()

    def _call(role, *, system, user, **k):
        with hit_lock:
            seen["n"] += 1
            first = seen["n"] == 1
        if first:
            raise RuntimeError("boom")
        return _keep(user)

    _install(monkeypatch, _call)
    v = cg.classify_page(md)

    assert v.kept_text is not None, "one bad batch must never take the whole page down"
    assert v.degraded is True, "a failed batch must mark the page degraded"
    assert all(v.keep), "a failed batch's units default to KEEP (conservative)"


def test_n_calls_is_not_lost_under_concurrency(gate_ready, monkeypatch):
    """`n_calls` is a read-modify-write across threads; a lost increment silently under-reports
    spend. It must equal the number of batches when every call returns."""
    md = _page(9)
    n_batches = len(cg._batch(cg._split_units(md)))
    _install(monkeypatch, lambda role, *, system, user, **k: _keep(user))
    assert cg.classify_page(md).n_calls == n_batches


# ── AIMD acts on backpressure only ───────────────────────────────────────────────

def test_429_halves_the_gate_but_other_errors_do_not(gate_ready, monkeypatch, restore_gate):
    """Halving on a missing key or a 5xx would shrink the pool for a reason unrelated to rate."""
    md = _page(9)

    class _Err(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status = status

    def raiser(status):
        def _call(role, *, system, user, **k):
            raise _Err(status)
        return _call

    before = cg._GATE_SEM._limit
    _install(monkeypatch, raiser(500))
    cg.classify_page(md)
    assert cg._GATE_SEM._limit == before, "a 500 is not backpressure — the limit must hold"

    _install(monkeypatch, raiser(429))
    cg.classify_page(md)
    assert cg._GATE_SEM._limit < before, "a 429 must halve the gate"
