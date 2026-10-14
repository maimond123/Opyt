"""Onboarding state is derived from disk: what this home can read from, and which connected
platforms have never had their collectors read. There is no consent phase.

The property that matters most here is that the `sources` phase has four ways out plus a skip.
Asking only "is X connected" with no bypass would leave a user who reads blogs or watches
research topics unable to reach curation at all.
"""

import pytest

from pipeline.curation import collectors
from pipeline.onboarding import phase
from pipeline.platforms import connections


@pytest.fixture()
def _ready(monkeypatch, kb_home):
    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    # Stubbed even when a test only cares about X: unstubbed, this probe scans the machine's
    # real OPYT profiles, so the suite's result would depend on whose laptop it runs on.
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)
    monkeypatch.setattr(collectors, "curation_pending", lambda live: [])
    return kb_home


def _no_sessions(monkeypatch):
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)


def _store(monkeypatch, **flags):
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: {**phase._NO_STORE_SOURCES, **flags})


def test_all_green_is_done(_ready):
    state = phase.derive()
    assert state["phase"] == "done"
    assert state["sources"]["x"] is True and state["sources"]["ok"] is True


def test_nothing_connected_and_an_empty_store_is_sources(_ready, monkeypatch):
    _no_sessions(monkeypatch)
    assert phase.derive()["phase"] == "sources"


# ── the four ways out of the sources phase, and the skip ────────────────────

@pytest.mark.parametrize("platform", ["x", "substack"])
def test_either_managed_session_exits_sources(_ready, monkeypatch, platform):
    _no_sessions(monkeypatch)
    _store(monkeypatch)
    monkeypatch.setattr(connections, "_connected_x", lambda: platform == "x")
    monkeypatch.setattr(connections, "_connected_substack", lambda: platform == "substack")

    state = phase.derive()

    assert state["phase"] != "sources"
    # Substack has a collector too, so it makes the curation phase mean something — the same
    # way X does, and unlike the two roots with no account to read.
    assert state["curation"]["applicable"] is True


@pytest.mark.parametrize("fact", ["oracles", "topics", "candidates"])
def test_one_store_row_exits_sources_with_no_session_at_all(_ready, monkeypatch, fact):
    """A user who named blogs (`oracles`), watched research topics (`topics`), or had a
    collector surface someone (`candidates`) has a real source, and no login can prove it."""
    _no_sessions(monkeypatch)
    _store(monkeypatch, **{fact: True})
    assert phase.derive()["phase"] != "sources"


def test_skip_exits_sources_and_is_never_re_asked(_ready, monkeypatch):
    _no_sessions(monkeypatch)
    _store(monkeypatch)
    assert phase.derive()["phase"] == "sources"
    phase.skip_sources()
    assert phase.derive()["phase"] != "sources"


# ── the curation phase only gates when a collector can actually run ─────────

def test_no_curation_run_is_curation_when_a_collector_platform_is_connected(_ready, monkeypatch):
    monkeypatch.setattr(collectors, "curation_pending", lambda live: sorted(live))
    assert phase.derive()["phase"] == "curation"


def test_curation_does_not_gate_done_for_a_user_with_no_collector(_ready, monkeypatch):
    """`curation.pending` is drawn from the connected collector platforms, and every collector reads a platform session — so a user
    whose root is named blogs or research topics would otherwise sit at `curation` forever."""
    _no_sessions(monkeypatch)
    _store(monkeypatch, oracles=True)
    monkeypatch.setattr(collectors, "curation_pending", lambda live: sorted(live))
    state = phase.derive()
    assert state["curation"]["applicable"] is False
    assert state["phase"] == "done"


# ── no consent phase ─────────────────────────────────────────────

def test_a_connected_home_goes_straight_from_sources_to_curation(_ready, monkeypatch):
    """Nothing is asked between connecting and the walk, so there is no phase for it."""
    monkeypatch.setattr(collectors, "curation_pending", lambda live: sorted(live))
    state = phase.derive()
    assert state["phase"] == "curation"
    assert "consent" not in state


# ── fail-safe ──────────────────────────────────────────────────────────────

def test_derive_never_raises_when_everything_is_broken(monkeypatch, kb_home):
    def boom(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr(connections, "_connected_x", boom)
    monkeypatch.setattr(connections, "_connected_substack", boom)
    monkeypatch.setattr(phase, "_store_sources", boom)
    monkeypatch.setattr(collectors, "curation_pending", boom)
    assert phase.derive()["phase"] == "sources"


def test_a_broken_probe_reads_as_not_done_never_as_done(monkeypatch, kb_home):
    """Direction, not just absence of a crash: a failed probe must send the user to a phase that
    re-checks. Reading as done would skip a real step silently."""
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: (_ for _ in ()).throw(RuntimeError("store")))
    _no_sessions(monkeypatch)
    assert phase.derive()["phase"] == "sources"

