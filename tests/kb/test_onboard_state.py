"""Onboarding state is derived from disk: the one key, what this home can read from, the consent
marker, and which connected platforms have never had their collectors read.

The property that matters most here is that the `sources` phase has FOUR ways out plus a skip.
Its predecessor asked only "is X connected" and had no bypass, so a user who reads blogs or
watches research topics never reached consent or curation at all.
"""

import pytest

from pipeline.ingestion import x_graphql
from pipeline.ingestion.sources import substack as sub
from pipeline.kb import onboard_state


@pytest.fixture()
def _ready(monkeypatch, kb_home):
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "ok", "message": ""})
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    # Stubbed even when a test only cares about X: unstubbed, this probe scans the machine's
    # real OPYT profiles, so the suite's result would depend on whose laptop it runs on.
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: False)
    (kb_home / "onboard_consent_asked").touch()
    monkeypatch.setattr(onboard_state, "_curation_pending", lambda live: [])
    return kb_home


def _no_sessions(monkeypatch):
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: False)


def _store(monkeypatch, **flags):
    monkeypatch.setattr(onboard_state, "_store_sources",
                        lambda: {**onboard_state._NO_STORE_SOURCES, **flags})


def test_all_green_is_done(_ready):
    state = onboard_state.derive()
    assert state["phase"] == "done"
    assert state["sources"]["x"] is True and state["sources"]["ok"] is True


def test_missing_openrouter_is_keys(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "missing", "message": ""})
    assert onboard_state.derive()["phase"] == "keys"


def test_unfunded_openrouter_is_also_keys(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "unfunded", "message": ""})
    assert onboard_state.derive()["phase"] == "keys"


def test_nothing_connected_and_an_empty_store_is_sources(_ready, monkeypatch):
    _no_sessions(monkeypatch)
    assert onboard_state.derive()["phase"] == "sources"


# ── the four ways out of the sources phase, and the skip ────────────────────

@pytest.mark.parametrize("platform", ["x", "substack"])
def test_either_managed_session_exits_sources(_ready, monkeypatch, platform):
    _no_sessions(monkeypatch)
    _store(monkeypatch)
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: platform == "x")
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: platform == "substack")

    state = onboard_state.derive()

    assert state["phase"] != "sources"
    # Substack has a collector too, so it makes the curation phase mean something — the same
    # way X does, and unlike the two roots with no account to read.
    assert state["curation"]["applicable"] is True


@pytest.mark.parametrize("fact", ["oracles", "watchlist", "candidates"])
def test_one_store_row_exits_sources_with_no_session_at_all(_ready, monkeypatch, fact):
    """⚠️ THE WHOLE POINT. A user who named blogs (`oracles`), watched research topics
    (`watchlist`), or had a collector surface someone (`candidates`) has a real source, and no
    login can prove it. The predecessor phase had no bypass and stranded all three."""
    _no_sessions(monkeypatch)
    _store(monkeypatch, **{fact: True})
    assert onboard_state.derive()["phase"] != "sources"


def test_skip_exits_sources_and_is_never_re_asked(_ready, monkeypatch):
    _no_sessions(monkeypatch)
    _store(monkeypatch)
    assert onboard_state.derive()["phase"] == "sources"
    onboard_state.mark_sources_skipped()
    assert onboard_state.derive()["phase"] != "sources"


# ── the curation phase only gates when a collector can actually run ─────────

def test_no_curation_run_is_curation_when_a_collector_platform_is_connected(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state, "_curation_pending", lambda live: sorted(live))
    assert onboard_state.derive()["phase"] == "curation"


def test_curation_does_not_gate_done_for_a_user_with_no_collector(_ready, monkeypatch):
    """⚠️ The same trap as the old browser phase, one level down. `curation.pending` is drawn from
    the CONNECTED collector platforms, and every collector reads a platform session — so a user
    whose root is named blogs or research topics would otherwise sit at `curation` forever."""
    _no_sessions(monkeypatch)
    _store(monkeypatch, oracles=True)
    monkeypatch.setattr(onboard_state, "_curation_pending", lambda live: sorted(live))
    state = onboard_state.derive()
    assert state["curation"]["applicable"] is False
    assert state["phase"] == "done"


# ── consent ────────────────────────────────────────────────────────────────

def test_never_asked_is_consent_even_with_no_markers(_ready):
    (_ready / "onboard_consent_asked").unlink()
    assert onboard_state.derive()["phase"] == "consent"


def test_answered_no_to_both_is_not_re_asked(_ready):
    assert not (_ready / "bookmark_catchup_consent").exists()
    assert not (_ready / "oracle_refresh_consent").exists()
    assert onboard_state.derive()["phase"] == "done"


def test_consent_markers_honor_the_rails_env_override(_ready, monkeypatch, tmp_path):
    marker = tmp_path / "elsewhere" / "bm-consent"
    marker.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_BOOKMARK_CATCHUP_CONSENT", str(marker))
    assert onboard_state.derive()["consent"]["bookmark"] is False
    marker.touch()
    assert onboard_state.derive()["consent"]["bookmark"] is True


# ── fail-safe ──────────────────────────────────────────────────────────────

def test_derive_never_raises_when_everything_is_broken(monkeypatch, kb_home):
    def boom(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr(x_graphql, "has_managed_x_session", boom)
    monkeypatch.setattr(sub, "has_managed_substack_session", boom)
    monkeypatch.setattr(onboard_state, "_store_sources", boom)
    monkeypatch.setattr(onboard_state, "_curation_pending", boom)
    monkeypatch.setattr(onboard_state.readiness, "openrouter", boom)
    assert onboard_state.derive()["phase"] == "keys"


def test_a_broken_probe_reads_as_not_done_never_as_done(monkeypatch, kb_home):
    """Direction, not just absence of a crash: a failed probe must send the user to a phase that
    re-checks. Reading as DONE would skip a real step silently."""
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "ok", "message": ""})
    monkeypatch.setattr(onboard_state, "_store_sources",
                        lambda: (_ for _ in ()).throw(RuntimeError("store")))
    _no_sessions(monkeypatch)
    assert onboard_state.derive()["phase"] == "sources"
