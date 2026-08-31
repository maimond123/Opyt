"""
tests/ingestion/test_x_cookie_delegation.py

X is the highest-blast-radius consumer of the generalized reader (it's the working,
load-bearing bookmark path), so it gets its own regression tests: the picker data
layer (list_x_logged_in_profiles) must yield the {profile,label} shape onboarding keys
on, and read_x_cookies must behave identically for the single-candidate case while
delegating selection to the shared reader.

The picker no longer carries cookies or the twid-derived account_id. Detection is a
row-presence check that never decrypts, so producing either would mean launching a
browser per candidate — see browser_cookies.list_logged_in.
"""

from __future__ import annotations

import pytest

from pipeline.ingestion import browser_cookies as bc
from pipeline.ingestion import x_graphql as xg
from pipeline.ingestion.utils import SyncAuthError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPYT_BROWSER", raising=False)
    monkeypatch.delenv("X_CHROME_PROFILE", raising=False)


def _candidate(browser, profile, label, consent):
    """One list_logged_in candidate, in the post-transplant shape: the backend and the
    cookie file it was detected in, and no decrypted values."""
    return {"browser": browser, "profile": profile, "label": label, "consent": consent,
            "backend": bc.backend_for(browser), "cookie_file": None}


def test_list_profiles_maps_to_x_shape(monkeypatch):
    """A Chromium candidate keeps its profile dir as the selector."""
    monkeypatch.setattr(bc, "list_logged_in", lambda domains, ac, browsers=None: (
        [_candidate("chrome", "Profile 3", "Work — a@b.com", "none")], []))
    profs = xg.list_x_logged_in_profiles()
    assert profs == [{"profile": "Profile 3", "label": "Work — a@b.com"}]


def test_list_profiles_browser_level_uses_browser_key_as_selector(monkeypatch):
    """A browser-level candidate (Safari/Firefox, profile=None) surfaces its browser key
    as the `profile` selector, so the CLI/env still has a string to key on."""
    monkeypatch.setattr(bc, "list_logged_in", lambda domains, ac, browsers=None: (
        [_candidate("safari", None, "Safari", "fda")], []))
    profs = xg.list_x_logged_in_profiles()
    assert profs[0]["profile"] == "safari"


def test_list_profiles_carries_no_session_secrets(monkeypatch):
    """The picker feeds an MCP payload. It must not carry live cookies through it."""
    monkeypatch.setattr(bc, "list_logged_in", lambda domains, ac, browsers=None: (
        [_candidate("chrome", "Default", "Default", "none")], []))
    assert set(xg.list_x_logged_in_profiles()[0]) == {"profile", "label"}


def test_read_x_cookies_single_candidate(monkeypatch):
    """Single-candidate case is unchanged from the caller's side: the lone session is
    resolved and returned. What changed is WHEN it is decrypted — here, not in the scan."""
    monkeypatch.setattr(bc, "list_logged_in", lambda domains, ac, browsers=None: (
        [_candidate("chrome", "Default", "Default", "none")], []))
    monkeypatch.setattr(bc, "_read_one", lambda b, d, *, cookie_file=None: (
        {"auth_token": "t", "ct0": "csrf"}, None))
    got = xg.read_x_cookies()
    assert got == {"auth_token": "t", "ct0": "csrf"}


def test_read_x_cookies_blocked_gives_actionable_error(monkeypatch):
    """No login found but Safari was blocked → the actionable FDA message, not a bare
    'not logged in'."""
    monkeypatch.setattr(bc, "list_logged_in", lambda domains, ac, browsers=None: (
        [],
        [{"browser": "safari", "profile": None, "kind": "fda_needed", "detail": "denied"}],
    ))
    with pytest.raises(SyncAuthError) as ei:
        xg.read_x_cookies()
    assert "Full Disk Access" in str(ei.value)
