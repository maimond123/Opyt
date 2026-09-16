"""`_fetch_all_posts` — the archive listing walk, at its network boundary.

Two contracts live here and neither was enforced against what the endpoint actually returns:

  • `since` means "on/after this date" (the CLI says so). The walk used the boundary page only to
    decide when to STOP, and appended that whole page first — so every older post sharing the
    boundary page was handed on for ingestion.
  • A listing that cannot be completed raises `SubstackListingError`, which the caller treats as
    "write nothing, mark nothing". A 200 response carrying valid JSON of the WRONG SHAPE bypassed
    that: `batch[-1]` raised `KeyError(-1)` straight out through the caller's fail-safe.

Offline: `requests.Session` and `time.sleep` are both replaced, so these make no network call and
take no wall-clock delay.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
import requests

from pipeline.ingestion.sources import substack as sub

SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    """Serves `pages` in order, one per GET, and records every call it received."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.gets: list[dict] = []

    def get(self, url, **kw):
        self.gets.append({"url": url, **kw})
        return _Resp(self._pages.pop(0) if self._pages else [])

    def close(self):
        return None


@pytest.fixture()
def archive(monkeypatch):
    """Install a fake session serving `pages`; returns it so a test can count the GETs."""
    def _install(pages):
        session = _Session(pages)
        monkeypatch.setattr(requests, "Session", lambda: session)
        return session
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return _install


def _post(pid, date):
    return {"id": pid, "post_date": date, "title": f"post {pid}"}


def test_a_page_wholly_older_than_since_yields_no_posts(archive):
    session = archive([[_post(1, "2025-01-01T00:00:00Z")]])
    assert sub._fetch_all_posts("https://carol.substack.com", since=SINCE) == []
    assert len(session.gets) == 1          # the boundary page still STOPS the walk


def test_the_boundary_page_keeps_only_the_posts_at_or_after_since(archive):
    archive([[_post(3, "2026-03-01T00:00:00Z"),
              _post(2, "2026-02-01T00:00:00Z"),
              _post(1, "2025-12-31T00:00:00Z")]])
    got = sub._fetch_all_posts("https://carol.substack.com", since=SINCE)
    assert [p["id"] for p in got] == [3, 2]


def test_a_record_with_no_usable_date_survives_the_filter(archive):
    """Fail-safe, matching the existing stop rule: an unparseable date is not evidence the post is
    old, so dropping it would silently lose real posts."""
    archive([[_post(2, ""), _post(1, "not-a-date"), _post(0, "2025-01-01T00:00:00Z")]])
    got = sub._fetch_all_posts("https://carol.substack.com", since=SINCE)
    assert [p["id"] for p in got] == [2, 1]


def test_without_since_every_page_is_returned_verbatim(archive):
    archive([[_post(2, "2026-02-01T00:00:00Z")], [_post(1, "2020-01-01T00:00:00Z")], []])
    got = sub._fetch_all_posts("https://carol.substack.com")
    assert [p["id"] for p in got] == [2, 1]


def test_a_json_object_instead_of_a_list_becomes_a_listing_error(archive):
    session = archive([{"error": "rate limited"}] * sub._ARCHIVE_RETRIES)
    with pytest.raises(sub.SubstackListingError):
        sub._fetch_all_posts("https://carol.substack.com", since=SINCE)
    assert len(session.gets) == sub._ARCHIVE_RETRIES   # retried as a transient listing failure


def test_a_list_of_non_posts_becomes_a_listing_error(archive):
    archive([["not-a-post"]] * sub._ARCHIVE_RETRIES)
    with pytest.raises(sub.SubstackListingError):
        sub._fetch_all_posts("https://carol.substack.com")


def test_a_shape_error_that_clears_on_retry_is_not_fatal(archive):
    """The shape check joins the EXISTING retry loop rather than short-circuiting it."""
    archive([{"error": "rate limited"}, [_post(1, "2026-02-01T00:00:00Z")], []])
    got = sub._fetch_all_posts("https://carol.substack.com")
    assert [p["id"] for p in got] == [1]


# ── managed_substack_session_ready — validity, not presence ─────────────────────────
#
# The onboarding "done" gate turns on this. Its whole reason for existing over
# `has_managed_substack_session` (presence) is that a session can be PRESENT — an anonymous
# `substack.sid`, or a login whose cookies Chrome has not flushed yet — without answering an
# authenticated request. So these assert what each real outcome of `own_user_id` means for the
# gate, and that the retry rides out the flush rather than blaming the user's sign-in for it.

def _ready(monkeypatch, *, cookies=None, uid_side_effects):
    """Drive `managed_substack_session_ready` with a scripted `own_user_id` and no real sleep."""
    calls = {"n": 0}

    def _uid(_cookies):
        i = calls["n"]
        calls["n"] += 1
        outcome = uid_side_effects[min(i, len(uid_side_effects) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(sub, "read_substack_cookies", lambda: cookies or {"substack.sid": "x"})
    monkeypatch.setattr(sub, "own_user_id", _uid)
    monkeypatch.setattr(sub.time, "sleep", lambda *_: None)
    return calls


def test_a_live_session_is_ready_on_the_first_try(monkeypatch):
    calls = _ready(monkeypatch, uid_side_effects=[12345])
    assert sub.managed_substack_session_ready() is True
    assert calls["n"] == 1                                  # no wasted retry once it answers


def test_a_cleanly_signed_out_session_is_not_ready(monkeypatch):
    calls = _ready(monkeypatch, uid_side_effects=[None])
    assert sub.managed_substack_session_ready(attempts=3) is False
    assert calls["n"] == 3                                  # spent every attempt before concluding


def test_a_presence_race_that_clears_on_retry_is_ready(monkeypatch):
    """The flush / write-lock race: the managed profile is momentarily unreadable, then settles.
    The retry is the whole point — one `SyncAuthError` must not become "you are not signed in"."""
    calls = _ready(monkeypatch, uid_side_effects=[sub.SyncAuthError("no session yet"), 999])
    assert sub.managed_substack_session_ready() is True
    assert calls["n"] == 2


def test_a_transient_refusal_is_not_blamed_on_the_user(monkeypatch):
    """A Cloudflare/transport refusal is not a signed-out session. Reporting it as "not ready"
    would tell a signed-in user to sign in again — the 2026-09-08 `own_user_id` trap."""
    _ready(monkeypatch, uid_side_effects=[sub.SubstackListingError("HTTP 403")])
    assert sub.managed_substack_session_ready() is True
