"""The X request meter survives the process — the one thing a detached rail child could not do.

Measured on a live store over two days: 94 of 148 logged failures were
a fresh rail child spending a REAL request to discover a bucket the previous child had already
emptied. `_RATE_STATE` was a process-local dict and every rail is a detached child, so the meter
existed for exactly as long as the process that filled it.

The bucket is server-side, so nothing here predicts anything — it remembers what x.com already
said, per home, because the session that spent the budget belongs to a home and not to a machine.
"""
from __future__ import annotations

import time

import pytest

from pipeline.platforms.x import rate


@pytest.fixture(autouse=True)
def _clear_process_state():
    rate._RATE_STATE.clear()
    yield
    rate._RATE_STATE.clear()


def test_a_fresh_process_inherits_the_spent_bucket():
    """THE point of the file. The dict is gone; the wall is not."""
    reset = time.time() + 600
    rate.record_rate_values("UserTweets", 0, reset)

    rate._RATE_STATE.clear()                       # a new rail child: same home, empty memory

    assert rate.rate_budget("UserTweets") == (0, pytest.approx(reset))
    with pytest.raises(rate.XRateLimited) as e:
        rate.refuse_if_spent("UserTweets")
    assert e.value.reset_at == pytest.approx(reset)


def test_a_rolled_over_window_is_forgotten_in_both_places():
    """A refusal past its reset instant would be a deadlock: only a request refills the meter,
    and the refusal is what prevents the request. Dropping it from the dict alone would leave the
    NEXT process reading a bucket that has been spent-forever since some hour it cannot see."""
    rate.record_rate_values("UserTweets", 0, time.time() - 1)

    rate.refuse_if_spent("UserTweets")            # allowed, and clears the stale reading

    assert rate.read("UserTweets") is None
    assert rate.rate_budget("UserTweets") is None


def test_the_stricter_reading_wins_within_one_window():
    """Two processes report against ONE server-side bucket, so the last writer is not the one
    that knows the most. Same window → the smaller `remaining` saw more of it spent."""
    reset = time.time() + 600
    rate.record("UserTweets", 3, reset)           # another child, further along
    rate._RATE_STATE["UserTweets"] = (40, reset)   # this one, from an earlier response

    assert rate.rate_budget("UserTweets") == (3, pytest.approx(reset))


def test_a_newer_window_supersedes_a_spent_one():
    """What dates a reading is the WINDOW it belongs to, never its clock age. A later `reset` is
    a bucket that has since refilled, so it replaces the old one outright rather than being
    merged with it — otherwise one spent window would suppress every window after it."""
    old_reset = time.time() + 100
    rate.record("UserTweets", 0, old_reset)
    rate.record("UserTweets", 50, old_reset + 900)

    assert rate.read("UserTweets") == (50, pytest.approx(old_reset + 900))


def test_each_operation_keeps_its_own_row():
    """Per-operation buckets (measured: UserTweets 50, UserByScreenName 150,
    TweetResultsByRestIds 500) — one meter row each, or a spent timeline walk would refuse a
    profile read that has 149 requests left."""
    reset = time.time() + 600
    rate.record_rate_values("UserTweets", 0, reset)
    rate.record_rate_values("TweetResultsByRestIds", 480, reset)

    rate._RATE_STATE.clear()

    assert rate.rate_budget("UserTweets") == (0, pytest.approx(reset))
    assert rate.rate_budget("TweetResultsByRestIds") == (480, pytest.approx(reset))


def test_one_homes_spending_never_refuses_anothers_requests(tmp_path, monkeypatch):
    """PER HOME, because the bucket belongs to that home's X SESSION. A hosted box serves many
    homes and many X accounts; a machine-wide meter would have one user's walk wall off
    another's."""
    other = tmp_path / "other-home"
    rate.record("UserTweets", 0, time.time() + 600, home=other)

    assert rate.read("UserTweets") is None                  # this home knows nothing of it
    assert rate.read("UserTweets", home=other) == (0, pytest.approx(time.time() + 600, abs=5))


def test_an_unreadable_meter_degrades_to_knowing_nothing(monkeypatch):
    """Fail-safe, per the invariant. A meter that cannot be opened leaves every caller exactly
    where it was before this file existed — blind for one request, which the next response fills
    in. It must never be the reason a pull does not happen."""
    monkeypatch.setattr(rate, "_connect", lambda p: (_ for _ in ()).throw(OSError("read-only")))

    assert rate.read("UserTweets") is None
    rate.record("UserTweets", 0, time.time() + 600)         # no raise
    rate.forget("UserTweets")                               # no raise
