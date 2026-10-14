"""The call after a local sign-in waits for the login to reach the disk.

Chrome writes new cookies to disk in batches, up to 30 s after they are set. A user
once said "done" 27 s after the X window opened; two `onboard` calls in the next second read the disk,
found no session and answered with the whole sources menu. The session was on disk 25 s later.
"""
from __future__ import annotations

import pytest

from pipeline.onboarding import connect, flow, phase, worker_check
from pipeline.platforms import connections
from pipeline.platforms.browser import cookies as bc, login


class Clock:
    """A clock that only `sleep` moves, so a test counts the waits."""

    def __init__(self):
        self.now, self.sleeps = 0.0, 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(login, "start", lambda url: bc.backend_for("chrome"))
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)
    monkeypatch.setattr(connections, "managed_substack_session_ready", lambda **kw: False)
    return tmp_path


def _lands_after(monkeypatch, probe: str, checks: int) -> None:
    """`probe` reads False for `checks` reads, then True: the flush landing."""
    seen = []
    monkeypatch.setattr(connections, probe,
                        lambda **kw: seen.append(1) is None and len(seen) > checks)


def test_no_sign_in_opened_waits_for_nothing(home):
    clock = Clock()
    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None
    assert clock.sleeps == 0


def test_the_call_after_a_sign_in_waits_until_the_session_is_on_disk(home, monkeypatch):
    connect.start("x")
    _lands_after(monkeypatch, "_connected_x", 3)
    clock = Clock()

    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None

    assert clock.sleeps == 3
    # Seen once: the next call has nothing to wait for.
    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None and clock.sleeps == 3


def test_a_sign_in_that_never_lands_is_held_not_sent_back_to_the_menu(home):
    connect.start("x")
    clock = Clock()

    held = connect.hold_for_signin(clock=clock, sleep=clock.sleep)

    assert (held["status"], held["source"]) == ("awaiting_login", "x")
    assert clock.now >= 30.0, "Chrome may hold a cookie for 30 s before writing it"
    assert "onboard(source='x')" in held["message"]
    # Still expected: the user finishes and says done again, and that call waits too.
    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is not None


def test_a_substack_cookie_alone_is_not_the_login(home, monkeypatch):
    """Substack hands a signed-out visitor a cookie, so the wait is for a session that answers."""
    connect.start("substack")
    monkeypatch.setattr(connections, "_connected_substack", lambda: True)
    _lands_after(monkeypatch, "managed_substack_session_ready", 2)
    clock = Clock()

    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None
    assert clock.sleeps == 2


def test_a_check_that_cannot_run_does_not_strand_the_user(home, monkeypatch):
    connect.start("x")
    monkeypatch.setattr(connections, "_connected_x", lambda: 1 / 0)
    clock = Clock()

    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None
    assert clock.sleeps == 0


@pytest.mark.parametrize("word", ["blog", "skip"])
def test_answering_another_way_stops_expecting_the_login(home, monkeypatch, word):
    monkeypatch.setattr(worker_check, "refuse_off_macos", lambda: None)
    monkeypatch.setattr(worker_check, "ensure", lambda: None)
    monkeypatch.setattr(phase, "_store_sources", lambda: dict(phase._NO_STORE_SOURCES))
    monkeypatch.setattr(flow.handoff, "answer", lambda state, work: {"status": "ok"})
    monkeypatch.setattr(flow.kickoff, "start_work", lambda state: {})
    connect.start("x")

    flow._answer(word, None)

    clock = Clock()
    assert connect.hold_for_signin(clock=clock, sleep=clock.sleep) is None
    assert clock.sleeps == 0


def test_onboard_holds_before_it_reads_where_setup_stands(home, monkeypatch):
    """The flow asks the hold first: deriving the phase from a disk that is 30 s behind is what
    produced the menu."""
    monkeypatch.setattr(worker_check, "refuse_off_macos", lambda: None)
    monkeypatch.setattr(connect, "hold_for_signin", lambda: {"status": "awaiting_login"})
    monkeypatch.setattr(phase, "derive", lambda: pytest.fail("derived before the hold"))

    assert flow._answer(None, None) == {"status": "awaiting_login"}
