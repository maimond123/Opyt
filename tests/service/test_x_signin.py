"""X sign-in on the service, for an install with no browser of its own (`service/x_signin.py`).

The desktop is the one stand-in (`FakeDesktop`): it starts nothing, and a test ends it the way the
real watcher does, through `on_done`. The slot, its deadlines and the routes are the real ones.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from service import x_signin, x_signin_desktop
from service.app import app
from service.x_signin_desktop import CONNECTED, EXPIRED, FAILED

SESSION = {"auth_token": "a" * 40, "ct0": "c" * 32, "twid": "u%3D42"}


class FakeDesktop:
    """Records what the slot asks of a desktop; `fail` makes `start` raise as a missing Xvfb would.
    It is `ready` (X's page drawn) as soon as it starts, unless a test says otherwise."""

    made: list["FakeDesktop"] = []
    fail = False

    def __init__(self, deadline, on_done):
        self.deadline, self.on_done = deadline, on_done
        self.started = self.restarted = self.closed = self.signed_in = self.ready = False
        self.vnc_port = None
        FakeDesktop.made.append(self)

    def start(self):
        if FakeDesktop.fail:
            raise x_signin_desktop.HostedBrowserError("Xvfb is not installed")
        self.started, self.vnc_port, self.ready = True, 5977, True

    def extend(self, deadline):
        self.deadline = deadline

    def restart_browser(self):
        self.restarted = True

    def end(self, outcome, session=None):
        """What the watcher does when X shows its home page, or the deadline does."""
        self.vnc_port = None
        self.on_done(outcome, session)


@pytest.fixture(autouse=True)
def slot(monkeypatch):
    monkeypatch.setattr(x_signin, "XSigninDesktop", FakeDesktop)
    monkeypatch.setattr(x_signin, "_signins", {})
    monkeypatch.setattr(x_signin, "_running", None)
    monkeypatch.setattr(x_signin, "_desktop", None)
    # A warm start runs on the calling thread here, so a test sees its outcome at once.
    monkeypatch.setattr(x_signin, "_spawn", lambda target, *args: target(*args))
    FakeDesktop.made, FakeDesktop.fail = [], False


def nonce_of(minted: dict) -> str:
    return minted["path"].rsplit("/", 1)[1]


def opened(now=0.0) -> tuple[dict, str]:
    """A minted sign-in whose page has claimed it: the minted reply and the tab's token."""
    minted = x_signin.mint(now)
    return minted, x_signin.claim(nonce_of(minted), None, now)["session"]


# ── decisions ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state, holds, busy, decision", [
    ("opened", True, False, "attach"),     # the desktop warmed for it at mint
    ("opened", False, False, "start"),
    ("opened", False, True, "busy"),
    ("running", True, False, "running"),
    ("connected", False, False, "connected"),
    ("failed", False, True, "failed"),
    ("expired", False, False, "expired"),
    ("minted", True, False, "expired"),    # a page cannot start before its link was opened
    (None, False, False, "expired"),
])
def test_what_a_page_start_gets(state, holds, busy, decision):
    assert x_signin.start_decision(state, holds, busy) == decision


@pytest.mark.parametrize("state, same_tab, decision", [
    ("minted", False, "claim"),
    ("opened", True, "claimed"),          # the same tab, reloaded
    ("running", True, "claimed"),
    ("running", False, "taken"),          # another tab, or a scanner that ran the script late
    ("connected", False, "connected"),
    ("failed", False, "failed"),
    ("expired", True, "expired"),
    (None, False, "expired"),
])
def test_what_a_page_claim_gets(state, same_tab, decision):
    assert x_signin.claim_decision(state, same_tab) == decision


@pytest.mark.parametrize("state, reply", [
    ("minted", "waiting"), ("opened", "waiting"), ("running", "waiting"),
    ("connected", "connected"), ("failed", "failed"), ("expired", "expired"), (None, "expired"),
])
def test_what_an_install_collect_gets(state, reply):
    assert x_signin.collect_reply(state) == reply


def test_x_home_page_is_recognised_by_its_window_title():
    tree = ('  0x400002 "Home / X": ("x.com__home" "Google-chrome")  1600x900+0+0\n')
    assert x_signin_desktop.home_shown(tree) and x_signin_desktop.x_page_shown(tree)
    login = tree.replace("Home / X", "Log in to X / X")
    assert x_signin_desktop.x_page_shown(login) and not x_signin_desktop.home_shown(login)
    assert not x_signin_desktop.x_page_shown(tree.replace("Home / X", "x.com/login"))
    assert x_signin_desktop.home_shown(tree.replace('"Home / X"', '"(3) Home / X"'))
    assert not x_signin_desktop.home_shown(tree.replace("Home / X", "Log in to X / X"))


def test_the_session_is_x_cookies_only_and_needs_auth_token_and_ct0():
    cookies = [{"domain": ".x.com", "name": k, "value": v} for k, v in SESSION.items()]
    cookies += [{"domain": ".x.com", "name": "guest_id", "value": "g"},
                {"domain": ".netflix.com", "name": "auth_token", "value": "not-x"}]
    assert x_signin_desktop.session_from(cookies) == SESSION
    without_ct0 = [c for c in cookies if c["name"] != "ct0"]
    assert x_signin_desktop.session_from(without_ct0) is None


# ── the flow ─────────────────────────────────────────────────────────────────────────────────

def test_a_sign_in_runs_from_link_to_one_collection():
    minted, session = opened()
    assert x_signin.collect(minted["signin"], 1.0) == {"status": "waiting"}
    started = x_signin.start(session, 1.0)
    # The page draws the whole laptop-sized screen and shows only Chrome's narrow window.
    assert started["status"] == "running" and (started["width"], started["height"]) == (1600, 900)
    assert started["window"] == {"width": 500, "height": 900}
    desktop = FakeDesktop.made[0]
    assert desktop.started and x_signin.vnc_port(session) == 5977

    desktop.end(CONNECTED, SESSION)
    assert x_signin.status(session, 2.0) == "connected"
    assert x_signin.collect(minted["signin"], 2.0) == {"status": "connected", "session": SESSION}
    # Given once: the session is gone from the service after its install takes it.
    assert x_signin.collect(minted["signin"], 2.0) == {"status": "expired"}


def test_the_collect_token_is_never_in_the_page_path():
    minted = x_signin.mint(0.0)
    assert minted["signin"] not in minted["path"]


def test_fetching_the_link_spends_nothing_and_the_first_tab_to_claim_holds_it():
    """A link preview fetched each link before the user's browser did and ran no script."""
    minted = x_signin.mint(0.0)
    assert x_signin.link_open(nonce_of(minted), 1.0)                # a scanner's fetch
    assert x_signin.link_open(nonce_of(minted), 2.0)                # and the person's
    first = x_signin.claim(nonce_of(minted), None, 3.0)
    assert first["status"] == "claimed"
    assert x_signin.claim(nonce_of(minted), first["session"], 4.0) == first   # a reload
    assert x_signin.claim(nonce_of(minted), None, 5.0) == {"status": "taken"}
    assert x_signin.claim(nonce_of(minted), "a-guess", 5.0) == {"status": "taken"}


def test_a_second_page_waits_while_the_desktop_is_busy():
    _first, one = opened()
    _second, two = opened()
    assert x_signin.start(one, 1.0)["status"] == "running"
    assert x_signin.start(two, 1.0) == {"status": "busy"}
    FakeDesktop.made[0].end(FAILED)
    assert x_signin.start(two, 2.0)["status"] == "running"


def test_a_reloaded_page_reattaches_to_its_running_desktop():
    _minted, session = opened()
    x_signin.start(session, 1.0)
    assert x_signin.start(session, 2.0)["status"] == "running"
    assert len(FakeDesktop.made) == 1


def test_a_desktop_that_cannot_start_fails_the_sign_in_and_frees_the_slot():
    FakeDesktop.fail = True
    minted, session = opened()
    assert x_signin.start(session, 1.0) == {"status": "failed"}
    assert x_signin.collect(minted["signin"], 1.0) == {"status": "failed"}
    FakeDesktop.fail = False
    _other, other = opened(2.0)
    assert x_signin.start(other, 2.0)["status"] == "running"


def test_an_unopened_link_expires_after_ten_minutes():
    minted = x_signin.mint(0.0)
    late = x_signin.LINK_TTL + 1
    assert not x_signin.link_open(nonce_of(minted), late)
    assert x_signin.claim(nonce_of(minted), None, late) == {"status": "expired"}
    assert x_signin.collect(minted["signin"], late) == {"status": "expired"}


def test_an_uncollected_session_is_dropped_after_ten_minutes(monkeypatch):
    minted, session = opened()
    x_signin.start(session, 1.0)
    monkeypatch.setattr(x_signin.time, "monotonic", lambda: 100.0)
    FakeDesktop.made[0].end(CONNECTED, SESSION)
    assert x_signin.collect(minted["signin"], 100.0 + x_signin.RESULT_TTL + 1) == {
        "status": "expired"}


def test_the_page_hears_finishing_while_the_session_is_written_down():
    minted, session = opened()
    x_signin.start(session, 1.0)
    FakeDesktop.made[0].signed_in = True
    assert x_signin.status(session, 2.0) == "finishing"
    assert x_signin.collect(minted["signin"], 2.0) == {"status": "waiting"}


def test_a_desktop_deadline_reads_as_expired():
    minted, session = opened()
    x_signin.start(session, 1.0)
    FakeDesktop.made[0].end(EXPIRED)
    assert x_signin.status(session, 2.0) == "expired"
    assert x_signin.collect(minted["signin"], 2.0) == {"status": "expired"}


def test_restart_reaches_only_a_desktop_this_page_holds():
    _minted, session = opened()
    x_signin.start(session, 1.0)
    assert x_signin.restart(session) is True and FakeDesktop.made[0].restarted
    FakeDesktop.made[0].end(FAILED)
    assert x_signin.restart(session) is False


# ── the warm start ───────────────────────────────────────────────────────────────────────────

def test_a_free_slot_warms_a_desktop_at_mint_and_the_page_attaches_to_it():
    minted = x_signin.mint(0.0)
    warm = FakeDesktop.made[0]
    assert warm.started and warm.deadline == x_signin.WARM_SECONDS
    session = x_signin.claim(nonce_of(minted), None, 30.0)["session"]
    assert x_signin.start(session, 30.0)["status"] == "running"
    assert len(FakeDesktop.made) == 1                       # attached, not a second desktop
    assert warm.deadline == 30.0 + x_signin.DESKTOP_TTL      # full time from the page's start


def test_a_warm_desktop_nobody_opens_frees_the_slot_and_the_link_still_works():
    minted = x_signin.mint(0.0)
    FakeDesktop.made[0].end(EXPIRED)                         # its warm deadline passed
    _other, other = opened(200.0)
    assert len(FakeDesktop.made) == 2                        # the slot was free again
    session = x_signin.claim(nonce_of(minted), None, 210.0)["session"]
    assert x_signin.start(session, 210.0) == {"status": "busy"}
    assert x_signin.collect(minted["signin"], 210.0) == {"status": "waiting"}


def test_the_page_hears_loading_until_x_has_drawn_its_page():
    _minted, session = opened()
    x_signin.start(session, 1.0)
    FakeDesktop.made[0].ready = False
    assert x_signin.status(session, 2.0) == "loading"
    FakeDesktop.made[0].ready = True
    assert x_signin.status(session, 2.0) == "running"


# ── the routes ───────────────────────────────────────────────────────────────────────────────

def test_the_routes_carry_a_sign_in_end_to_end():
    client = TestClient(app)
    minted = client.post("/v1/x-signin").json()
    page = client.get(minted["path"])
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
    assert client.get(minted["path"]).status_code == 200          # a fetch spends nothing
    claim = f"{minted['path']}/claim"
    session = client.post(claim, json={}).json()["session"]
    assert session not in page.text                               # the token comes from claiming
    assert client.post(claim, json={}).json() == {"status": "taken"}
    assert client.post(f"/x-signin/s/{session}/start").json()["status"] == "running"
    assert client.get(f"/x-signin/s/{session}/status").json() == {"status": "running"}
    FakeDesktop.made[0].end(CONNECTED, SESSION)
    collected = client.post("/v1/x-signin/collect", json={"signin": minted["signin"]}).json()
    assert collected == {"status": "connected", "session": SESSION}


def test_a_stream_with_no_desktop_is_closed():
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/x-signin/s/nobody/stream") as ws:
            ws.receive_bytes()


def test_the_page_script_and_novnc_are_served():
    client = TestClient(app)
    assert client.get("/static/desktop.js").status_code == 200
    assert client.get("/static/novnc/core/rfb.js").status_code == 200


def test_the_page_links_the_opyt_mark():
    """A chat app draws a link's card from these; without them Muse drew "AP". The mark itself
    is served from `website/`, and `tests/website/test_service_mark.py` checks that."""
    client = TestClient(app)
    page = client.get(client.post("/v1/x-signin").json()["path"]).text
    assert 'href="/favicon.ico"' in page and 'href="/icon.png"' in page


def test_the_cookie_store_shows_a_session_by_its_auth_token_row(tmp_path):
    """The slow signal, for an X whose home page is not titled in English."""
    import sqlite3

    store = tmp_path / "Default" / "Network" / "Cookies"
    store.parent.mkdir(parents=True)
    with sqlite3.connect(store) as conn:
        conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, encrypted_value BLOB)")
        conn.execute("INSERT INTO cookies VALUES ('.x.com', 'guest_id', x'00')")
    assert not x_signin_desktop._store_has_session(tmp_path)
    with sqlite3.connect(store) as conn:
        conn.execute("INSERT INTO cookies VALUES ('.x.com', 'auth_token', x'00')")
    assert x_signin_desktop._store_has_session(tmp_path)
