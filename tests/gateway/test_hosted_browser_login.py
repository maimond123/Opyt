"""The site-agnostic hosted browser boundary: profile, desktop, relay, and shared Chrome.

Split from `test_hosted_x_login.py` on 2026-09-07 with the module it covers. Everything here
is true of any site that gets an interactive sign-in; what is true only of X stayed behind.
"""
from __future__ import annotations

import ast
import contextlib
import json
import inspect
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from gateway import app as gateway_app
from gateway import children
from gateway.children import Child, ChildPool
from pipeline.ingestion import cdp, hosted_browser


class _Process:
    """The pool's view of a child: alive until something sets `returncode`.

    `terminate` and `wait` are here for `ChildPool.shutdown`, which any test that runs the app's
    lifespan reaches on the way out.
    """

    pid = 1
    returncode = None

    def terminate(self) -> None:
        self.returncode = 0

    async def wait(self) -> int:
        return self.returncode


def _live_child(pool: ChildPool, subject: str = "42", key: str = "child-key") -> Child:
    child = Child(subject=subject, home=pool.homes_root / subject, port=5432,
                  proc=_Process(), last_seen=0, interaction_key=key)  # type: ignore[arg-type]
    pool._children[child.subject] = child
    return child


def _open_desktop(pool: ChildPool, *, key: str = "child-key", nonce: str = "n",
                  kind: str = "x") -> tuple[str, str]:
    """Take one nonce all the way to an open desktop, and return its two capabilities."""
    assert pool.register_interaction(key, nonce, kind)
    entry = pool.consume_interaction(nonce, kind)
    assert entry is not None
    return pool.create_login_session(entry)


def test_the_completion_is_single_use_but_the_stream_can_be_re_attached(monkeypatch, tmp_path):
    """The action is one-shot; watching the desktop is not, because phones drop sockets.

    A user leaving the browser for a 2FA code loses the WebSocket while the desktop keeps
    running, so a stream capability spent on first use stranded them. Completing is the action
    the login URL's one-shot boundary was about, and it stays one-shot.
    """
    now = [100.0]
    monkeypatch.setattr(children.time, "monotonic", lambda: now[0])
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example", login_ttl_seconds=10)
    child = _live_child(pool)

    assert pool.register_interaction("child-key", "nonce-a", "x") is True
    entry = pool.consume_interaction("nonce-a", "x")
    assert entry is not None and entry.child is child
    stream, complete = pool.create_login_session(entry)

    assert pool.read_login_session(stream) is not None
    assert pool.read_login_session(stream) is not None
    assert pool.consume_login_completion(complete) is not None
    assert pool.consume_login_completion(complete) is None

    now[0] += 11
    assert pool.read_login_session(stream) is None

    assert pool.register_interaction("child-key", "nonce-b", "x") is True
    now[0] += 11
    assert pool.consume_interaction("nonce-b", "x") is None


def test_a_desktop_capability_is_stamped_when_the_page_opens_not_when_the_link_was_minted(
        monkeypatch, tmp_path):
    """A slow user must not get a shorter desktop than a fast one.

    The child starts the desktop's ten minutes when the page asks for it. Inheriting the LINK's
    deadline here made both capabilities die early by however long the user took to open the
    link, leaving a desktop running that nobody could complete or re-attach to.
    """
    now = [100.0]
    monkeypatch.setattr(children.time, "monotonic", lambda: now[0])
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example", login_ttl_seconds=600)
    _live_child(pool)

    assert pool.register_interaction("child-key", "nonce-a", "x") is True
    entry = pool.consume_interaction("nonce-a", "x")
    now[0] += 300
    stream, complete = pool.create_login_session(entry)

    now[0] += 400
    assert pool.read_login_session(stream) is not None
    assert pool.consume_login_completion(complete) is not None


def test_openrouter_and_a_sign_in_keep_separate_one_time_routes(tmp_path):
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    _live_child(pool)

    assert pool.register_interaction("child-key", "x-nonce", "x")
    assert pool.register_interaction("child-key", "or-nonce", "openrouter")
    assert pool.consume_interaction("x-nonce", "openrouter") is None
    assert pool.consume_interaction("or-nonce", "openrouter") is not None


def test_a_capability_minted_for_one_site_cannot_be_redeemed_at_another(tmp_path):
    """The site is half the capability. Without this check an X nonce opens `/login/substack/…`
    and the page then reports on a profile the user never asked to connect."""
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    _live_child(pool)

    assert pool.register_interaction("child-key", "x-nonce", "x")
    assert pool.consume_interaction("x-nonce", "substack") is None
    assert pool.consume_interaction("x-nonce", "x") is not None


def test_a_second_sign_in_supersedes_the_first_even_for_another_site(tmp_path):
    """One profile per child means one desktop per child, whatever site asked for it.

    `HostedLoginManager.create` already clears every pending nonce, so a gateway that kept the
    older one would route a link its child has forgotten."""
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    _live_child(pool)

    assert pool.register_interaction("child-key", "x-nonce", "x")
    assert pool.register_interaction("child-key", "substack-nonce", "substack")

    assert pool.consume_interaction("x-nonce", "x") is None
    assert pool.consume_interaction("substack-nonce", "substack") is not None


def test_an_unknown_interaction_kind_is_refused(tmp_path):
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    _live_child(pool)

    assert pool.register_interaction("child-key", "n", "reddit") is False


# ── admission control on the sign-in path ────────────────────────────────────────────────────

def test_a_desktop_stops_counting_when_it_completes_or_expires(monkeypatch, tmp_path):
    """The count is only as good as its decrements, and an over-count refuses real users."""
    now = [100.0]
    monkeypatch.setattr(children.time, "monotonic", lambda: now[0])
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example", login_ttl_seconds=10)
    _live_child(pool)

    _, completion = _open_desktop(pool, nonce="finished")
    assert pool.live_signins() == 1
    assert pool.consume_login_completion(completion) is not None
    assert pool.live_signins() == 0

    # Abandoned: the visitor closes the tab and never completes. The child's own timer closes
    # that desktop at the login TTL, and the count has to follow it down without being told.
    _open_desktop(pool, nonce="abandoned")
    assert pool.live_signins() == 1
    now[0] += 11
    assert pool.live_signins() == 0


def test_a_reaped_child_stops_counting_its_desktop(tmp_path):
    """A child takes its desktop with it — the whole process tree dies together."""
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    child = _live_child(pool)
    _open_desktop(pool)

    assert pool.live_signins() == 1
    child.proc.returncode = 0
    assert pool.live_signins() == 0


def test_a_superseded_sign_in_stops_counting_the_desktop_it_closed(tmp_path):
    """`HostedLoginManager.create` closes the child's open desktop before registering the new
    nonce, so counting the old one would hold a slot against a desktop nobody has."""
    pool = ChildPool(tmp_path, interaction_registration_url="http://gateway/register",
                     interaction_url="https://gateway.example")
    _live_child(pool)
    _open_desktop(pool, nonce="first")

    assert pool.live_signins() == 1
    assert pool.register_interaction("child-key", "second", "substack")
    assert pool.live_signins() == 0


class _StartedChild:
    """The child's `/start`, answered the way a live one answers, so success is a real 200."""

    async def post(self, url, **kwargs):
        import httpx

        return httpx.Response(200, json={"status": "started"})

    async def aclose(self) -> None:
        pass


def test_a_refused_sign_in_leaves_the_link_usable(monkeypatch, tmp_path):
    """The ruling this branch exists for: the capacity check runs BEFORE `consume_interaction`.

    Move it below the pop and this test fails on its last line — the refusal would have spent
    the visitor's nonce, and the only way back would be a new `onboard` call. That is the
    accident a later reader is most likely to have, which is why the assertion is the same
    nonce redeemed twice rather than anything about the counter.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("OPYT_GATEWAY_MAX_SIGNINS", "1")
    app = gateway_app.build_app(base_url="https://gw.example.com", client_id="cid",
                                client_secret="secret", homes_root=tmp_path)

    with TestClient(app) as client:
        app.state.http = _StartedChild()
        pool = app.state.pool
        _live_child(pool, subject="holder", key="holder-key")
        _, held = _open_desktop(pool, key="holder-key", nonce="held")

        _live_child(pool, subject="visitor", key="visitor-key")
        assert pool.register_interaction("visitor-key", "mine", "x")

        refused = client.get("/login/x/mine")
        assert refused.status_code == 503
        assert "still works" in refused.text

        # The holder finishes and its desktop closes.
        assert pool.consume_login_completion(held) is not None

        opened = client.get("/login/x/mine")

    assert opened.status_code == 200, "the refusal consumed the nonce it refused"
    assert "/static/desktop.js" in opened.text


def test_a_substack_link_opens_the_same_one_shot_desktop_as_x(tmp_path):
    """The 2026-09-15 flow adoption, pinned at the route: no paste page, no second flow.

    Substack's guided paste existed for one measured reason — the box's browser could not get
    Substack to send the sign-in email. Re-measured portless the same day it delivered 2-for-2
    and the code signed in (docs/plans/2026-09-15-substack-adopts-the-x-desktop-flow.md), so
    Substack takes X's desktop path whole: same page, same capacity ordering, and a link that
    is one-shot again — nothing about the desktop flow requires reopening.
    """
    from starlette.testclient import TestClient

    app = gateway_app.build_app(base_url="https://gw.example.com", client_id="cid",
                                client_secret="secret", homes_root=tmp_path)
    with TestClient(app) as client:
        pool = app.state.pool
        _live_child(pool)
        assert pool.register_interaction("child-key", "n", "substack")

        opened = client.get("/login/substack/n")
        again = client.get("/login/substack/n")

    assert opened.status_code == 200
    assert "/static/desktop.js" in opened.text
    assert "Connect your Substack account" in opened.text
    assert "/login/redeem/" not in opened.text
    assert again.status_code == 404, "a desktop link must not serve twice"


def test_hosted_profiles_are_scoped_to_the_selected_home(monkeypatch, tmp_path):
    first_home, second_home = tmp_path / "homes" / "first", tmp_path / "homes" / "second"
    monkeypatch.setenv("OPYT_HOME", str(first_home))
    first = hosted_browser.profile_dir()
    first.mkdir(parents=True)
    (first / "profile-marker").write_text("first")

    monkeypatch.setenv("OPYT_HOME", str(second_home))
    second = hosted_browser.profile_dir()

    assert first == first_home / "chrome-profile"
    assert second == second_home / "chrome-profile"
    assert second != first
    assert not second.exists(), "one hosted home acquired another home's browser profile"


def _code_without_prose(module) -> str:
    """A module's executable source, with every comment and docstring removed.

    The scan below must see CODE. These modules name the local cookie readers in prose,
    precisely to say they are forbidden here, and a check over raw source would read that
    sentence as the violation it prohibits.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(tree))


def test_no_hosted_module_can_expose_a_cookie():
    """The boundary is the whole point, so it is asserted over every module inside it."""
    from pipeline.ingestion import hosted_substack, hosted_x

    for module in (hosted_browser, hosted_x, hosted_substack):
        code = _code_without_prose(module)
        assert "Storage.getCookies" not in code, module.__name__
        assert "browser_cookies" not in code, module.__name__
        assert "read_x_cookies" not in code, module.__name__
        assert "read_substack_cookies" not in code, module.__name__
        # Added 2026-09-15 with the completion wait, which stats the cookie store to learn
        # WHEN Chrome last wrote it. Statting a file is not reading it, and this is the line
        # that keeps the difference honest: no hosted module may open that database.
        assert "sqlite3" not in code, module.__name__
    assert not hasattr(hosted_browser.ChromeRequestRunner, "get_cookies")


def test_chrome_request_result_is_typed_and_only_carries_allowed_rate_fields():
    ok = hosted_browser.ChromeRequestRunner._result({
        "status": 200,
        "data": {"data": {"bookmark_timeline_v2": {}}},
        "rate": {"remaining": "49", "reset": "123.5"},
    })
    assert ok.status is hosted_browser.ChromeRequestStatus.OK
    assert ok.data == {"data": {"bookmark_timeline_v2": {}}}
    assert ok.rate_limit == hosted_browser.RateLimit(remaining=49, reset_at=123.5)

    denied = hosted_browser.ChromeRequestRunner._result({"status": 403, "data": {"detail": "no"}})
    assert denied.status is hosted_browser.ChromeRequestStatus.UNAUTHENTICATED
    assert denied.data is None


def test_a_site_without_rate_limit_headers_reports_no_rate_limit():
    """One fetch script serves both sites; Substack publishes neither header."""
    result = hosted_browser.ChromeRequestRunner._result(
        {"status": 200, "data": {"subscriberLists": []}, "rate": {"remaining": None,
                                                                  "reset": None}})
    assert result.status is hosted_browser.ChromeRequestStatus.OK
    assert result.rate_limit is None


# ── native desktop lifecycle ─────────────────────────────────────────────────────────────────

class _DesktopProcess:
    def __init__(self, argv=(), **kwargs):
        self.argv, self.kwargs = list(argv), kwargs
        self.returncode = None
        self.actions: list[str] = []
        self.waits: list[float | None] = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.actions.append("TERM")
        self.returncode = -15

    def kill(self):
        self.actions.append("KILL")
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.returncode


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    """A `_DesktopLogin` whose display stack is stubbed, plus the processes it started."""
    xvfb = _DesktopProcess()
    started: list[_DesktopProcess] = []

    def popen(argv, **kwargs):
        proc = _DesktopProcess(argv, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(hosted_browser, "_profile_lock", hosted_browser.threading.Lock())
    monkeypatch.setattr(hosted_browser, "_shared_chrome", hosted_browser._SharedChrome())
    xvfb_size: list[tuple[int, int]] = []
    monkeypatch.setattr(hosted_browser, "_start_xvfb",
                        lambda width, height: (xvfb_size.append((width, height)),
                                               (":77", xvfb))[1])
    monkeypatch.setattr(hosted_browser, "_chrome_binary", lambda: tmp_path / "chrome")
    monkeypatch.setattr(hosted_browser, "_desktop_binary", lambda name: name)
    monkeypatch.setattr(hosted_browser.subprocess, "Popen", popen)
    monkeypatch.setattr(hosted_browser, "_wait_for_vnc", lambda proc, port: None)
    monkeypatch.setattr(hosted_browser, "_free_loopback_port", lambda: 59077)
    return xvfb, started, xvfb_size


def test_login_owns_an_isolated_desktop_without_a_debugger(desktop, tmp_path):
    xvfb, started, _sizes = desktop

    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))
    login.start()

    chrome, vnc = started
    assert chrome.kwargs["env"]["DISPLAY"] == ":77"
    assert "--remote-debugging-port=0" not in chrome.argv
    assert "--headless=new" not in chrome.argv
    assert f"--user-data-dir={tmp_path / 'chrome-profile'}" in chrome.argv
    assert vnc.argv[:5] == ["x11vnc", "-display", ":77", "-localhost", "-nopw"]
    assert ["-rfbport", "59077"] == vnc.argv[-2:]
    assert login.vnc_port == 59077

    login.close()

    assert vnc.actions == ["TERM"]
    assert chrome.actions == ["TERM"]
    assert xvfb.actions == ["TERM"]
    assert hosted_browser._profile_lock.locked() is False


def test_the_desktop_opens_the_site_the_capability_named(desktop):
    """The login URL travels with the nonce, so a Substack link cannot open x.com."""
    _xvfb, started, _sizes = desktop

    login = hosted_browser._DesktopLogin(
        "substack", "https://substack.com/sign-in", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))
    login.start()
    chrome, _vnc = started

    assert chrome.argv[-1] == "https://substack.com/sign-in"
    login.close()


def test_a_pending_sign_in_carries_its_site_from_mint_to_desktop(desktop, monkeypatch):
    manager = hosted_browser.HostedLoginManager()
    monkeypatch.setattr(hosted_browser, "_manager", manager)
    _xvfb, started, _sizes = desktop

    nonce = manager.create("substack", "https://substack.com/sign-in")
    manager.start(nonce)

    assert manager.current(nonce).site == "substack"
    assert started[0].argv[-1] == "https://substack.com/sign-in"
    manager.current(nonce).close()


def test_the_desktop_is_created_at_the_size_the_page_reported(desktop):
    """The whole reason the page starts its own desktop: one fixed size cannot fit two windows.

    noVNC scales the display into `#screen` and that element is given the display's aspect
    ratio, so the scale is exactly elementHeight / displayHeight. Unless Xvfb AND Chrome are
    both built at the reported box, the visitor gets a shrunken browser -- measured 2026-09-09
    at 0.71 on a laptop against 0.98 on a large monitor with a fixed 1600x900.
    """
    _xvfb, started, sizes = desktop

    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1138, 640))
    login.start()
    chrome, _vnc = started

    assert sizes == [(1138, 640)]
    assert "--window-size=1138,640" in chrome.argv
    login.close()


def test_a_login_desktop_refuses_chrome_s_130mb_of_per_profile_downloads(desktop):
    """Hosted, every user gets their own profile, and Chrome fills each one with the SAME
    130 MB of ML models and component payloads.

    Measured 2026-09-12 on the box, against a throwaway profile that had visited one sign-in
    page: 132 MB baseline against 5.5 MB with these flags, a 96% cut, and the two profiles held
    the same 15 cookies for the same hosts -- cookies being the only part of a profile OPYT
    reads. On a 639 GB disk that is the difference between roughly 4,000 and 30,000 users.
    """
    _xvfb, started, _sizes = desktop

    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(800, 600))
    login.start()
    chrome, _vnc = started

    for flag in cdp.LEAN_PROFILE_FLAGS:
        assert flag in chrome.argv
    login.close()


def test_no_chrome_is_launched_around_the_shared_argv_builder():
    """⚠️ A launch site that hand-rolls its argv costs 130 MB per user and looks fine.

    That is the failure mode worth a test: nothing breaks, no error is logged, and the disk
    fills up a year later. `--no-default-browser-check` is the marker every OPYT Chrome launch
    carries, so it existing OUTSIDE `cdp.chrome_argv` means someone built an argv by hand and
    the lean flags went with it.
    """
    builder = inspect.getsource(cdp.chrome_argv)
    assert "--no-default-browser-check" in builder

    for module in (cdp, hosted_browser):
        source = inspect.getsource(module)
        outside = source.count("--no-default-browser-check") - (module is cdp)
        assert outside == 0, (
            f"{module.__name__} builds a Chrome argv by hand; route it through "
            f"cdp.chrome_argv so it cannot miss LEAN_PROFILE_FLAGS")


def test_a_reported_desktop_size_is_clamped_before_anything_uses_it():
    """The size arrives from a public page, so it is bounded once and trusted after.

    The floor keeps a sign-in form usable when a small window reports a small box; the ceiling
    bounds the framebuffer a stranger can make this box allocate and encode.
    """
    size = hosted_browser.clamp_desktop_size(1138, 640)
    assert (size.width, size.height, size.pixels) == (1138, 640, (1138, 640))
    assert hosted_browser.clamp_desktop_size(9999, 9999).pixels[0] * \
        hosted_browser.clamp_desktop_size(9999, 9999).pixels[1] \
        <= hosted_browser._LOGIN_DESKTOP_MAX_PIXELS
    # A page that reports nothing usable falls back rather than failing the sign-in.
    assert hosted_browser.clamp_desktop_size(None, "tall") == hosted_browser.DesktopSize(
        hosted_browser._LOGIN_DESKTOP_WIDTH, hosted_browser._LOGIN_DESKTOP_HEIGHT, 1)


def test_a_phone_sized_box_grows_to_the_floor_and_keeps_its_shape():
    """The floor is measured twice over: X's sign-in needs 480 CSS pixels and Chrome refuses
    to open a window narrower than 500, which silently hangs the difference off the display.

    The growth has to be proportional. noVNC contain-fits the framebuffer into `#screen`, and
    the page gives that element the desktop's own aspect ratio, so a per-axis clamp would come
    back to the user as black bands down one side.
    """
    size = hosted_browser.clamp_desktop_size(390, 560)

    assert size.width == hosted_browser._LOGIN_DESKTOP_MIN_WIDTH
    assert abs(size.width / size.height - 390 / 560) < 0.01


def test_a_phone_gets_two_device_pixels_per_css_pixel_and_a_laptop_does_not():
    """The scale is what a phone screen needs and what the box can afford, in that order."""
    phone = hosted_browser.clamp_desktop_size(390, 560, 3)
    laptop = hosted_browser.clamp_desktop_size(1440, 790, 2)

    # Asking for 3 gets 2: past that the gain is small and the pixels are 2.25 times as many.
    assert phone.scale == 2
    assert phone.pixels == (phone.width * 2, phone.height * 2)
    # The same ceiling bounds both terms, so a laptop-sized viewport cannot reach 2 and the
    # desktop path everyone else is on is left exactly as it was.
    assert laptop.scale == 1
    assert laptop.pixels[0] * laptop.pixels[1] <= hosted_browser._LOGIN_DESKTOP_MAX_PIXELS


def test_an_absurd_shape_cannot_ask_this_box_for_an_enormous_screen():
    """Growing a reported box proportionally is only safe if its shape is bounded first.

    A page can report anything. Without the aspect bound, 20000x1 grows until its short side
    clears the floor -- a screen 200 000 pixels wide whose AREA is politely under the ceiling.
    """
    for width, height in [(20000, 1), (1, 20000), (0, 0), (-5, -5)]:
        size = hosted_browser.clamp_desktop_size(width, height)
        assert size.pixels[0] <= 4000 and size.pixels[1] <= 4000
        assert size.pixels[0] * size.pixels[1] <= hosted_browser._LOGIN_DESKTOP_MAX_PIXELS


def test_chrome_is_told_css_pixels_and_xvfb_device_pixels(desktop):
    """The trap this exists to stop: `--window-size` is CSS pixels, an X screen is device ones.

    Measured 2026-09-09 -- passing one number to both at scale 2 lays the page out at twice the
    width and clips half of it off the display.
    """
    _xvfb, started, sizes = desktop

    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(480, 690, 2))
    login.start()
    chrome, _vnc = started

    assert sizes == [(960, 1380)]
    assert "--window-size=480,690" in chrome.argv
    assert "--force-device-scale-factor=2" in chrome.argv
    login.close()


def test_a_reset_replaces_the_browser_and_nothing_under_it(desktop):
    """The way out of a browser window that has no way out.

    Measured on the box 2026-09-09: X's "Continue with Google" opens a SECOND Chrome window
    that covers the screen with no back button, no tab strip and, since the desktop runs no
    window manager, no title bar. Alt+Left in it does nothing. Nothing on that screen can
    dismiss it, so the escape is here -- and it replaces the browser ALONE, because the display
    and the VNC server are what the user is watching through and restarting those would end the
    sign-in this exists to rescue.
    """
    xvfb, started, _sizes = desktop
    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(500, 900))
    login.start()
    chrome, vnc = started

    login.restart_browser()

    replacement = started[-1]
    assert replacement is not chrome
    assert chrome.actions == ["TERM"]
    # Same profile, same size, same first page: a reset is the desktop as it opened, and the
    # cookies of anything already signed in are on disk rather than in the process.
    assert replacement.argv == chrome.argv
    assert vnc.actions == [] and xvfb.actions == []
    assert login.vnc_port == 59077

    login.close()

    assert replacement.actions == ["TERM"]
    with pytest.raises(hosted_browser.HostedBrowserError):
        login.restart_browser()


def test_desktop_process_is_waited_after_an_escalated_kill():
    class _StubbornProcess(_DesktopProcess):
        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise hosted_browser.subprocess.TimeoutExpired("desktop", timeout)
            return self.returncode

    proc = _StubbornProcess()
    hosted_browser._reap_process(proc)

    assert proc.actions == ["TERM", "KILL"]
    assert proc.waits == [10, None]


@pytest.fixture
def instant_settle(monkeypatch):
    """Run the durability wait at test speed. Its real deadlines are Chromium's, not ours."""
    monkeypatch.setattr(hosted_browser, "_LOGIN_LEAD_SECONDS", 0.0)
    monkeypatch.setattr(hosted_browser, "_LOGIN_SETTLE_SECONDS", 0.2)
    monkeypatch.setattr(hosted_browser, "_LOGIN_SETTLE_POLL_SECONDS", 0.01)


def test_completion_closes_the_desktop_before_using_the_request_browser(
        monkeypatch, instant_settle):
    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))
    closed = []
    monkeypatch.setattr(login, "close", lambda: closed.append(True))
    monkeypatch.setattr(hosted_browser, "site_connected", lambda site: site == "x")

    assert login.complete() == "connected"
    assert closed == [True]


def test_completion_validates_the_site_that_signed_in(monkeypatch, instant_settle):
    """A Substack desktop must not be confirmed by X's viewer request, or every Substack
    sign-in reads as connected on a home whose X profile is live."""
    asked: list[str] = []
    monkeypatch.setattr(hosted_browser, "site_connected",
                        lambda site: asked.append(site) or False)
    login = hosted_browser._DesktopLogin(
        "substack", "https://substack.com/sign-in", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))

    assert login.complete() == "not_connected"
    assert asked == ["substack"]


def test_completion_waits_for_the_sign_in_to_be_written_before_it_closes_the_browser(
        monkeypatch, instant_settle):
    """The measured bug, 2026-09-15: Chromium holds a new cookie in memory for ~30 s and no
    signal makes it flush on the way out, so closing the browser at the click reported "not
    connected" for sign-ins that had really succeeded — and the promptest users lost most."""
    monkeypatch.setattr(hosted_browser, "_LOGIN_SETTLE_SECONDS", 5.0)
    events: list[str] = []
    writes = [("store", 1, 10), ("store", 1, 10), ("store", 2, 20)]

    def signature():
        events.append("poll")
        return writes[0] if len(writes) == 1 else writes.pop(0)

    monkeypatch.setattr(hosted_browser, "_cookie_store_signature", signature)
    monkeypatch.setattr(hosted_browser, "site_connected", lambda site: True)
    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))
    monkeypatch.setattr(login, "close", lambda: events.append("close"))

    assert login.complete() == "connected"
    # Polled until the store changed, and only then closed the browser holding the session.
    assert events.count("poll") >= 3
    assert events[-1] == "close"


def test_completion_stops_waiting_at_the_deadline_and_still_answers(monkeypatch, instant_settle):
    """A desktop closed without signing in writes nothing, so the wait has to end by itself —
    and the verdict stays the validation's to give, not the timeout's."""
    monkeypatch.setattr(hosted_browser, "_cookie_store_signature", lambda: ("unchanged",))
    monkeypatch.setattr(hosted_browser, "site_connected", lambda site: False)
    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() + 60,
        hosted_browser.DesktopSize(1280, 720))
    monkeypatch.setattr(login, "close", lambda: None)

    started = hosted_browser.time.monotonic()
    assert login.complete() == "not_connected"
    assert hosted_browser.time.monotonic() - started >= 0.2


def test_an_expired_desktop_is_not_waited_on(monkeypatch, instant_settle):
    """Nothing is signing in on a desktop whose window closed, so it pays no settle."""
    polled = []
    monkeypatch.setattr(hosted_browser, "_cookie_store_signature",
                        lambda: polled.append(True) or ())
    monkeypatch.setattr(hosted_browser, "site_connected", lambda site: True)
    login = hosted_browser._DesktopLogin(
        "x", "https://x.com/login", hosted_browser.time.monotonic() - 1,
        hosted_browser.DesktopSize(1280, 720))
    monkeypatch.setattr(login, "close", lambda: None)

    assert login.complete() == "expired"
    assert polled == []


def test_a_commit_to_the_store_or_any_sidecar_changes_the_signature(monkeypatch, tmp_path):
    """What the wait watches. A journal-mode commit rewrites the sidecar rather than the
    database, so a signature taken from the database alone would miss the write."""
    monkeypatch.setattr(hosted_browser, "profile_dir", lambda: tmp_path)
    store = tmp_path / "Default" / "Cookies"
    store.parent.mkdir(parents=True)

    assert hosted_browser._cookie_store_signature() == ()
    store.write_bytes(b"cookies")
    first = hosted_browser._cookie_store_signature()
    assert first != ()

    store.write_bytes(b"cookies and one more")
    assert hosted_browser._cookie_store_signature() != first

    second = hosted_browser._cookie_store_signature()
    store.with_name("Cookies-journal").write_bytes(b"journal")
    assert hosted_browser._cookie_store_signature() != second


# ── binary RFB child relay ───────────────────────────────────────────────────────────────────

class _RelayLogin:
    vnc_port = 59077

    def close(self):
        pass


def _child_app(login):
    from starlette.applications import Starlette

    hosted_browser._manager._active, hosted_browser._manager._active_nonce = login, "n1"
    return hosted_browser.hosted_child_app(Starlette(routes=[]))


def test_the_rfb_relay_refuses_a_socket_without_the_child_interaction_key(monkeypatch):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("OPYT_HOSTED_INTERACTION_KEY", "child-key")
    client = TestClient(_child_app(_RelayLogin()))

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/_hosted-login/n1/stream") as socket:
            socket.receive_bytes()


def test_completion_endpoint_uses_the_desktop_owner(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("OPYT_HOSTED_INTERACTION_KEY", "child-key")
    monkeypatch.setattr(hosted_browser._manager, "complete", lambda nonce: "connected")
    client = TestClient(_child_app(_RelayLogin()))

    response = client.post("/_hosted-login/n1/complete",
                           headers={"X-Opyt-Hosted-Interaction-Key": "child-key"})

    assert response.json() == {"status": "connected"}


def test_a_reset_needs_the_child_interaction_key(monkeypatch):
    """It restarts a browser holding a live sign-in, so it is not reachable without the key."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("OPYT_HOSTED_INTERACTION_KEY", "child-key")
    reset: list[str] = []
    monkeypatch.setattr(hosted_browser._manager, "restart", reset.append)
    client = TestClient(_child_app(_RelayLogin()))

    assert client.post("/_hosted-login/n1/restart").status_code == 404
    assert reset == []

    response = client.post("/_hosted-login/n1/restart",
                           headers={"X-Opyt-Hosted-Interaction-Key": "child-key"})

    assert response.json() == {"status": "restarted"}
    assert reset == ["n1"]


def test_a_refused_login_start_names_its_failing_step_in_the_child_log(monkeypatch, capsys):
    """The generic 404 is the whole of what the gateway learns, so the child must record why.

    Without this the only evidence of a failed hosted sign-in is an unexplained page error —
    measured on the InterServer box on 2026-09-07, where neither log held anything at all.
    """
    from starlette.testclient import TestClient

    def refuse(_nonce, _size=None):
        raise hosted_browser.HostedBrowserError(
            "x11vnc exited before accepting the login desktop")

    monkeypatch.setenv("OPYT_HOSTED_INTERACTION_KEY", "child-key")
    monkeypatch.setattr(hosted_browser._manager, "start", refuse)
    client = TestClient(_child_app(_RelayLogin()))
    capsys.readouterr()

    response = client.post("/_hosted-login/n1/start",
                           headers={"X-Opyt-Hosted-Interaction-Key": "child-key"})

    assert response.status_code == 404
    assert "x11vnc" not in response.text
    assert "x11vnc" in capsys.readouterr().err


# ── the public sign-in page ────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_script_parses_as_a_module(tmp_path):
    """The page's script is one module: a syntax error anywhere in it disables the whole page.

    This is not hypothetical. An edit to the retired paste page once introduced a second
    `const response` in one scope, and every button on it stopped responding -- with every
    substring assertion in this file still passing, because the text was all present and only
    the JavaScript was dead.
    """
    html = gateway_app._login_html("x", "s", "c")
    script = re.search(r"<script type=module>(.*?)</script>", html, re.S)
    assert script is not None, "the page has no module script"
    source = tmp_path / "login.mjs"
    source.write_text(script.group(1))

    check = subprocess.run(["node", "--check", str(source)], capture_output=True, text=True)

    assert check.returncode == 0, check.stderr


def test_login_page_uses_vendored_novnc_not_the_retired_canvas_surface():
    """The page draws its desktop through `/static/desktop.js`, and that module is noVNC.

    Both sign-in pages import the module rather than noVNC directly: it carries the keyboard a
    phone can raise and the reconnect a phone needs, and one copy of either is the point.
    """
    page = gateway_app._login_html("x", "stream-capability", "completion-capability")
    module = (Path(gateway_app.__file__).parent / "static" / "desktop.js").read_text()

    assert "/static/desktop.js" in page
    assert "/static/novnc/core/rfb.js" in module
    assert "/login/session/stream-capability" in page
    assert "/login/complete/completion-capability" in page
    assert "<canvas" not in page and "Input.dispatch" not in page


def test_the_desktop_page_carries_a_way_out_and_a_way_to_type():
    """The two things a phone could not do, as controls a phone can see.

    A canvas raises no on-screen keyboard and a hosted desktop has no browser chrome a thumb
    can hit, so the page carries its own. The reset addresses the stream capability, not a
    new one: whoever can move the mouse on that desktop can already do everything a restart
    does.
    """
    html = gateway_app._login_html("x", "stream-capability", "c")

    assert "id=keyboard" in html and 'data-hide="Hide keyboard"' in html
    assert "id=restart" in html
    # The stage is what the desktop is seen through while the keyboard covers half the screen.
    # Dropping it changes nothing a substring test would otherwise notice: the page keeps
    # working and the desktop silently goes back to being unreadable while typing.
    assert "stage:" in html
    assert "/login/restart/stream-capability" in html


def test_the_desktop_page_names_the_site_it_is_connecting():
    """A Substack sign-in that said "Connect your X account" is the failure this replaces.

    Both sites reach this one page since 2026-09-15, so the per-site substitution carries the
    whole difference between them — which is now only the name. Neither site carries a note:
    nothing about signing in to either needs saying that the site's own page does not already
    say, and Substack's "don't tap the email's link" warning was removed on 2026-09-16 (its
    stated consequence was never measured in that direction).
    """
    x_page = gateway_app._login_html("x", "s", "c")
    substack_page = gateway_app._login_html("substack", "s", "c")

    assert "Connect your X account" in x_page
    assert "Connect your Substack account" in substack_page
    # An empty note must leave NO element behind, or the actions row grows a blank line above
    # its buttons. `.note:empty` is what does that, and it only works if the substitution puts
    # nothing at all between the tags.
    assert "<p class=note></p>" in substack_page and "<p class=note></p>" in x_page
    assert ".note:empty { display:none; }" in substack_page
    assert "Don't tap the email's link" not in substack_page


def test_every_wait_on_the_page_spins_and_every_prompt_does_not():
    """The completion check holds the page for up to 45 s while Chrome commits the session to
    disk, and David read a still sentence as a hung tab (2026-09-16). The rule the spinner
    encodes: waiting on the SERVER spins, asking the USER to act does not.

    Asserted through `setStatus`, because the trap here is a later edit writing
    `status.textContent` directly — which silently deletes the spinner element, since it is a
    child of the status line, and leaves a wait looking exactly as frozen as before.
    """
    page = gateway_app._login_html("substack", "s", "c")

    assert "status.textContent =" not in page, "a status write that bypasses setStatus"
    # The three server waits, each spinning.
    for waiting in ("Opening a private sign-in desktop…", "Reconnecting to the sign-in desktop…",
                    "Checking your Substack connection"):
        line = next(l for l in page.splitlines() if waiting in l)
        assert line.rstrip().endswith(", true);"), f"this wait does not spin: {waiting}"
    # ...and the prompt that is waiting on the person, which must not.
    assert "setStatus(SIGN_IN_STEPS);" in page
    # The animation is not mandatory, but it is not dropped either: a motion-sensitive user
    # needs the progress signal most during the longest wait, so it slows instead of stopping.
    assert "prefers-reduced-motion" in page and "animation-duration:2.4s" in page


def test_a_sign_in_profile_is_told_to_keep_session_cookies(tmp_path, monkeypatch):
    """⚠️ The reason hosted sign-in did not stick, measured on the box 2026-09-12.

    A site's auth cookie is session-scoped and Chrome drops session cookies at shutdown unless
    the profile resumes its session. Both sign-in paths read the profile only AFTER the browser
    that signed in is gone, so each was creating a real session and throwing it away: Substack
    ACCEPTED the pasted link — its /sign-in hop carried no `error=`, where a refusal carries
    `error=Login+link+expired` — and the profile still held no session cookie.
    """
    from pipeline.ingestion import hosted_browser

    profile = tmp_path / "chrome-profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text(json.dumps({"other": "kept"}))
    monkeypatch.setattr(hosted_browser, "profile_dir", lambda: profile)

    hosted_browser._persist_session_cookies()
    prefs = json.loads((profile / "Default" / "Preferences").read_text())

    assert prefs["session"]["restore_on_startup"] == 1   # "continue where you left off"
    assert prefs["session"]["startup_urls"] == []        # ...without reopening old tabs
    assert prefs["other"] == "kept"                      # nothing else in the file is touched


def test_a_profile_chrome_has_never_opened_does_not_crash_the_sign_in(tmp_path, monkeypatch):
    """Fail-safe: Chrome writes Preferences itself on first run, so a missing file is a normal
    state and not a reason to refuse a sign-in."""
    from pipeline.ingestion import hosted_browser

    monkeypatch.setattr(hosted_browser, "profile_dir", lambda: tmp_path / "never-opened")
    hosted_browser._persist_session_cookies()

    prefs = json.loads((tmp_path / "never-opened" / "Default" / "Preferences").read_text())
    assert prefs["session"]["restore_on_startup"] == 1


def test_no_escape_in_a_page_template_is_eaten_before_the_browser_sees_it():
    """⚠️ The trap that cost a debugging round on 2026-09-11, and will again.

    `_LOGIN_PAGE` is an ordinary triple-quoted Python string, so a backslash written into the
    JavaScript is interpreted at IMPORT time, not by the browser. Writing `.join('\\n')` put a
    REAL newline inside a JS string literal, which is a syntax error, which silently killed
    the ENTIRE module -- no handler anywhere on the page was wired, and nothing said so.
    `\\\\` and `\\u` are fine: the first survives as one backslash, the second is a deliberate
    character. Anything else is almost certainly meant for the browser.
    """
    source = Path(gateway_app.__file__).read_text()
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != "_LOGIN_PAGE":
            continue
        eaten = set(re.findall(r"\\(.)", ast.get_source_segment(source, node.value))) - {"\\", "u"}
        assert not eaten, (
            f"{node.targets[0].id} has escape(s) Python eats before the browser sees them: "
            f"{sorted(eaten)}. Double the backslash, or avoid needing one.")


def test_a_login_page_failure_reaches_the_visitor_without_the_server_reason():
    """This URL is opened in a browser from a chat, so its failures must read as a page."""
    used = gateway_app._login_error_html_response(
        "substack", "This sign-in link was already used or has expired.", 404)
    unstarted = gateway_app._login_error_html_response(
        "x", "The sign-in desktop could not start.", 503)
    unknown = gateway_app._login_error_html_response(None, "That sign-in link is not valid.", 404)

    assert (used.status_code, unstarted.status_code) == (404, 503)
    for response in (used, unstarted, unknown):
        body = response.body.decode()
        assert response.media_type == "text/html"
        assert response.headers["Cache-Control"] == "no-store"
        # A page, not a JSON blob, and it names a way forward. The wording is not pinned: an
        # earlier version asserted one sentence of it and blocked a copy fix for no gain.
        assert body.startswith("<!doctype html>") and "<h1>" in body
    # The one thing worth pinning is that a failure names the site the visitor was connecting,
    # since "start the sign-in again" is useless if they cannot tell which one failed.
    assert "Substack" in used.body.decode()
    assert "X" in unstarted.body.decode()


# ── reused post-login request browser ──────────────────────────────────────────

class _FakeRunner:
    launches = 0
    is_alive = True

    def __init__(self):
        self._running = False

    def start(self):
        type(self).launches += 1
        self._running = True

    def close(self):
        self._running = False

    @property
    def running(self):
        return self._running

    def alive(self):
        return self._running and self.is_alive


@pytest.fixture
def fake_chrome(monkeypatch):
    _FakeRunner.launches, _FakeRunner.is_alive = 0, True
    monkeypatch.setattr(hosted_browser, "ChromeRequestRunner", _FakeRunner)
    monkeypatch.setattr(hosted_browser, "_shared_chrome", hosted_browser._SharedChrome())
    monkeypatch.setattr(hosted_browser, "_profile_lock", hosted_browser.threading.Lock())
    yield _FakeRunner
    hosted_browser._shared_chrome.close()


def test_a_burst_of_requests_shares_one_browser(fake_chrome):
    for _ in range(5):
        with hosted_browser.shared_chrome():
            pass
    assert fake_chrome.launches == 1


def test_the_shared_browser_is_reaped_after_its_idle_window(fake_chrome, monkeypatch):
    monkeypatch.setattr(hosted_browser, "_CHROME_IDLE_SECONDS", 0.05)
    with hosted_browser.shared_chrome():
        pass

    deadline = hosted_browser.time.monotonic() + 3
    while (hosted_browser._shared_chrome._runner is not None
           and hosted_browser.time.monotonic() < deadline):
        hosted_browser.time.sleep(0.02)

    assert hosted_browser._shared_chrome._runner is None
    with hosted_browser.shared_chrome():
        pass
    assert fake_chrome.launches == 2


def test_a_browser_that_died_in_the_idle_window_is_replaced(fake_chrome):
    with hosted_browser.shared_chrome():
        pass
    _FakeRunner.is_alive = False
    with hosted_browser.shared_chrome():
        pass
    assert fake_chrome.launches == 2


def test_a_failed_request_drops_the_browser_rather_than_reusing_it(fake_chrome):
    with pytest.raises(ValueError):
        with hosted_browser.shared_chrome():
            raise ValueError("request blew up")
    assert hosted_browser._shared_chrome._runner is None


def test_a_login_and_a_pull_never_hold_the_same_chrome_profile(fake_chrome):
    with hosted_browser.shared_chrome():
        pass
    hosted_browser._profile_lock.acquire()
    try:
        original = hosted_browser._PROFILE_WAIT_SECONDS
        hosted_browser._PROFILE_WAIT_SECONDS = 0.05
        with pytest.raises(hosted_browser.HostedBrowserError):
            with hosted_browser.shared_chrome():
                pass
    finally:
        hosted_browser._PROFILE_WAIT_SECONDS = original
        hosted_browser._profile_lock.release()


# ── one page per site origin ───────────────────────────────────────────────────

class _PageBrowser:
    """A CDP browser socket that records the targets a runner opens."""

    def __init__(self):
        self.created: list[str] = []

    def call(self, method, params=None, session_id=None):
        if method == "Target.createTarget":
            self.created.append(params["url"])
            return {"targetId": f"t{len(self.created)}"}
        if method == "Target.attachToTarget":
            return {"sessionId": f"s{len(self.created)}"}
        return {}

    def wait_for_event(self, method, *, session_id=None, timeout=None):
        return {}


def test_each_site_gets_its_own_page_and_keeps_it(monkeypatch):
    """A credentialed cross-origin fetch is subject to the TARGET's CORS policy, so a
    substack.com request issued from an x.com page is refused by Substack. One page per origin
    is what lets one browser serve both, and caching it is what stops a per-request navigation.
    """
    runner = hosted_browser.ChromeRequestRunner()
    browser = _PageBrowser()
    runner._browser = browser
    navigated: list[str] = []
    monkeypatch.setattr(hosted_browser.cdp.Page, "call",
                        lambda self, method, params=None:
                        navigated.append(params["url"]) if method == "Page.navigate" else {})
    monkeypatch.setattr(hosted_browser.cdp.Page, "wait_for_event",
                        lambda self, method, **kwargs: {})

    first = runner.page("https://x.com/home")
    again = runner.page("https://x.com/home")
    other = runner.page("https://substack.com/inbox")

    assert first is again
    assert other is not first
    assert navigated == ["https://x.com/home", "https://substack.com/inbox"]


def test_a_chrome_that_died_fails_the_sign_in():
    """Measured 2026-09-08: Chrome refuses a profile another LIVE Chrome holds by handing its
    URL to that instance and exiting 21 — at 152 ms, while `_wait_for_vnc` returns at ~150 ms.
    A single `poll()` lost that race about half the time, and the loser got a working desktop
    with nothing drawn on it: Xvfb and x11vnc healthy, the RFB stream connected, no error
    anywhere, and a black rectangle until the link expired.
    """
    import subprocess
    import sys

    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(21)"])
    dead.wait()
    with pytest.raises(hosted_browser.HostedBrowserError):
        hosted_browser._require_chrome_survived(dead)


def test_a_chrome_that_stays_up_opens_the_sign_in():
    import subprocess
    import sys

    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        hosted_browser._require_chrome_survived(alive)      # returns, does not raise
    finally:
        alive.kill()
        alive.wait()


# ── the profile a dead child left behind ─────────────────────────────────────────────────

def _fake_chrome(tmp_path, profile):
    """A live process whose command line looks like Chrome on `profile`.

    The reaper decides by command line, not by the symlink, so a test of that decision needs a
    real process to read — not a mock of the reading.
    """
    script = tmp_path / "chrome"
    script.write_text("#!/bin/sh\nsleep 60\n")
    script.chmod(0o755)
    return subprocess.Popen([str(script), f"--user-data-dir={profile}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _hold_profile(profile, pid):
    """Write the `SingletonLock` Chrome itself writes: `<hostname>-<pid>`."""
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "SingletonLock").symlink_to(f"somehost-{pid}")


def test_a_chrome_left_by_a_dead_child_is_found_through_its_profile_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    profile = hosted_browser.profile_dir()
    stray = _fake_chrome(tmp_path, profile)
    try:
        _hold_profile(profile, stray.pid)
        assert hosted_browser._profile_holder_pid() == stray.pid
    finally:
        stray.kill()
        stray.wait()


def _lease_taken_by_another_process(home: Path) -> bool:
    """Can a SEPARATE process take this home's profile lease right now?

    A second process, not a second descriptor: the whole point of the lease is the boundary a
    `threading.Lock` cannot see across, and only a real fork proves it holds there.
    """
    probe = ("import fcntl, os, sys\n"
             "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
             "try:\n"
             "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
             "except OSError:\n"
             "    raise SystemExit(1)\n"
             "raise SystemExit(0)\n")
    lock_file = home / hosted_browser._PROFILE_LOCK_FILE
    return subprocess.run([sys.executable, "-c", probe, str(lock_file)]).returncode == 0


def test_a_rail_child_cannot_take_a_profile_the_mcp_child_is_driving(monkeypatch, tmp_path):
    """The 2026-09-15 incident, from the intruder's side.

    Two processes drive one home — the resident child in-process and a rail child the worker
    launched — and before the lease the second one SIGTERMed the first one's healthy Chrome.
    """
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    lock = hosted_browser._ProfileLock()

    assert _lease_taken_by_another_process(tmp_path) is True   # free before anyone holds it
    assert lock.acquire(timeout=1) is True
    try:
        assert _lease_taken_by_another_process(tmp_path) is False
    finally:
        lock.release()
    assert _lease_taken_by_another_process(tmp_path) is True


def test_a_home_that_does_not_exist_yet_still_gets_a_real_lease(monkeypatch, tmp_path):
    """Proved on the box 2026-09-15: pointed at a home that was not there, the lease logged its
    degrade and BOTH processes walked in. A silent fallback to no exclusion is the bug."""
    home = tmp_path / "not-created-yet"
    monkeypatch.setenv("OPYT_HOME", str(home))
    lock = hosted_browser._ProfileLock()
    assert lock.acquire(timeout=1) is True
    try:
        assert (home / hosted_browser._PROFILE_LOCK_FILE).exists()
        assert _lease_taken_by_another_process(home) is False
    finally:
        lock.release()


def test_a_waiter_that_times_out_holds_nothing(monkeypatch, tmp_path):
    """A refused lease must leave the thread half free, or the next caller deadlocks on it."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    holder, waiter = hosted_browser._ProfileLock(), hosted_browser._ProfileLock()
    assert holder.acquire(timeout=1) is True
    try:
        assert waiter.acquire(timeout=0.1) is False
        assert waiter.locked() is False
    finally:
        holder.release()
    assert waiter.acquire(timeout=1) is True
    waiter.release()


def test_the_claim_never_runs_outside_the_lease(monkeypatch, tmp_path):
    """`_claim_profile` ends whatever holds the profile, so it may only run under the lease —
    that is what proves the holder is dead rather than a sibling mid-pull."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(hosted_browser, "_profile_lock", hosted_browser._ProfileLock())
    monkeypatch.setattr(hosted_browser, "_shared_chrome", hosted_browser._SharedChrome())
    monkeypatch.setattr(hosted_browser, "_persist_session_cookies", lambda: None)
    monkeypatch.setattr(hosted_browser, "_chrome_binary", lambda: Path("/nonexistent/chrome"))

    @contextlib.contextmanager
    def _no_browser(*_args, **_kwargs):
        yield object()
    monkeypatch.setattr(hosted_browser.cdp, "controlled_browser", _no_browser)

    # The real `ChromeRequestRunner.start` runs, so this records the lease state at the exact
    # moment the claim would SIGTERM whatever it finds.
    held: list[bool] = []
    monkeypatch.setattr(hosted_browser, "_claim_profile",
                        lambda: held.append(not _lease_taken_by_another_process(tmp_path)))
    try:
        with hosted_browser.shared_chrome():
            pass
    finally:
        hosted_browser._shared_chrome.close()
    assert held == [True]


# ── what a contended connect leaves in the log ───────────────────────────────────
#
# §3 of `docs/plans/2026-09-16-hosted-chrome-contention-handoff.md` asks how long two contenders
# actually overlap on a real connect and how many Chrome relaunches that costs, and rules that
# nothing be built until that is measured. It could not be: a wait and an instant acquire left
# identical evidence, and so did a relaunch and a cold start. These three cover the instrument
# that answers it, because a log line nothing asserts on is a log line that quietly stops.

def _captured_log(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(hosted_browser, "log", lines.append)
    return lines


def _no_real_chrome(monkeypatch, tmp_path) -> None:
    """`shared_chrome()` with everything below CDP stubbed out — the recipe
    `test_the_claim_never_runs_outside_the_lease` uses, which exercises the real lease."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(hosted_browser, "_profile_lock", hosted_browser._ProfileLock())
    monkeypatch.setattr(hosted_browser, "_shared_chrome", hosted_browser._SharedChrome())
    monkeypatch.setattr(hosted_browser, "_persist_session_cookies", lambda: None)
    monkeypatch.setattr(hosted_browser, "_claim_profile", lambda: None)
    monkeypatch.setattr(hosted_browser, "_chrome_binary", lambda: Path("/nonexistent/chrome"))

    class _AnsweringBrowser:
        """Answers `Browser.getVersion`, which is all `ChromeRequestRunner.alive()` asks of it —
        so a second `shared_chrome()` on one runner reaches the reuse path instead of dying in
        the liveness probe."""

        def call(self, *_args, **_kwargs):
            return {}

    @contextlib.contextmanager
    def _no_browser(*_args, **_kwargs):
        yield _AnsweringBrowser()
    monkeypatch.setattr(hosted_browser.cdp, "controlled_browser", _no_browser)


def test_a_wait_for_a_siblings_profile_is_logged_with_its_duration(monkeypatch, tmp_path):
    """The overlap half of §3's number. A rail that waited 9s for the MCP child and one that
    walked straight in are the same two log lines without this, so the connect minute could be
    described but never costed."""
    _no_real_chrome(monkeypatch, tmp_path)
    lines = _captured_log(monkeypatch)

    holder = hosted_browser._ProfileLock()
    assert holder.acquire(timeout=1) is True
    released = threading.Timer(0.6, holder.release)
    released.daemon = True
    released.start()
    try:
        with hosted_browser.shared_chrome():
            pass
    finally:
        released.cancel()
        hosted_browser._shared_chrome.close()

    waits = [ln for ln in lines if "waited" in ln and "release this home's profile" in ln]
    assert len(waits) == 1
    # The DURATION, not just the fact. A boolean "there was contention" cannot answer §3.
    assert re.search(r"waited (\d+\.\d)s", waits[0])
    assert float(re.search(r"waited (\d+\.\d)s", waits[0]).group(1)) >= 0.5


def test_an_uncontended_acquire_stays_silent(monkeypatch, tmp_path):
    """The other half, and it is what keeps the first one readable. A burst takes and releases
    the lease once per request, so a line per acquire would bury the contention it exists to
    show under the traffic of a pull that had none."""
    _no_real_chrome(monkeypatch, tmp_path)
    lines = _captured_log(monkeypatch)
    try:
        for _ in range(3):
            with hosted_browser.shared_chrome():
                pass
    finally:
        hosted_browser._shared_chrome.close()

    assert not [ln for ln in lines if "waited" in ln]


def test_a_relaunch_after_an_eviction_reads_differently_from_a_cold_start(monkeypatch):
    """The relaunch half of §3's number, and the distinction is the whole value. A sibling that
    took the lease correctly ended this process's idle browser; the next `acquire()` pays 3-6s
    (measured 2026-09-07, slower on a 1-2 vCPU VPS) to get it back. A cold start pays the same
    seconds for a browser nobody destroyed, so counting launches alone measures nothing."""
    lines = _captured_log(monkeypatch)

    class _FakeRunner:
        def __init__(self) -> None:
            self.living = True

        def start(self) -> None:
            pass

        def alive(self) -> bool:
            return self.living

        def close(self) -> None:
            self.living = False

    made: list[_FakeRunner] = []
    monkeypatch.setattr(hosted_browser, "ChromeRequestRunner",
                        lambda: made.append(_FakeRunner()) or made[-1])

    shared = hosted_browser._SharedChrome()
    first = shared.acquire()
    assert [ln for ln in lines if "started" in ln] and not [ln for ln in lines if "RELAUNCH" in ln]

    shared.release()
    first.living = False              # a sibling took the lease and ended this idle browser
    lines.clear()
    assert shared.acquire() is not first
    assert len([ln for ln in lines if "RELAUNCHED after eviction" in ln]) == 1
    shared.close()


def test_a_reused_pid_running_something_else_is_never_claimed(monkeypatch, tmp_path):
    """PIDs are reused, and this answer gets a process killed."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    profile = hosted_browser.profile_dir()
    other = subprocess.Popen(["sleep", "60"])
    try:
        _hold_profile(profile, other.pid)
        assert hosted_browser._profile_holder_pid() is None
    finally:
        other.kill()
        other.wait()


def test_a_chrome_on_a_different_home_is_never_claimed(monkeypatch, tmp_path):
    """One box runs one profile per user. Reaping by name alone would cross homes."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    profile = hosted_browser.profile_dir()
    stray = _fake_chrome(tmp_path, tmp_path / "someone-elses-profile")
    try:
        _hold_profile(profile, stray.pid)
        assert hosted_browser._profile_holder_pid() is None
    finally:
        stray.kill()
        stray.wait()


def test_a_holder_that_has_already_exited_is_not_claimed(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    profile = hosted_browser.profile_dir()
    gone = subprocess.Popen(["true"])
    gone.wait()
    _hold_profile(profile, gone.pid)
    assert hosted_browser._profile_holder_pid() is None


def test_no_lock_and_a_malformed_lock_both_mean_nobody_holds_the_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    profile = hosted_browser.profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    assert hosted_browser._profile_holder_pid() is None
    (profile / "SingletonLock").symlink_to("somehost-not-a-pid")
    assert hosted_browser._profile_holder_pid() is None


