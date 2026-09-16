"""The hosted browser boundary: one Chrome profile, one interactive login, no credential ever
crossing into Python.

Site-agnostic. Chrome alone uses the persistent profile to make each site's narrowly-defined
authenticated requests; this module never opens a cookie database and never asks CDP for
cookies. `hosted_x` and `hosted_substack` are its two consumers, and each owns the URLs,
headers and validation request of its own site.

The local cookie transports (`x_graphql_core.read_x_cookies`,
`sources.substack.read_substack_cookies`) remain for local installs only, and both refuse to
run in a hosted child.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import select
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:                       # POSIX only, which is every platform the hosted boundary runs on
    import fcntl
except ImportError:        # pragma: no cover - a local Windows install never reaches this module
    fcntl = None           # type: ignore[assignment]

from opyt_core.paths import opyt_path
from pipeline.ingestion import cdp
from pipeline.ingestion.utils import log

# Two deadlines, because they bound two different things. The LINK deadline stops a minted
# capability being redeemed long after it was handed out. The DESKTOP deadline bounds how long a
# running desktop — an Xvfb, an x11vnc and a Chrome — may hold the box's memory.
#
# They were one value until 2026-09-08, and the desktop inherited the mint stamp. Every second
# spent returning the link to the user came out of the time they had to sign in, and the connector
# round trip that returns it has been measured above 120 s, so a user routinely opened a desktop
# with seven of its ten minutes left. Measured failure: `docs/plans/
# 2026-09-08-hosted-substack-signin.md`.
_LOGIN_LINK_TTL_SECONDS = 10 * 60
_LOGIN_DESKTOP_TTL_SECONDS = 10 * 60
# ── When "Done signing in" is allowed to close the browser ───────────────────────────────────
#
# ⚠️ Closing this Chrome THROWS AWAY a session it has not written down yet, and both halves of
# that were measured on the box 2026-09-15. Chromium batches cookie writes on a ~30 s commit
# timer: a profile opened at x.com held 0 rows on disk at t=30 s and 6 at t=35 s, and a later
# navigation's cookies appeared 16 s after it. A graceful shutdown is NOT a flush — SIGTERM,
# SIGINT, SIGHUP and SIGQUIT each left the store at 0 rows and `exit_type` at "Crashed" — the
# same reading the retired paste path's settle wait recorded on 2026-09-12.
#
# So completion cannot be "close the desktop and read the profile". It is: give the site's own
# sign-in response a moment to land, WAIT FOR THE STORE TO BE WRITTEN, and only then close. Any
# one commit writes every cookie pending at that moment, so the first write after the lead-in
# carries the session whatever else it carries.
#
# The failure this repairs, reported 2026-09-15: pressing X's "Continue" and then "Done signing
# in" straight away answered "X is not connected" for a sign-in that had really succeeded — the
# promptest users were the ones who could not connect. A desktop's user may have signed in a
# minute ago or a second ago, so this watches for the write instead of serving everybody the
# worst case with a flat sleep.
_LOGIN_LEAD_SECONDS = 10.0
# The commit interval with margin. Reaching it means nothing was pending, which is what a
# desktop closed without signing in looks like; shortening it below the interval turns a real
# sign-in back into a silent loss rather than a slow answer.
_LOGIN_SETTLE_SECONDS = 45.0
_LOGIN_SETTLE_POLL_SECONDS = 1.0
# The isolated desktop is a whole sign-in screen, so it must be large enough for a desktop
# login card while remaining cheap enough for a short-lived session.
#
# Width and height are separate because Chrome is told the same numbers. Left to itself Chrome
# opens a 1050x880 window and leaves the rest of the display black -- measured 2026-09-09, which
# is a quarter of the desktop the visitor is looking at.
#
# The FALLBACK size only. The real one comes from the client: the page measures the box it
# will draw the desktop into and reports it, and the desktop is created at exactly that size.
#
# Why it has to come from the client. noVNC scales the whole display into `#screen`
# (`rfb.scaleViewport`, gateway/app.py) and that element is given the display's own aspect
# ratio, so both terms of min(elementW/displayW, elementH/displayH) are equal and the scale
# reduces to elementHeight / displayHeight. Any fixed height is therefore 1:1 for exactly one
# window and shrinks every other. Measured 2026-09-09: at 1600x900 a 1695x1060 viewport got
# 0.98 while a 1440x790 laptop got 0.71. Matching the display to the element makes it 1.0 for
# both. Full record: docs/plans/2026-09-09-hosted-desktop-measured-tuning.md.
_LOGIN_DESKTOP_WIDTH = 1600
_LOGIN_DESKTOP_HEIGHT = 900
# The narrowest viewport a login desktop is built at. Two measurements, both on this box,
# both 2026-09-09, and the larger one wins:
#   * X's sign-in needs 480 CSS px. Photographed at 390, 440, 480 and 900: below 480 its
#     buttons and terms line run off the right edge.
#   * Chrome will not make a browser window narrower than 500 CSS px. Asked for 440, 480 and
#     500 it opened 1000 device pixels wide every time (at scale 2), and 1040 for 520. On a
#     960-pixel screen that put 40 pixels of the window, including the menu button, off the
#     right edge of the display, where no user can reach them.
# It is a floor on the WIDTH alone: the height follows the shape the page reported.
_LOGIN_DESKTOP_MIN_WIDTH = 500
_LOGIN_DESKTOP_MIN_HEIGHT = 360
# The ceiling on what a stranger with a sign-in link can make this box allocate and encode. It
# bounds the framebuffer, so the pixel SCALE below spends the same budget the viewport does.
_LOGIN_DESKTOP_MAX_PIXELS = 1920 * 1200
# Device pixels per CSS pixel. A phone reports 2 or 3; the desktop is built at 2 or not at all,
# because 3 buys little over 2 and costs 2.25 times the pixels.
_LOGIN_DESKTOP_MAX_SCALE = 2
# A viewport is somewhere between 1:3 and 3:1. This bound is what makes the proportional growth
# in `clamp_desktop_size` safe: a reported 20000x1 would otherwise grow to a screen 200 000
# pixels wide while its AREA stayed politely under the ceiling.
_LOGIN_DESKTOP_MAX_ASPECT = 3


@dataclass(frozen=True)
class DesktopSize:
    """A login desktop's viewport in CSS pixels, and how many device pixels each one is.

    The two units are not interchangeable and mixing them is a measured trap: Chrome reads
    `--window-size` in CSS pixels while Xvfb's screen is device pixels, so passing one number to
    both lays the page out at twice the width and clips half of it off the display (measured
    2026-09-09, docs/plans/2026-09-09-hosted-signin-from-a-phone.md §3).
    """

    width: int
    height: int
    scale: int = 1

    @property
    def pixels(self) -> tuple[int, int]:
        """The framebuffer: what Xvfb allocates and what noVNC receives."""
        return self.width * self.scale, self.height * self.scale


def clamp_desktop_size(width: object, height: object, scale: object = 1) -> DesktopSize:
    """The one place a client-reported desktop size becomes a trusted, buildable desktop.

    It arrives from a public page, so it is bounded once here and trusted afterwards. Every step
    that resizes preserves the reported SHAPE, because noVNC contain-fits the framebuffer into
    `#screen` and any ratio the page did not ask for comes back to the user as black bands.
    """
    try:
        wanted_width, wanted_height, wanted_scale = int(width), int(height), int(scale)
    except (TypeError, ValueError):
        return DesktopSize(_LOGIN_DESKTOP_WIDTH, _LOGIN_DESKTOP_HEIGHT)
    if wanted_width < 1 or wanted_height < 1:
        return DesktopSize(_LOGIN_DESKTOP_WIDTH, _LOGIN_DESKTOP_HEIGHT)
    wanted_height = min(max(wanted_height, wanted_width // _LOGIN_DESKTOP_MAX_ASPECT),
                        wanted_width * _LOGIN_DESKTOP_MAX_ASPECT)
    grow = max(1.0,
               _LOGIN_DESKTOP_MIN_WIDTH / wanted_width,
               _LOGIN_DESKTOP_MIN_HEIGHT / wanted_height)
    viewport_width = max(1, round(wanted_width * grow))
    viewport_height = max(1, round(wanted_height * grow))
    # Rounded first, then shrunk by truncation, so the ceiling is one the result never crosses.
    fit = min(1.0, (_LOGIN_DESKTOP_MAX_PIXELS
                    / (viewport_width * viewport_height)) ** 0.5)
    viewport_width = max(1, int(viewport_width * fit))
    viewport_height = max(1, int(viewport_height * fit))
    density = min(max(wanted_scale, 1), _LOGIN_DESKTOP_MAX_SCALE)
    while (density > 1
           and viewport_width * viewport_height * density * density > _LOGIN_DESKTOP_MAX_PIXELS):
        density -= 1
    return DesktopSize(viewport_width, viewport_height, density)
_DESKTOP_START_SECONDS = 15.0
# How long the shared request Chrome outlives its last request. Well under the gateway's 900s
# child idle, so the browser is always reaped before the process that owns it.
_CHROME_IDLE_SECONDS = 120.0
# A request holds the profile for seconds; a login start waits that long rather than failing a
# user who pressed the button while a rail was mid-pull. This is contention on an external
# browser process, not a guard around an in-process call.
_PROFILE_WAIT_SECONDS = 15.0
# How long a login desktop watches its Chrome before calling itself open. Six times the
# measured failure, and invisible next to the 2-10 s Chrome needs to paint the site anyway.
_CHROME_SETTLE_SECONDS = 1.0
# How long a Chrome orphaned by a dead child gets to exit on SIGTERM before it is killed. It is
# holding the only profile this home has, so waiting longer than a browser needs to close its
# tabs just extends an outage.
_STRAY_CHROME_EXIT_SECONDS = 5.0
# How often a waiter retries the cross-process half of the profile lock. Chrome needs seconds to
# launch, so a quarter-second poll costs nothing and keeps the handoff prompt.
_PROFILE_POLL_SECONDS = 0.25
# The lease file lives in the HOME, beside the profile rather than inside it, because the profile
# is the thing that gets burned and re-made. A lease held on a path that is renamed out from under
# it stops excluding anybody, and stops silently.
_PROFILE_LOCK_FILE = "chrome-profile.lock"


class _ProfileLock:
    """One Chrome at a time on one `--user-data-dir`, across every process sharing this home.

    Chrome refuses a second process on the same profile, and an interactive login and a
    background pull would otherwise both want it. One lock for every site, because there is one
    profile for every site.

    TWO LAYERS, AND THE SECOND ONE IS THE POINT. A home is driven by more than one process: the
    resident MCP child runs collectors in-process while the rail worker launches `--once` rail
    children for the same home. `rail_jobs.claim_next` excludes rails from each other and knows
    nothing about the child, so a thread lock excludes nothing that matters. Measured on the box
    2026-09-15: connect starts BOTH at once by design (`onboard_tools` queues the backlog rail
    and spawns the in-process curation walk in one call), and at 20:58:33 a `bookmark_catchup`
    rail landed three seconds into the child's walk, `_claim_profile` read the child's healthy
    Chrome as an orphan and SIGTERMed it, and both sides died — the child's next collector with
    "browser exited (code 0) before opening DevTools" and the rail with "DevTools closed the
    connection during the handshake", taking the user's whole bookmark import with it until the
    next hourly pass. The same collision repeated at 00:34:12 against `substack_saved_catchup`,
    whose next pass is SIX hours out. Both times it was the first minute after a sign-in.

    THE KERNEL DROPS IT WHEN THE HOLDER DIES, and that is what makes `_claim_profile` honest: a
    Chrome found on the profile while this lease is held has no live owner, so ending it is
    orphan recovery rather than murder. The lease is therefore taken BEFORE the claim, never
    after.

    Degrades to the thread lock alone when the lease file cannot be opened, and on a platform
    with no `fcntl`. Both fail in the direction the module already fails: a local install never
    reaches this code, and a hosted home that cannot write into itself has a larger problem than
    contention.
    """

    def __init__(self) -> None:
        self._local = threading.Lock()
        self._fd: int | None = None

    def acquire(self, timeout: float = -1) -> bool:
        # `threading.Lock`'s own signature, negative-means-forever included, because this stands
        # in for one at three call sites and in every test that fakes it.
        deadline = None if timeout < 0 else time.monotonic() + timeout
        if not self._local.acquire(timeout=timeout):
            return False
        if fcntl is None:
            return True
        lease = opyt_path(_PROFILE_LOCK_FILE)
        try:
            # The home, not just the file. A missing directory is the one plausible way this
            # degrades on a real box (a first call that beats the home's creation), and a
            # degrade here is exactly the silent no-exclusion this class exists to end.
            lease.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lease, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as e:
            log(f"[hosted-browser] no cross-process profile lease ({type(e).__name__}); "
                "this process is excluding only itself")
            return True
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return True
            except OSError:
                if deadline is not None and time.monotonic() >= deadline:
                    os.close(fd)
                    self._local.release()
                    return False
                time.sleep(_PROFILE_POLL_SECONDS)

    def release(self) -> None:
        # The descriptor first and the thread lock last, so no waiter is ever woken into a
        # profile this process still holds a lease on.
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)
        self._local.release()

    def locked(self) -> bool:
        return self._local.locked()


_profile_lock = _ProfileLock()


class HostedBrowserError(RuntimeError):
    """The hosted browser boundary could not prepare or use its Chrome profile.

    Every message raised in this module is an authored literal, never an interpolated profile
    path, URL, cookie or environment value. The child's login-start route logs the message
    verbatim, so that rule is what makes the log safe to keep.
    """


class ChromeRequestStatus(str, Enum):
    OK = "ok"
    UNAUTHENTICATED = "unauthenticated"
    RATE_LIMITED = "rate_limited"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RateLimit:
    """The only response headers that leave Chrome's request context."""

    remaining: int
    reset_at: float


@dataclass(frozen=True)
class ChromeRequestResult:
    """Sanitized result of one Chrome-mediated site request.

    ``data`` is JSON response data on success. No response headers other than ``RateLimit`` and
    no request headers ever cross this boundary.
    """

    status: ChromeRequestStatus
    data: dict | None = None
    rate_limit: RateLimit | None = None


def enabled() -> bool:
    """Whether this process is a hosted child (the gateway sets this, local never does).

    The variable is still spelled `OPYT_HOSTED_X` because it predates the second site. Renaming
    it would couple this change to reinstalling the worker unit on a running box, for no change
    in behavior; `tests/gateway/test_worker_jobs.py` pins the current name in both places.
    """
    return os.environ.get("OPYT_HOSTED_X") == "1"


def profile_dir() -> Path:
    """Chrome's sole hosted credential container, scoped by the process's OPYT home.

    ONE profile for every site, not one per site. Chrome refuses two processes on a
    `--user-data-dir`, so a second profile would only buy parallelism by running a second
    Chrome — 200-400 MB more on a box sized for ~100 MB children (`gateway/DEPLOY.md` §1).
    A sign-in therefore blocks the other site's requests for as long as the desktop is open,
    which is the same contention an X sign-in already had with X's own rails.
    """
    return opyt_path("chrome-profile")


def _chrome_binary() -> Path:
    configured = os.environ.get("OPYT_HOSTED_CHROME")
    candidates = [configured] if configured else [
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise HostedBrowserError(
        "Hosted Opyt needs Google Chrome or Chromium on the server. Install it, or set "
        "OPYT_HOSTED_CHROME to its executable path.")


def _desktop_binary(name: str) -> str:
    """One required program in the hosted login desktop, named by its Ubuntu package."""
    path = shutil.which(name)
    if path:
        return path
    raise HostedBrowserError(
        f"A hosted sign-in needs {name}. Install the server display stack before starting the "
        "gateway.")


def _reap_process(proc: subprocess.Popen) -> None:
    """End and collect one desktop process this login owns, escalating only when needed."""
    if proc.poll() is not None:
        with contextlib.suppress(OSError):
            proc.wait()
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        # A kill ends a process but does not collect it. The session that started it is the
        # lifecycle owner, so collection cannot be left to its parent or a later gateway restart.
        with contextlib.suppress(OSError):
            proc.wait()


def _start_xvfb(width: int, height: int) -> tuple[str, subprocess.Popen]:
    """Start one private X display and read its collision-free number from Xvfb itself."""
    read_fd, write_fd = os.pipe()
    try:
        try:
            proc = subprocess.Popen(
                [_desktop_binary("Xvfb"), "-displayfd", str(write_fd), "-screen", "0",
                 f"{width}x{height}x24", "-nolisten", "tcp"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
    except Exception:
        os.close(read_fd)
        raise
    try:
        ready, _, _ = select.select([read_fd], [], [], _DESKTOP_START_SECONDS)
        if not ready:
            raise HostedBrowserError("Xvfb did not allocate a display before the login timed out")
        number = os.read(read_fd, 32).decode().strip()
    except Exception:
        # The caller cannot own this Popen until this function returns its display number, so
        # every failed startup path has to collect it here.
        _reap_process(proc)
        raise
    finally:
        os.close(read_fd)
    if proc.poll() is not None or not number.isdecimal():
        _reap_process(proc)
        raise HostedBrowserError("Xvfb exited before it allocated a usable display")
    return f":{number}", proc


def _profile_holder_pid() -> int | None:
    """The PID of a Chrome already holding this profile, or None.

    Chrome writes `<hostname>-<pid>` into `SingletonLock` and refuses to start a second process
    while that PID is alive. That symlink is the ONLY record of the holder that outlives the
    process which started it, and outliving it is exactly the case here: this child's
    `_profile_lock` and `_shared_chrome` are module globals, so a child that dies without
    reaping its browser leaves a Chrome no later child has a handle to. Measured 2026-09-08 on
    the box — an orphan held a home's profile for 3h14m and every sign-in on that home failed
    with "Chrome could not open the sign-in desktop" until it was killed by hand.

    The cmdline is checked, not just the symlink: PIDs are reused, and this answer gets a
    process killed.
    """
    try:
        target = os.readlink(profile_dir() / "SingletonLock")
    except OSError:
        return None
    _, _, pid_text = target.rpartition("-")
    if not pid_text.isdigit():
        return None
    command = subprocess.run(["ps", "-ww", "-o", "command=", "-p", pid_text],
                             capture_output=True, text=True).stdout
    if "chrome" not in command.lower():
        return None
    if f"--user-data-dir={profile_dir()}" not in command:
        return None
    return int(pid_text)


def _persist_session_cookies() -> None:
    """Make Chrome keep session cookies for this profile instead of dropping them on exit.

    ⚠️ This is what makes a hosted sign-in stick, and it is invisible in every log if removed.
    A site's auth cookie is session-scoped, and Chrome discards session cookies at shutdown
    unless the profile is set to resume its session — so both sign-in paths were creating a
    real session and then throwing it away, because each reads the profile only AFTER the
    browser that signed in is gone.

    Measured on the box 2026-09-12: Substack ACCEPTED a pasted link — the `/sign-in` hop it
    redirected through carried no `error=` (a refusal carries
    `error=Login+link+expired`, measured against a bogus token in the same run) and then
    server-redirected to `/`, which only an authenticated browser is given — and the profile
    still held no session cookie afterwards. 25 cookies, every one of them persistent, and
    never an auth cookie for any site.

    Fail-safe: a profile Chrome has never opened has no `Preferences`, and Chrome writes one
    itself on first run, so there is nothing to do and nothing to crash over.
    """
    prefs_path = profile_dir() / "Default" / "Preferences"
    try:
        prefs = json.loads(prefs_path.read_text())
    except (OSError, ValueError):
        prefs = {}
    session = prefs.setdefault("session", {})
    if session.get("restore_on_startup") == 1 and session.get("startup_urls") == []:
        return
    # 1 is "continue where you left off", which is the setting Chromium builds its cookie store
    # around: session cookies get written to the store instead of being held in memory only.
    # The empty URL list keeps that from also reopening whatever tabs were last open.
    session["restore_on_startup"] = 1
    session["startup_urls"] = []
    try:
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        prefs_path.write_text(json.dumps(prefs))
    except OSError:
        # A sign-in that cannot persist is still worth attempting; it just will not stick, and
        # the operator line is the only place that says why.
        log("[hosted-browser] could not make this profile keep session cookies")


def _cookie_store_signature() -> tuple:
    """A fingerprint of WHEN Chrome last wrote the profile's cookie store, not what is in it.

    The store is never opened. This is `stat` on the file and its SQLite sidecars, because the
    only question being asked is whether a commit has happened — the module docstring's rule
    that no cookie database is opened and no credential reaches Python is untouched by it.

    Both locations are checked because Chromium moved the store under `Network/` and older
    profiles keep the flat path; the box's Chrome writes the flat one (measured 2026-09-15).
    The sidecars matter as much as the file: in journal mode a commit rewrites the journal, and
    a signature taken from the database alone can miss the write it is waiting for.
    """
    parts = []
    for name in ("Cookies", "Network/Cookies"):
        store = profile_dir() / "Default" / name
        for path in (store, *(store.with_name(store.name + s) for s in ("-wal", "-journal"))):
            try:
                stat = path.stat()
            except OSError:
                continue          # fail-safe: an absent file is simply not part of the shape
            parts.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def _claim_profile() -> None:
    """End any Chrome left on this profile by a dead child, so a new one can start.

    ⚠️ CALL THIS ONLY WHILE `_profile_lock` IS HELD. The lease is what proves the holder is
    dead: a live owner would hold it, and the kernel drops it when that owner exits. Called
    without it, this reads a healthy sibling's browser as an orphan and SIGTERMs it — which is
    the 2026-09-15 mutual-eviction incident `_ProfileLock` records.
    """
    pid = _profile_holder_pid()
    if pid is None:
        return
    log("[hosted-browser] a Chrome from a dead child still holds the profile; ending it")
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + _STRAY_CHROME_EXIT_SECONDS
    while time.monotonic() < deadline:
        if _profile_holder_pid() != pid:
            return
        time.sleep(0.2)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


def _require_chrome_survived(proc: subprocess.Popen) -> None:
    """Fail a sign-in whose Chrome died, instead of serving a desktop with nothing drawn on it.

    A single `poll()` here is a race, and it loses about half the time. Chrome refuses a profile
    that another LIVE Chrome holds by handing its URL to that instance and exiting 21 — measured
    on the box 2026-09-08 at 152 ms, while `_wait_for_vnc` returns at about 150 ms. Xvfb and
    x11vnc are healthy in that case and the RFB stream connects normally, so nothing anywhere
    reports a fault; the visitor just watches a black rectangle until the link expires.

    Waiting is the only honest test available. There is no cheap positive signal that Chrome
    opened — the framebuffer is opaque to this process, and polling `xwininfo` for a mapped
    window would put a binary this module does not otherwise need into the critical path.
    """
    deadline = time.monotonic() + _CHROME_SETTLE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HostedBrowserError("Chrome could not open the sign-in desktop")
        time.sleep(0.05)


def _free_loopback_port() -> int:
    """A currently-unused loopback port for this session's VNC listener."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _wait_for_vnc(proc: subprocess.Popen, port: int) -> None:
    """Wait for the owned VNC subprocess to accept locally, or report its failed start."""
    deadline = time.monotonic() + _DESKTOP_START_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HostedBrowserError("x11vnc exited before accepting the login desktop")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise HostedBrowserError("x11vnc did not accept the login desktop before the timeout")


def _rate_limit(raw: object) -> RateLimit | None:
    if not isinstance(raw, dict):
        return None
    try:
        return RateLimit(remaining=int(raw["remaining"]), reset_at=float(raw["reset"]))
    except (KeyError, TypeError, ValueError):
        return None


class ChromeRequestRunner:
    """One owned headless Chrome session that can make only Opyt's authenticated site requests.

    It keeps ONE PAGE PER SITE. A request has to run inside a page the site itself served:
    `credentials: "include"` on a cross-origin fetch is subject to the target's CORS policy, so
    a substack.com request issued from an x.com page is refused by Substack. The page is also
    where a session value may be read — the csrf token X wants is in `document.cookie`, and it
    never leaves the renderer.
    """

    def __init__(self) -> None:
        self._context = None
        self._browser = None
        self._pages: dict[str, cdp.Page] = {}

    def __enter__(self) -> "ChromeRequestRunner":
        self.start()
        return self

    def __exit__(self, *_unused) -> None:
        self.close()

    def start(self) -> None:
        if self._browser is not None:
            raise HostedBrowserError("hosted Chrome is already running")
        _claim_profile()
        _persist_session_cookies()
        self._context = cdp.controlled_browser(
            _chrome_binary(), user_data_dir=profile_dir(), headless=True)
        self._browser = self._context.__enter__()

    def close(self) -> None:
        if self._context is not None:
            context, self._context = self._context, None
            self._browser = None
            self._pages.clear()
            context.__exit__(None, None, None)

    @property
    def running(self) -> bool:
        return self._browser is not None

    def alive(self) -> bool:
        """One loopback round trip that proves this browser still answers.

        Chrome can die inside the idle window — an OOM kill, a crash, an operator. The request
        path swallows CDPError and reports UNAVAILABLE, so nothing else would notice; without
        this probe one dead browser poisons every request until the idle timer fires.
        """
        if self._browser is None:
            return False
        try:
            self._browser.call("Browser.getVersion")
        except (cdp.CDPError, OSError):
            return False
        return True

    def page(self, home_url: str) -> cdp.Page:
        """This runner's page on `home_url`'s site, opened and navigated on first use."""
        page = self._pages.get(home_url)
        if page is not None:
            return page
        if self._browser is None:
            raise HostedBrowserError("hosted Chrome is not running")
        page = cdp.new_page(self._browser, "about:blank")
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Page.navigate", {"url": home_url})
        # Page navigation is an external network operation; this is the bounded readiness wait,
        # not a timeout around an in-process function.
        page.wait_for_event("Page.loadEventFired", timeout=30.0)
        self._pages[home_url] = page
        return page

    def evaluate(self, home_url: str, expression: str) -> dict | None:
        """Run one authored script inside `home_url`'s page and take back only its value."""
        try:
            result = self.page(home_url).call("Runtime.evaluate", {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            })
        except cdp.CDPError:
            return None
        value = (result.get("result") or {}).get("value")
        return value if isinstance(value, dict) else None

    @staticmethod
    def _fetch_script(url: str, headers_js: str) -> str:
        """A request whose session values remain wholly inside the Chrome renderer.

        `headers_js` is a JavaScript object literal authored in this repo, never a value from a
        request or a user. It is source rather than data because X's csrf header can only be
        computed in the page, from `document.cookie`.

        The two rate-limit headers are X's. A site that does not publish them yields nulls and
        `_rate_limit` returns None, which is why one script serves both sites.
        """
        return f"""(async () => {{
            const response = await fetch({json.dumps(url)}, {{
                credentials: "include",
                headers: {headers_js},
            }});
            let data = null;
            try {{ data = await response.json(); }} catch (_ignored) {{}}
            return {{
                status: response.status,
                data,
                rate: {{
                    remaining: response.headers.get("x-rate-limit-remaining"),
                    reset: response.headers.get("x-rate-limit-reset"),
                }},
            }};
        }})()"""

    @staticmethod
    def _result(raw: dict | None) -> ChromeRequestResult:
        if raw is None:
            return ChromeRequestResult(ChromeRequestStatus.UNAVAILABLE)
        status = raw.get("status")
        rate = _rate_limit(raw.get("rate"))
        if isinstance(status, int) and 200 <= status < 300 and isinstance(raw.get("data"), dict):
            return ChromeRequestResult(ChromeRequestStatus.OK, raw["data"], rate)
        if status in (401, 403):
            return ChromeRequestResult(ChromeRequestStatus.UNAUTHENTICATED, rate_limit=rate)
        if status == 429:
            return ChromeRequestResult(ChromeRequestStatus.RATE_LIMITED, rate_limit=rate)
        if status in (400, 404):
            return ChromeRequestResult(ChromeRequestStatus.REJECTED, rate_limit=rate)
        return ChromeRequestResult(ChromeRequestStatus.UNAVAILABLE, rate_limit=rate)

    def fetch_json(self, home_url: str, url: str, headers_js: str = "{}") -> ChromeRequestResult:
        """One authenticated GET, made by Chrome inside `home_url`'s page, sanitized on return."""
        return self._result(self.evaluate(home_url, self._fetch_script(url, headers_js)))


class _SharedChrome:
    """One reused Chrome per hosted child, reaped after `_CHROME_IDLE_SECONDS` of no use.

    Every request used to build its own ``ChromeRequestRunner``. Measured 2026-09-07 on a fast
    Mac: 3.0-6.0 seconds to launch Chrome, navigate and tear it down — **per request**. A
    twenty-request pull spent one to two minutes starting browsers, and a 1-2 vCPU VPS is
    slower.

    Reaped rather than resident because a live Chrome is 200-400 MB against a box sized for
    ~100 MB children (`gateway/DEPLOY.md` §1). A burst of requests shares one browser; an idle
    user's browser goes away. The window sits well under the gateway's 900 s child idle, so
    Chrome is always the first of the two to go.
    """

    def __init__(self) -> None:
        self._runner: ChromeRequestRunner | None = None
        self._timer: threading.Timer | None = None

    def acquire(self) -> ChromeRequestRunner:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        # EVICTED, not merely absent, and the two are worth telling apart in a log. A browser we
        # still hold a reference to but that is no longer alive was ended by a SIBLING: it went
        # idle between our requests, a sibling took the lease, `_claim_profile` correctly read it
        # as ownerless and SIGTERMed it. That is the handoff cost §3 of
        # `docs/plans/2026-09-16-hosted-chrome-contention-handoff.md` is about, and until this
        # line it was invisible — a relaunch and a cold start left identical evidence, which is
        # why nobody could say how many relaunches a real connect pays.
        evicted = self._runner is not None and not self._runner.alive()
        if evicted:
            self.close()
        if self._runner is None:
            started = time.monotonic()
            runner = ChromeRequestRunner()
            runner.start()
            self._runner = runner
            log(f"[hosted-browser] Chrome {'RELAUNCHED after eviction' if evicted else 'started'} "
                f"in {time.monotonic() - started:.1f}s")
        return self._runner

    def release(self) -> None:
        """Start the idle countdown. The browser stays up for the next request in a burst."""
        if self._runner is None:
            return
        self._timer = threading.Timer(_CHROME_IDLE_SECONDS, self.close)
        self._timer.daemon = True
        self._timer.start()

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._runner is not None:
            runner, self._runner = self._runner, None
            with contextlib.suppress(Exception):
                runner.close()


_shared_chrome = _SharedChrome()


@contextlib.contextmanager
def shared_chrome():
    """The one Chrome this child uses for every site's requests, held for one request.

    The lock is the profile, not the object. Chrome refuses two processes on one
    ``--user-data-dir``, and ``cdp._Socket`` serializes nothing, so exactly one caller may be
    driving this profile at a time — an interactive login included.
    """
    waited_from = time.monotonic()
    if not _profile_lock.acquire(timeout=_PROFILE_WAIT_SECONDS):
        log(f"[hosted-browser] gave up waiting for this home's profile after "
            f"{time.monotonic() - waited_from:.1f}s")
        raise HostedBrowserError("this home's Chrome is busy with a sign-in or another pass; "
                                 "try again after it finishes")
    # Only a wait long enough to have POLLED is a real one. An uncontended acquire returns in
    # microseconds and happens once per request, so logging those would bury the signal under a
    # burst's own noise. A logged wait means a sibling process actually held the profile, and the
    # duration is one half of §3's missing number — the other half is the relaunch line above.
    waited = time.monotonic() - waited_from
    if waited >= _PROFILE_POLL_SECONDS:
        log(f"[hosted-browser] waited {waited:.1f}s for another process to release this "
            f"home's profile")
    try:
        runner = _shared_chrome.acquire()
        try:
            yield runner
        except Exception:
            # A half-broken browser must not be handed to the next caller. Dropping it costs
            # one relaunch; keeping it costs every request after this one.
            _shared_chrome.close()
            raise
        else:
            _shared_chrome.release()
    finally:
        _profile_lock.release()


def site_connected(site: str) -> bool:
    """Whether the profile now holds a live session for `site`, by one authenticated request.

    The dispatch is here rather than in the caller because a login's completion check and a
    phase probe must agree on what "connected" means for a site, and there is exactly one
    answer per site: the request that site's own module makes.
    """
    from pipeline.ingestion import hosted_substack, hosted_x
    validators = {"x": hosted_x.has_connection, "substack": hosted_substack.has_connection}
    validate = validators.get(site)
    if validate is None:
        raise HostedBrowserError("that site has no hosted sign-in")
    return validate()


class _DesktopLogin:
    """One user-driven Chrome desktop, its private X display, and its VNC endpoint.

    No debugger attaches to this Chrome. The VNC client supplies real X11 input straight to
    the display, which is the property Google requires at the sign-in boundary. The child owns
    every subprocess because the persistent profile is shared with its post-login CDP reader.
    """

    def __init__(self, site: str, login_url: str, expires_at: float,
                 size: DesktopSize) -> None:
        self.site = site
        self.login_url = login_url
        self.expires_at = expires_at
        self.size = size
        self._lock = threading.RLock()
        self._closed = False
        self._timer: threading.Timer | None = None
        self._holds_profile = False
        # Named rather than a list, because the browser alone is replaceable: `restart_browser`
        # below reaps and relaunches it while the display and the VNC server, and so the RFB
        # connection the user is watching through, stay up.
        self._xvfb: subprocess.Popen | None = None
        self._chrome: subprocess.Popen | None = None
        self._vnc: subprocess.Popen | None = None
        self._display: str | None = None
        self._vnc_port: int | None = None

    @property
    def vnc_port(self) -> int:
        with self._lock:
            if self._closed or self._vnc_port is None:
                raise HostedBrowserError("that hosted sign-in is not running")
            return self._vnc_port

    def _launch_chrome(self) -> subprocess.Popen:
        """This desktop's browser, on this desktop's display, at this desktop's size."""
        return subprocess.Popen(
            cdp.chrome_argv(_chrome_binary(), profile_dir())
            # CSS pixels here and device pixels on Xvfb, which is why the scale is passed
            # rather than folded into one number. A phone gets 500 CSS px of sign-in form
            # drawn at 1000 device pixels; a laptop gets its own box at 1.
            + [f"--window-size={self.size.width},{self.size.height}",
               f"--force-device-scale-factor={self.size.scale}",
               "--window-position=0,0", self.login_url],
            env=dict(os.environ, DISPLAY=self._display),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def restart_browser(self) -> None:
        """Put the browser back on the sign-in page, from any state the user reached.

        This exists because a hosted desktop has no way out of a Chrome popup window, and a
        sign-in page is full of buttons that open one. Measured on the box 2026-09-09: X's
        "Continue with Google" opens a SECOND Chrome window that covers the screen and carries
        no tab strip, no back button and no close button, and Xvfb runs no window manager, so
        it has no title bar either. Alt+Left in it does nothing (it has no history to go back
        to); only closing the window recovers, and the page cannot be the one to ask for that,
        because Ctrl+W on the LAST window exits Chrome and takes the desktop with it.

        So the reset happens here, where the processes are owned and the difference between
        "close a popup" and "kill the sign-in" is knowable. The display and the VNC server are
        untouched, so the user keeps watching the same screen, and the profile is on disk, so
        whatever they had already signed into survives.
        """
        with self._lock:
            if self._closed or self._chrome is None:
                raise HostedBrowserError("that hosted sign-in is not running")
            _reap_process(self._chrome)
            self._chrome = self._launch_chrome()
            _require_chrome_survived(self._chrome)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise HostedBrowserError("that hosted sign-in has expired")
            # The interactive desktop and the CDP request runner share Chrome's one persistent
            # profile. Exclusion is structural: Chrome itself refuses two processes on it.
            #
            # Timed for the same reason `shared_chrome()` times its wait, and this is the site
            # where the number is worth the most: a sign-in is the one contender a HUMAN is
            # waiting on, and connect starts a rail against the same profile in the same second
            # (§6 of the contention handoff). A user who waited 9s to see a login window and a
            # user who saw one instantly leave identical logs without this.
            waited_from = time.monotonic()
            if not _profile_lock.acquire(timeout=_PROFILE_WAIT_SECONDS):
                log(f"[hosted-login] the profile was still busy after "
                    f"{time.monotonic() - waited_from:.1f}s; the sign-in did not start")
                raise HostedBrowserError("hosted Chrome is busy; try the sign-in link again")
            waited = time.monotonic() - waited_from
            if waited >= _PROFILE_POLL_SECONDS:
                log(f"[hosted-login] waited {waited:.1f}s for a pass to release this home's "
                    f"profile before starting the sign-in")
            self._holds_profile = True
            _shared_chrome.close()
            _claim_profile()
            _persist_session_cookies()
            try:
                display, xvfb = _start_xvfb(*self.size.pixels)
                self._display = display
                self._xvfb = xvfb
                self._chrome = self._launch_chrome()
                port = _free_loopback_port()
                # `-threads` gives each client its own input and output thread instead of one
                # loop serving everything. Measured on the box 2026-09-09 with
                # `scripts/rfb_latency.py`, over 100 keystrokes: input-to-frame p50 fell from
                # 30.5 ms to 11.4 ms and p95 from 51.3 ms to 14.3 ms. `-wait 10` and `-defer 10`
                # were measured in the same run and changed nothing, alone or added to this, so
                # they are deliberately absent -- two magic numbers for no gain.
                vnc = subprocess.Popen(
                    [_desktop_binary("x11vnc"), "-display", display, "-localhost", "-nopw",
                     "-forever", "-shared", "-threads", "-rfbport", str(port)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._vnc = vnc
                _wait_for_vnc(vnc, port)
                _require_chrome_survived(self._chrome)
                self._vnc_port = port
                self._timer = threading.Timer(
                    max(0, self.expires_at - time.monotonic()), self.close)
                self._timer.daemon = True
                self._timer.start()
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._vnc_port = None
            # Stop VNC before the browser and display: clients lose their endpoint before any
            # process beneath it starts to close, and every Popen is collected by its owner.
            for proc in (self._vnc, self._chrome, self._xvfb):
                if proc is not None:
                    _reap_process(proc)
            self._vnc = self._chrome = self._xvfb = None
            self._display = None
            if self._holds_profile:
                self._holds_profile = False
                _profile_lock.release()

    def _await_durable_session(self) -> None:
        """Hold the desktop open until Chrome has written this sign-in down.

        See the settle constants for the measurement: a browser killed before Chromium's commit
        timer fires takes the session with it, and no signal makes it flush on the way out. The
        wait therefore watches the store rather than the clock — the first write after the
        lead-in carries every cookie pending at that moment, the new session included.

        Reaching the deadline is not an error to raise. It means nothing was pending, which is
        what a desktop closed without signing in looks like, and the validation below is the
        thing entitled to say so.
        """
        deadline = time.monotonic() + _LOGIN_SETTLE_SECONDS
        # The watch starts after a lead-in, not at the click: a user can press "Done signing in"
        # while the site's own sign-in response is still in flight, and a write observed before
        # that response lands would close the browser on a session it had not yet created.
        time.sleep(min(_LOGIN_LEAD_SECONDS, _LOGIN_SETTLE_SECONDS))
        written = _cookie_store_signature()
        while time.monotonic() < deadline:
            if self._closed or _cookie_store_signature() != written:
                return
            time.sleep(_LOGIN_SETTLE_POLL_SECONDS)
        log("[hosted-login] the profile's cookie store was never written while waiting; "
            "closing anyway, and this sign-in will read as not connected")

    def complete(self) -> str:
        """Let the sign-in become durable, close the desktop, then validate the profile."""
        with self._lock:
            expired = self._closed or time.monotonic() >= self.expires_at
        # Judged BEFORE the wait, and deliberately not re-judged after it: the user finished
        # inside their window, and a desktop that ran out of time while Opyt was waiting on
        # Chrome must not be charged to them.
        if not expired:
            self._await_durable_session()
        self.close()
        if expired:
            return "expired"
        try:
            return "connected" if site_connected(self.site) else "not_connected"
        except HostedBrowserError:
            return "not_connected"


@dataclass(frozen=True)
class _PendingLogin:
    """A minted sign-in capability: which site it opens, and when it was minted.

    `start` honours the mint time for `_LOGIN_LINK_TTL_SECONDS`; the running desktop then gets
    a deadline of its own (see the two-deadline comment at the top of this file).
    """

    site: str
    login_url: str
    created_at: float


class HostedLoginManager:
    """The hosted child's in-memory desktop state; no profile or credential leaves it."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingLogin] = {}
        self._active: _DesktopLogin | None = None
        self._active_nonce: str | None = None

    def create(self, site: str, login_url: str) -> str:
        # One desktop at a time, whatever the site: there is one profile, and Chrome refuses a
        # second process on it. A new request therefore supersedes any pending or open sign-in.
        if self._active is not None:
            self._active.close()
            self._active = None
        self._active_nonce = None
        self._pending.clear()
        nonce = secrets.token_urlsafe(32)
        self._pending[nonce] = _PendingLogin(
            site=site, login_url=login_url, created_at=time.monotonic())
        return nonce

    def start(self, nonce: str, size: DesktopSize | None = None) -> None:
        pending = self._pending.pop(nonce, None)
        if pending is None or time.monotonic() >= pending.created_at + _LOGIN_LINK_TTL_SECONDS:
            raise HostedBrowserError("that hosted sign-in link has expired")
        # The desktop's clock starts HERE, when the user opens it, not when the link was minted.
        login = _DesktopLogin(pending.site, pending.login_url,
                              time.monotonic() + _LOGIN_DESKTOP_TTL_SECONDS,
                              size or DesktopSize(_LOGIN_DESKTOP_WIDTH,
                                                  _LOGIN_DESKTOP_HEIGHT))
        login.start()
        self._active = login
        self._active_nonce = nonce

    def current(self, nonce: str) -> _DesktopLogin | None:
        return self._active if self._active_nonce == nonce else None

    def restart(self, nonce: str) -> None:
        login = self.current(nonce)
        if login is None:
            raise HostedBrowserError("that hosted sign-in is not running")
        login.restart_browser()

    def complete(self, nonce: str) -> str:
        login = self.current(nonce)
        if login is None:
            return "expired"
        result = login.complete()
        if self._active is login:
            self._active = None
            self._active_nonce = None
        return result


_manager = HostedLoginManager()


def begin_login(site: str, login_url: str) -> str:
    """Mint and register the one-time public capability; Chrome starts only on redemption."""
    register_url = os.environ.get("OPYT_HOSTED_INTERACTION_REGISTER_URL")
    register_key = os.environ.get("OPYT_HOSTED_INTERACTION_KEY")
    public_base = os.environ.get("OPYT_HOSTED_INTERACTION_URL")
    if not register_url or not register_key or not public_base:
        raise HostedBrowserError("a hosted sign-in is not configured by the gateway")
    nonce = _manager.create(site, login_url)
    request = urllib.request.Request(
        register_url,
        data=json.dumps({"kind": site, "nonce": nonce}).encode(),
        headers={"Content-Type": "application/json",
                 "X-Opyt-Hosted-Interaction-Key": register_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
    except (OSError, urllib.error.URLError):
        _manager._pending.pop(nonce, None)
        raise HostedBrowserError("could not create the hosted sign-in link") from None
    return f"{public_base.rstrip('/')}/login/{site}/{nonce}"


def hosted_child_app(mcp_app):
    """Wrap FastMCP's app with loopback-only hosted-login endpoints for the gateway to relay.

    The nonce alone addresses a sign-in — the child remembers which site that nonce opens, so
    no route below repeats it.
    """
    from contextlib import asynccontextmanager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route, WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect

    key = os.environ.get("OPYT_HOSTED_INTERACTION_KEY", "")

    def authorized(value: str | None) -> bool:
        return bool(key and value and secrets.compare_digest(key, value))

    async def start(request: Request):
        if not authorized(request.headers.get("X-Opyt-Hosted-Interaction-Key")):
            return JSONResponse({"error": "not_found"}, status_code=404)
        # The gateway has already clamped this. Absent body means the caller had no size to
        # give, and the fallback constants apply.
        size = None
        with contextlib.suppress(Exception):
            body = await request.json()
            size = clamp_desktop_size(body.get("width"), body.get("height"),
                                      body.get("scale", 1))
        try:
            await asyncio.to_thread(_manager.start, request.path_params["nonce"], size)
        except HostedBrowserError as failure:
            # The gateway is told only that the start failed, so this child log is the ONLY
            # record of which step refused: an expired link, a busy profile, a missing display
            # binary, Xvfb, x11vnc, or a Chrome that exited. Debugging a hosted sign-in from
            # the gateway's generic response alone is guesswork; that cost was measured on
            # 2026-09-07 against a failure that left no trace in either log.
            log(f"[hosted-login] desktop did not start: {failure}")
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse({"status": "started"})

    async def restart(request: Request):
        """Send this desktop's browser back to the sign-in page."""
        if not authorized(request.headers.get("X-Opyt-Hosted-Interaction-Key")):
            return JSONResponse({"error": "not_found"}, status_code=404)
        try:
            await asyncio.to_thread(_manager.restart, request.path_params["nonce"])
        except HostedBrowserError as failure:
            log(f"[hosted-login] the browser did not restart: {failure}")
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse({"status": "restarted"})

    async def stream(websocket: WebSocket) -> None:
        """Relay raw RFB bytes between the gateway and this login's loopback VNC server."""
        if not authorized(websocket.headers.get("X-Opyt-Hosted-Interaction-Key")):
            await websocket.close(code=4404)
            return
        login = _manager.current(websocket.path_params["nonce"])
        if login is None:
            await websocket.close(code=4404)
            return
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", login.vnc_port)
        except (HostedBrowserError, OSError):
            await websocket.close(code=1011)
            return
        await websocket.accept()

        async def browser_to_vnc() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = message.get("bytes")
                if payload is None:
                    await websocket.close(code=1003)
                    return
                writer.write(payload)
                await writer.drain()

        async def vnc_to_browser() -> None:
            while payload := await reader.read(65536):
                await websocket.send_bytes(payload)

        tasks = {asyncio.create_task(browser_to_vnc()), asyncio.create_task(vnc_to_browser())}
        try:
            _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def complete(request: Request):
        if not authorized(request.headers.get("X-Opyt-Hosted-Interaction-Key")):
            return JSONResponse({"error": "not_found"}, status_code=404)
        status = await asyncio.to_thread(_manager.complete, request.path_params["nonce"])
        return JSONResponse({"status": status})

    @asynccontextmanager
    async def lifespan(app):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
        active = _manager.current(_manager._active_nonce or "")
        if active is not None:
            active.close()
        # A graceful exit reaps its own browser. It is not enough on its own — a SIGKILL or an
        # OOM kill runs nothing — which is why `_claim_profile` exists on the start path.
        _shared_chrome.close()

    return Starlette(routes=[
        Route("/_hosted-login/{nonce}/start", start, methods=["POST"]),
        Route("/_hosted-login/{nonce}/restart", restart, methods=["POST"]),
        Route("/_hosted-login/{nonce}/complete", complete, methods=["POST"]),
        WebSocketRoute("/_hosted-login/{nonce}/stream", stream),
        Mount("/", app=mcp_app),
    ], lifespan=lifespan)
