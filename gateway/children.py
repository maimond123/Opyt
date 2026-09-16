"""
gateway/children.py — the pool of per-user `opyt-mcp` children, and the reaper that ends them.

One child per user, spawned on demand, listening on loopback with that user's `$OPYT_HOME` in
its environment. **This is the whole reason the hosted server needs no tool-code changes:** a
child serves exactly one home, so every module-level global inside it stays per-user, which is
the same property the stdio server gets from being one process per session.

The rejected alternative was one shared multi-tenant process with a request-scoped data home.
It is cheaper in RAM and needs an audit of every module-level cache in the codebase, where a
miss is a silent cross-tenant leak. Flip condition: concurrent users exceed one box's RAM.
Design record: docs/plans/2026-09-02-hosted-opyt-remote-connector.md.

**The boundary this module holds:** the pool owns processes and ports. It never reads a home,
never learns a user's onboarding phase, and never touches a credential. Everything it knows
describes a process it started and can kill, which is why its state is in memory and nowhere
else — a persisted row would outlive the process it describes and become a second source of
truth about what is running.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The subject arrives inside a verified OAuth token and then becomes a DIRECTORY NAME, so this
# is the trust boundary and it is validated here exactly once. No dot is allowed, which is what
# makes traversal ("..", "a/../../etc") unrepresentable rather than merely filtered. Google's
# `sub` is digits; the wider alphabet is for a future upstream provider, not for user input.
_SUBJECT_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Stripped from a child's environment. The gateway's OAuth secret and an operator-provided X
# bearer have no use inside a hosted child; the Chrome profile is the only X credential container.
#
# `OPYT_WORKER_DB` is deliberately NOT here. It is the one non-secret path of the worker's
# control database, and a child inherits it so that a product action can queue durable rail work
# the separate worker will run. Adding it to this tuple would leave every hosted queue request
# writing into a per-home file nothing reads (`rail_jobs.worker_db_path` raises rather than let
# that happen silently).
_GATEWAY_ONLY = (
    "OPYT_GATEWAY_GOOGLE_CLIENT_ID",
    "OPYT_GATEWAY_GOOGLE_CLIENT_SECRET",
    "OPYT_GATEWAY_BASE_URL",
    "OPYT_GATEWAY_INTERNAL_URL",
    "OPYT_HOMES_ROOT",
    "X_WEB_BEARER",
    # The trial's management key is the most dangerous value this box holds: it mints spend
    # against the operator's OpenRouter balance, where every other secret here only proves an
    # identity. A child has no use for it — it asks the gateway to mint and receives one capped
    # key — so it never travels, and `test_a_child_never_inherits_the_management_key` pins that.
    # The ledger path goes too: a child that could write it could grant itself another allowance.
    "OPYT_TRIAL_MANAGEMENT_KEY",
    "OPYT_TRIAL_LEDGER",
)

CHILD_LOG = "mcp_child.log"


class BadSubject(ValueError):
    """The token's subject cannot name a directory. A 403, never a spawn."""


class SpawnFailed(RuntimeError):
    """The child exited, or never listened, before the deadline."""


def home_for(root: Path, subject: str) -> Path:
    """`<root>/<subject>`, after the one validation that makes that concatenation safe."""
    if not _SUBJECT_RE.fullmatch(subject):
        raise BadSubject(f"subject is not a usable directory name: {subject!r}")
    return root / subject


def _free_port() -> int:
    """A port nothing is listening on right now.

    There is a race between this close and the child's bind, and it is accepted rather than
    engineered away: a collision surfaces as `SpawnFailed` on one request, and the next request
    for that subject picks a new port. The alternative — the child choosing and reporting its
    own port — means parsing the child's stdout, which under stdio is the JSON-RPC channel and
    is exactly the coupling this design avoids everywhere else.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _child_env(home: Path, subject: str, *,
               interaction_registration_url: str | None = None,
               interaction_url: str | None = None,
               interaction_key: str | None = None,
               interaction_trial_url: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["OPYT_HOME"] = str(home)
    # The subject, validated once by `home_for`, carried forward as the child's own job key.
    # This is the whole of the gateway's worker integration: it hands down a name it already
    # proved safe, and never reads a job row, schedules a rail, or enrols a home of its own.
    env["OPYT_WORKER_HOME_ID"] = subject
    if interaction_registration_url and interaction_url and interaction_key:
        # The child owns the X profile and OpenRouter PKCE verifier; the gateway only routes
        # the short-lived public interaction.
        env["OPYT_HOSTED_X"] = "1"
        env["OPYT_HOSTED_OPENROUTER"] = "1"
        env["OPYT_HOSTED_INTERACTION_REGISTER_URL"] = interaction_registration_url
        env["OPYT_HOSTED_INTERACTION_URL"] = interaction_url
        env["OPYT_HOSTED_INTERACTION_KEY"] = interaction_key
        if interaction_trial_url:
            # Set ONLY when this gateway can actually mint. The child reads its presence as
            # "a starter allowance is on offer here", so advertising it on a gateway with no
            # management key would put a step in front of the user that can only fail.
            env["OPYT_HOSTED_INTERACTION_TRIAL_URL"] = interaction_trial_url
    for k in _GATEWAY_ONLY:
        env.pop(k, None)
    return env


@dataclass
class Child:
    """One running `opyt-mcp`. `last_seen` is monotonic, so a clock change cannot reap."""
    subject: str
    home: Path
    port: int
    proc: asyncio.subprocess.Process
    last_seen: float
    inflight: int = 0
    interaction_key: str = ""

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None


# The sites that get an interactive sign-in desktop, and every kind a child may register.
# A route segment allow-list: the child names which site a nonce opens, and the gateway checks
# the name against this set before it will route anything to it.
BROWSER_LOGIN_KINDS = frozenset({"x", "substack"})
INTERACTION_KINDS = BROWSER_LOGIN_KINDS | {"openrouter"}

# How many sign-in desktops may be open at once, across every user.
#
# This cap exists to produce a REFUSAL PAGE, not to raise throughput. Without it a full box kills
# a desktop with no message, and the visitor sees onboarding stall with nothing to read and no
# retry instruction. That outcome needs no observed contention to be worth preventing, which is
# why the cap ships ahead of any load that would justify tuning the number. The number is the
# cheap part; having a sentence to show is the point. See ruling R2 below on ordering.
#
# Six is a deliberate launch floor, set below anything measured rather than at a capacity
# estimate. A desktop holds a headed Chrome (measured 200-400 MB), an Xvfb and an x11vnc for up
# to the ten-minute login TTL, so six costs at most ~3G of the gateway cgroup's 20G ceiling and
# leaves the rest for MCP children at 82.7-99.3 MB each. `OPYT_GATEWAY_MAX_SIGNINS` overrides it;
# every refusal is logged, and that log line is the only evidence that would justify raising it.
DEFAULT_MAX_SIGNINS = 6


@dataclass(frozen=True)
class InteractionNonce:
    """A one-time public interaction route, never a profile or credential record."""

    child: Child
    nonce: str
    kind: str
    expires_at: float


@dataclass(frozen=True)
class LoginSession:
    """The post-redemption browser channel, still only a route to one live child."""

    child: Child
    nonce: str
    expires_at: float


@dataclass(frozen=True)
class LoginCompletion:
    """The one public action allowed to end and validate an attached sign-in desktop."""

    child: Child
    nonce: str
    expires_at: float


class ChildPool:
    """Subject → running child, plus the reaper.

    Not safe across processes, and it must never be run in more than one: two gateway workers
    would keep two tables and spawn two children per user, each writing the same home. Run the
    gateway single-process — it is pure I/O proxying, so there is nothing to parallelize.
    """

    def __init__(self, homes_root: Path, *, idle_seconds: float = 900.0,
                 spawn_timeout: float = 30.0, reap_period: float = 60.0,
                 interaction_registration_url: str | None = None,
                 interaction_url: str | None = None,
                 interaction_trial_url: str | None = None,
                 login_ttl_seconds: float = 600.0,
                 max_signins: int = DEFAULT_MAX_SIGNINS) -> None:
        self.homes_root = Path(homes_root)
        self.idle_seconds = idle_seconds
        self.spawn_timeout = spawn_timeout
        self.reap_period = reap_period
        self.interaction_registration_url = interaction_registration_url
        self.interaction_url = interaction_url
        self.interaction_trial_url = interaction_trial_url
        self.login_ttl_seconds = login_ttl_seconds
        self.max_signins = max_signins
        self._children: dict[str, Child] = {}
        self._interaction_nonces: dict[str, InteractionNonce] = {}
        self._login_sessions: dict[str, LoginSession] = {}
        self._login_completions: dict[str, LoginCompletion] = {}
        # One lock per subject, so a burst from ONE user spawns once while other users are
        # unaffected. An MCP client opens with `initialize`, `notifications/initialized` and
        # `tools/list` within milliseconds; without this, first contact spawns three children.
        # The dict is never pruned: it holds one small lock per user who has ever connected to
        # this process, and pruning it would race with a waiter that has not yet acquired.
        self._locks: dict[str, asyncio.Lock] = {}
        self._watchers: set[asyncio.Task] = set()
        self._reaper_task: asyncio.Task | None = None

    # ── The request path ────────────────────────────────────────────────────────────────────

    async def acquire(self, subject: str) -> Child:
        """The child for `subject`, spawning one if needed, with `inflight` already counted.

        Every caller MUST `release()` in a finally, including when the response is a stream —
        an in-flight request is what stops the reaper mid-call.
        """
        home = home_for(self.homes_root, subject)      # validates before anything is created
        lock = self._locks.setdefault(subject, asyncio.Lock())
        async with lock:
            child = self._children.get(subject)
            if child is not None and not child.alive:
                # Crashed between requests. Drop it here as well as in the watcher, because a
                # watcher that has not been scheduled yet would otherwise hand out a dead port.
                self._children.pop(subject, None)
                child = None
            if child is None:
                child = await self._spawn(subject, home)
                self._children[subject] = child
                self._watch(child)
            child.last_seen = time.monotonic()
            child.inflight += 1
            return child

    def release(self, child: Child) -> None:
        child.inflight = max(0, child.inflight - 1)
        child.last_seen = time.monotonic()

    # ── Lifecycle ───────────────────────────────────────────────────────────────────────────

    async def _spawn(self, subject: str, home: Path) -> Child:
        home.mkdir(parents=True, exist_ok=True)
        log_path = home / CHILD_LOG
        port = _free_port()
        interaction_key = (secrets.token_urlsafe(32)
                           if self.interaction_registration_url else "")
        logf = open(log_path, "a")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mcp_server.server", "--http", str(port),
                cwd=str(REPO_ROOT),
                env=_child_env(home, subject,
                               interaction_registration_url=self.interaction_registration_url,
                               interaction_url=self.interaction_url,
                               interaction_key=interaction_key,
                               interaction_trial_url=self.interaction_trial_url),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=logf, stderr=logf,
                # Its own session, so `_terminate` has a process GROUP to signal. A child
                # spawns Chrome for hosted browser work, and that Chrome is the thing a
                # per-child kill must reach. This does not weaken the unit's
                # `KillMode=control-group`: a new session stays in the same cgroup, so a
                # gateway restart still takes every child with it.
                start_new_session=True,
            )
        finally:
            # Only after the spawn: the child has dup'd the fd by now, and closing it earlier
            # would hand the child a closed descriptor. Same ordering as
            # `pipeline/kb/rail_worker.RailWorker._launch`, for the same reason.
            logf.close()

        deadline = time.monotonic() + self.spawn_timeout
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise SpawnFailed(
                    f"child for {subject} exited {proc.returncode} before listening; "
                    f"see {log_path}")
            if await _listening(port):
                return Child(subject=subject, home=home, port=port, proc=proc,
                             last_seen=time.monotonic(), interaction_key=interaction_key)
            await asyncio.sleep(0.05)

        await _terminate(proc)
        raise SpawnFailed(
            f"child for {subject} did not listen on {port} within {self.spawn_timeout}s; "
            f"see {log_path}")

    def _watch(self, child: Child) -> None:
        """Await the child's exit, which both collects its status and drops its route.

        Without the await the exited child stays a zombie holding a process-table slot. That is
        the literal meaning of reaping, and a gateway that spawns without it exhausts the pid
        table.
        """
        async def watcher() -> None:
            await child.proc.wait()
            if self._children.get(child.subject) is child:
                del self._children[child.subject]

        task = asyncio.create_task(watcher())
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

    async def reap_once(self) -> list[str]:
        """End every idle child with nothing in flight. Returns the subjects reaped."""
        now = time.monotonic()
        doomed = [c for c in list(self._children.values())
                  if c.inflight == 0 and now - c.last_seen > self.idle_seconds]
        for child in doomed:
            # Stop routing BEFORE signalling. A request arriving in the gap would otherwise be
            # proxied into a process already shutting down, and the caller sees a connection
            # reset instead of a one-second cold start.
            if self._children.get(child.subject) is child:
                del self._children[child.subject]
            await _terminate(child.proc)
        return [c.subject for c in doomed]

    async def run_reaper(self) -> None:
        while True:
            await asyncio.sleep(self.reap_period)
            try:
                await self.reap_once()
            except Exception:
                # A bad pass must never end the reaper; the next one retries.
                pass

    def start_reaper(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self.run_reaper())

    async def shutdown(self) -> None:
        """End every child. Children share the gateway's process group, so a restart starts
        from an empty table and nothing survives to be stale."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
        for child in list(self._children.values()):
            self._children.pop(child.subject, None)
            await _terminate(child.proc)
        self._interaction_nonces.clear()
        self._login_sessions.clear()
        self._login_completions.clear()

    # ── Hosted interactive capability routing ──────────────────────────────────────────────

    def register_interaction(self, key: str, nonce: str, kind: str) -> bool:
        """Record one child-minted nonce after proving which live child minted it.

        The gateway knows only a child port, interaction kind, and short expiry. It never derives
        a profile path, opens a home, receives a site credential, or sees an OpenRouter key. The
        kinds are an allow-list of route segments, not a second home for what a site IS — the
        child supplies its own sign-in URL and the gateway never learns it.

        Registering a browser sign-in also RETIRES that child's previous one, pending or open,
        because the child closed it before calling here.
        """
        if kind not in INTERACTION_KINDS or not nonce or any(ch.isspace() for ch in nonce):
            return False
        child = self.child_for_key(key)
        if child is None:
            return False
        now = time.monotonic()
        self._expire_interactions(now)
        # A new action supersedes the one it can conflict with. A child has one OpenRouter
        # approval at a time, and ONE browser profile for every site — so a Substack sign-in
        # supersedes a pending X one, exactly as the child's own single desktop does.
        superseded = BROWSER_LOGIN_KINDS if kind in BROWSER_LOGIN_KINDS else {kind}
        for value, entry in list(self._interaction_nonces.items()):
            if entry.child is child and entry.kind in superseded:
                self._interaction_nonces.pop(value, None)
        if kind in BROWSER_LOGIN_KINDS:
            # The OPEN desktop is superseded too, and only here. `HostedLoginManager.create`
            # closes it inside the child before that child registers this nonce, so by now its
            # stream and completion capabilities address a desktop that no longer exists. They
            # would otherwise survive to the login TTL, and `live_signins` would refuse other
            # users on behalf of a desktop nobody holds.
            self._login_sessions = {token: held for token, held in self._login_sessions.items()
                                    if held.child is not child}
            self._login_completions = {
                token: held for token, held in self._login_completions.items()
                if held.child is not child
            }
        self._interaction_nonces[nonce] = InteractionNonce(
            child=child, nonce=nonce, kind=kind, expires_at=now + self.login_ttl_seconds)
        return True

    def child_for_key(self, key: str) -> Child | None:
        """The live child that holds this interaction key, proved in constant time.

        Split out of `register_interaction` on 2026-09-11 for the trial mint, which needs the
        same proof for a different purpose: it resolves the key to a child in order to read that
        child's SUBJECT. The gateway must never accept a subject a child sends — a child that
        could name its own subject could claim an allowance per name it invents.
        """
        return next((item for item in self._children.values()
                     if item.alive and item.interaction_key
                     and secrets.compare_digest(item.interaction_key, key)), None)

    def consume_interaction(self, nonce: str, kind: str) -> InteractionNonce | None:
        """Atomically consume one public capability before routing its one allowed action."""
        entry = self._interaction_nonces.get(nonce)
        if (entry is None or entry.kind != kind or entry.expires_at <= time.monotonic()
                or not entry.child.alive):
            return None
        return self._interaction_nonces.pop(nonce)

    def create_login_session(self, entry: InteractionNonce) -> tuple[str, str]:
        """Replace the login URL with one stream and one completion capability.

        RFB is a binary-only protocol, so its WebSocket cannot carry the page's explicit
        "I'm signed in" action. The two capabilities preserve the original one-use boundary:
        one attaches to the desktop, the other may complete it once.
        """
        if entry.kind not in BROWSER_LOGIN_KINDS:
            raise ValueError("only a browser sign-in owns a desktop stream")
        now = time.monotonic()
        self._expire_logins(now)
        session = secrets.token_urlsafe(32)
        completion = secrets.token_urlsafe(32)
        # A deadline of their own, starting HERE. The page this mints for is served at the
        # moment its desktop starts, and the child stamps that desktop from the same instant,
        # so inheriting the LINK's deadline made both capabilities die early by however long
        # the user took to open the link -- leaving a running desktop nobody could complete or
        # re-attach to, and uncounted by `live_signins`. Same fix the child took 2026-09-08.
        self._login_sessions[session] = LoginSession(
            child=entry.child, nonce=entry.nonce, expires_at=now + self.login_ttl_seconds)
        self._login_completions[completion] = LoginCompletion(
            child=entry.child, nonce=entry.nonce, expires_at=now + self.login_ttl_seconds)
        return session, completion

    def read_login_session(self, session: str) -> LoginSession | None:
        """Read the stream capability WITHOUT spending it. Both of its uses need it to survive.

        The page starts its own desktop before it attaches, because only the page knows how
        large to make it, and it re-attaches after every WebSocket drop -- which on a phone is
        every time its owner leaves the browser for a 2FA code. The capability lives exactly as
        long as the desktop it addresses: `_expire_logins` drops it at the login TTL, the
        deadline the child's own timer closes that desktop on.
        """
        self._expire_logins(time.monotonic())
        entry = self._login_sessions.get(session)
        if entry is None or not entry.child.alive:
            return None
        return entry

    def consume_login_completion(self, completion: str) -> LoginCompletion | None:
        """Atomically consume the separate completion action for a sign-in desktop."""
        self._expire_logins(time.monotonic())
        entry = self._login_completions.pop(completion, None)
        if entry is None or not entry.child.alive:
            return None
        return entry

    def live_signins(self) -> int:
        """Sign-in desktops open right now, across every child.

        Not a second ledger. One completion capability is minted when a desktop starts and is
        gone exactly when that desktop dies: consumed by the page's own completion, dropped by
        `_expire_logins` at the login TTL — the same deadline the child's timer closes the
        desktop on — dropped when its child exits, and dropped by `register_interaction` when a
        new sign-in supersedes it. Counting them is counting desktops, which is why the count
        needs no decrement of its own to maintain and cannot drift.
        """
        self._expire_logins(time.monotonic())
        return len(self._login_completions)

    def hold_login_child(self, child: Child) -> None:
        """Keep a child alive for its relayed browser stream, just like an MCP stream."""
        child.inflight += 1
        child.last_seen = time.monotonic()

    def release_login_child(self, child: Child) -> None:
        self.release(child)

    def _expire_logins(self, now: float) -> None:
        self._expire_interactions(now)
        self._login_sessions = {key: value for key, value in self._login_sessions.items()
                                if value.expires_at > now and value.child.alive}
        self._login_completions = {
            key: value for key, value in self._login_completions.items()
            if value.expires_at > now and value.child.alive
        }

    def _expire_interactions(self, now: float) -> None:
        self._interaction_nonces = {
            key: value for key, value in self._interaction_nonces.items()
            if value.expires_at > now and value.child.alive
        }

    # ── Introspection (ops only) ────────────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        return [{"subject": c.subject, "pid": c.proc.pid, "port": c.port,
                 "inflight": c.inflight, "idle_seconds": round(now - c.last_seen, 1)}
                for c in self._children.values()]


async def _listening(port: int) -> bool:
    """True once something accepts on 127.0.0.1:port.

    Never "localhost": this resolves to ::1 first on macOS, and a child bound only to IPv4
    would look dead until the spawn deadline.
    """
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal every process the child started, not just the child.

    `_spawn` passes `start_new_session=True`, so a child of ours always leads its own group and
    the group id IS its pid. The confirmation before signalling is not a re-check of our own
    construction — it is what keeps a destructive syscall aimed at a target we own. The pid
    arrives on an object the caller supplies, and `killpg` on a pid that leads no group of its
    own would signal whatever group that pid happens to sit in.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, sig)


async def _terminate(proc: asyncio.subprocess.Process, grace: float = 5.0) -> None:
    """SIGTERM the child's whole process group, then SIGKILL it if that is ignored.

    SIGTERM rather than SIGKILL first so the child closes its SQLite connections and flushes
    the write-ahead log; a hosted child is disposable, but its home is not.

    The GROUP, not the process. A child that has done any hosted browser work owns a Chrome it
    started, and SIGTERM's default disposition ends the interpreter WITHOUT unwinding — so the
    context manager holding that browser never reaches its `finally`, and Chrome survives its
    parent. Measured 2026-09-08: two headless Chromes reparented to init, ten hours old, still
    holding the profile's `SingletonLock`. The next real sign-in launched a Chrome that took one
    look at that lock, handed its URL to the orphan and exited 21, leaving a live desktop with
    nothing drawn on it. Every other process in that desktop was healthy, so nothing reported a
    fault and the user watched a black rectangle until they gave up.
    """
    if proc.returncode is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except (asyncio.TimeoutError, TimeoutError):
        _signal_group(proc, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
