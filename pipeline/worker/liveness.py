"""
pipeline/worker/liveness.py — the one answer to "will something claim this home's work?", which
every promise of background work reads.

`repair_or_report_cached` (or, for `onboard`, the full `repair_or_report`) is the read behind
every "it fills in on its own" sentence. It reads the heartbeat first, on any platform; only a
silent local Mac is repaired, through the LaunchAgent installer (`opyt_core/install_worker.py`):
a sticky launchd disable, an agent missing or not loaded. Hosted has nothing to install: the
box's resident worker claims every user's rows.
"""

from __future__ import annotations

import os
import time

from opyt_core import install_worker
from opyt_core.paths import opyt_home

from . import home as worker_home
from . import job_queue


def status() -> dict:
    """Is the resident worker actually there? `install_worker.agent_state()`, plus
    `{"beating"}`. A probe: it never writes, and never creates the jobs database.

    The worker is the only thing that launches rails, so any promise of background work is a
    claim about this. `beating` answers "is anything claiming now" (`job_queue.worker_is_alive`),
    local or hosted, including a worker run by hand. An unreadable plist or launchctl reports
    False rather than raising.
    """
    return {**install_worker.agent_state(), "beating": job_queue.worker_is_alive()}


# Where a user on Windows or Linux goes instead: local installs are macOS-only, and a library is
# only ever built by an install. One string, because `onboard` and this module's report both say it.
USE_A_MAC = ("If they have a Mac, set OPYT up there (useopyt.com); its library can then be read "
             "from claude.ai or ChatGPT on any computer through the Opyt connector.")

# How long `repair_or_report` waits for a first beat before it judges. A live loop beats on its
# first pass after waking or starting, so a few seconds separates "just woke" from "not running".
REPORT_PATIENCE = 10.0


# How long `repair_or_report_cached` trusts the last answer, for hot paths that ask on every call;
# far inside `BEAT_STALE_AFTER`.
REPORT_CACHE_SECONDS = 30.0

# home key -> (monotonic time, report). Keyed by home so a process that switches `$OPYT_HOME`
# (the test suite) never reads another home's answer.
_REPORT_CACHE: dict[tuple, tuple[float, dict]] = {}


def repair_or_report_cached(*, max_age: float | None = None) -> dict:
    """`repair_or_report(patience=0)`, reused for `REPORT_CACHE_SECONDS`. Same shape.

    The one read behind every promise of background work: such a sentence reads this, or the
    full `repair_or_report`, and nothing else. Never waits, so it is safe on any tool call; it
    still fixes what it finds, at most once per window.
    """
    key = _cache_key()
    window = REPORT_CACHE_SECONDS if max_age is None else max_age
    hit = _REPORT_CACHE.get(key)
    if hit is not None and time.monotonic() - hit[0] < window:
        return hit[1]
    return repair_or_report(patience=0)


def _cache_key() -> tuple:
    return (str(opyt_home()), os.environ.get(worker_home.WORKER_DB_ENV),
            os.environ.get(worker_home.WORKER_HOME_ID_ENV))


def repair_or_report(*, patience: float | None = None) -> dict:
    """Make sure something is claiming this home's work, or say exactly why not.
    `{"ok", "status", "cause", "step", "detail"}`.

    Fix the worker; never fall back. This either leaves a beating worker behind it, or returns
    the cause in plain words and the one step the user can take. `cause` and `step` are for the
    person; `detail` is for whoever debugs it.

    In order, stopping at the first thing that works:
      1. a live beat — `ok`, nothing touched, on any platform; off macOS, nothing else is
         tried, and a silent home is `unsupported`;
      2. a sticky launchd disable — `repair()`, the narrow fix;
      3. no plist — `install()`; installed but not loaded — `install()` again, which reloads;
      4. wait up to `patience` seconds for a beat. Never judge on one read: a Mac just woken
         reads dead until the loop's next pass, which beats at once.

    Hosted has nothing to install: the box's resident worker claims every user's rows, so a dead
    beat there is ours to fix, and the report says so instead of handing the user a step.

    Fail-safe: never raises. A probe that cannot answer reports that, rather than claiming the
    worker is fine. Every answer is kept for `repair_or_report_cached`.
    """
    report = _repair_or_report(REPORT_PATIENCE if patience is None else patience)
    _REPORT_CACHE[_cache_key()] = (time.monotonic(), report)
    return report


def _repair_or_report(patience: float) -> dict:
    """Gather (the queue, the agent's state and the beat), decide (`report_before_acting`, then
    `action_for`), act (repair or install), then judge after a short wait for a beat
    (`report_after`)."""
    try:
        if worker_home.queue_is_shared():
            return hosted_report(job_queue.worker_is_alive())
        st = status()
        if (early := report_before_acting(st)) is not None:
            return early
        try:
            acted = _act(action_for(st))
        except FileNotFoundError as e:
            return uv_missing(e)
        if refused(acted):
            return report_after(acted, None, [])     # judged now: no beat can follow a refusal
        return report_after(acted, _wait_for_beat(patience), _log_tail())
    except Exception as e:
        return {"ok": False, "status": "unknown",
                "cause": "OPYT couldn't check its background helper.",
                "step": "Call `onboard` again in a minute.",
                "detail": {"error": f"{type(e).__name__}: {e}"}}


def _act(action: str | None) -> dict | None:
    """Do what `action_for` decided, through the LaunchAgent installer: `repair`, `install`, or
    nothing. Raises `FileNotFoundError` when the launcher an install would write is missing."""
    if action == "repair":
        return install_worker.repair()
    if action == "install":
        return install_worker.install()
    return None


def report_before_acting(st: dict) -> dict | None:
    """The report that needs no action, from `status()`: a live beat is `running` on any
    platform, a worker run by hand included; a silent home off macOS (`supported`, the one
    platform check) is `unsupported`. None: a silent Mac, which `action_for` repairs."""
    if st["beating"] is True:
        return {"ok": True, "status": "running", "cause": None, "step": None, "detail": None}
    if not st["supported"]:
        return {"ok": False, "status": "unsupported",
                "cause": "OPYT's background helper runs on macOS only.",
                "step": USE_A_MAC, "detail": None}
    return None


def uv_missing(e: FileNotFoundError) -> dict:
    """`uvx_command` raising: the launcher this install would write does not exist."""
    return {"ok": False, "status": "not_installed",
            "cause": "OPYT couldn't set up its background helper because uv, the "
                     "tool it runs with, isn't installed.",
            "step": "Install uv (https://docs.astral.sh/uv/), then call `onboard` again.",
            "detail": {"error": str(e)}}


# What a successful action is called once the worker is confirmed beating.
_ACTED = {"REPAIRED": "repaired", "INSTALLED": "installed", "UPDATED": "reloaded"}


def hosted_report(alive: bool | None) -> dict:
    """The report on hosted, where the box's resident worker claims every user's rows. None
    (unreadable) reads as resident, not down."""
    if alive is False:
        return {"ok": False, "status": "hosted_down",
                "cause": "OPYT's background service is down right now. That is on "
                         "OPYT's side, and the work waits for it.",
                "step": None, "detail": None}
    return {"ok": True, "status": "resident", "cause": None, "step": None, "detail": None}


def action_for(st: dict) -> str | None:
    """What fixes a local worker that is not beating, from `status()`: `repair` a sticky launchd
    disable, `install` an agent that is missing or not loaded (which reloads it), or nothing."""
    if st["installed"] and st.get("disabled") is True:
        return "repair"
    if not st["installed"] or not st["loaded"]:
        return "install"
    return None


def refused(acted: dict | None) -> bool:
    """launchd would not load the agent: the installer's status starts with `ERROR`."""
    return str((acted or {}).get("status", "")).startswith("ERROR")


def report_after(acted: dict | None, beat: bool | None, log_tail: list[str]) -> dict:
    """The report once the wait for a beat is over, given what was done (`acted`) and the last
    read of the heartbeat (`beat`). `log_tail` is the worker log's last lines, for the case that
    needs them. A refusal is judged first, whatever the beat; the flow asks for it before the
    wait."""
    if refused(acted):
        return _refused_to_load(acted)
    if beat is True:
        return {"ok": True, "status": _ACTED.get((acted or {}).get("status"), "running"),
                "cause": None, "step": None, "detail": acted}
    if (acted or {}).get("status") in _ACTED and beat is False:
        # Just put back and not beating yet: "keeps stopping" would be a guess. A worker that
        # really dies on start is caught on the next read.
        return {"ok": True, "status": "starting", "cause": None, "step": None,
                "detail": acted}
    if beat is None:
        return {"ok": False, "status": "unreadable",
                "cause": "OPYT can't read its own job queue right now, so it can't tell "
                         "whether the background helper is working.",
                "step": "Call `onboard` again in a minute.", "detail": acted}
    # Loaded, or just loaded, and silent through the wait: it starts and dies — most likely
    # a release that crashes on start. The log is the evidence (the plist sends both streams
    # there), and it goes in `detail`, never in the words for the person.
    return {"ok": False, "status": "not_beating",
            "cause": "OPYT's background helper is installed but keeps stopping.",
            "step": "Nothing to do right now: it retries every minute, and a fixed version "
                    "reaches it on its own. If this keeps showing, send the log lines OPYT "
                    "reports to hello@useopyt.com.",
            "detail": {"acted": acted, "log_tail": log_tail}}


def _wait_for_beat(patience: float) -> bool | None:
    """Read the heartbeat once a second for up to `patience` seconds; the last read wins.
    True at the first live beat. None only when the store stayed unreadable to the end."""
    deadline = time.monotonic() + patience
    while True:
        beat = job_queue.worker_is_alive()
        if beat is True or time.monotonic() >= deadline:
            return beat
        time.sleep(1.0)


def _refused_to_load(res: dict) -> dict:
    """launchd would not load the agent; the likely cause is a sticky disable, and the raw detail
    is kept for whoever debugs it."""
    return {"ok": False, "status": "refused",
            "cause": "macOS stopped OPYT's background helper from running.",
            "step": "Open System Settings → General → Login Items & Extensions, and under "
                    "\"Allow in the Background\" switch OPYT's helper back on.",
            "detail": res}


def _log_tail(lines: int = 20) -> list[str]:
    try:
        text = (opyt_home() / install_worker.WORKER_LOG).read_text(errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]
