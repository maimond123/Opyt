"""
opyt_core/install_worker.py

Installs the resident rail worker as a macOS LaunchAgent, so a local OPYT keeps ingesting
without an MCP session open. The hosted equivalent is `gateway/deploy/opyt-worker.service`;
both launch the same `opyt-worker` entry point against the same durable jobs database.
Design record: `docs/plans/2026-09-06-persistent-rail-worker-migration.md`.

    opyt-install-worker
    opyt-install-worker --dry-run
    opyt-install-worker --uninstall

**macOS only, deliberately.** Ruling F1 approves a LaunchAgent and the InterServer systemd
service; a Linux user unit and a Windows scheduled task are not approved and are not written
here, because an unverified resident service is worse than an honest absence. The worker core
is portable and can be run by hand anywhere with `opyt-worker`.

A LaunchAgent, not a LaunchDaemon: the worker runs as the user, reads that user's `$OPYT_HOME`,
and has nothing to do before login. It opens no socket and exposes no port — an MCP client
reaches the tools through the stdio server, never through this process.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from opyt_core.install_client import uvx_command
from opyt_core.paths import opyt_home

LABEL = "com.useopyt.worker"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
WORKER_LOG = "worker.log"


def running_distribution() -> dict:
    """WHICH COPY OF OPYT IS EXECUTING RIGHT NOW — `{"kind", "root", "launcher"}`.

    ⚠️ VERSION IS NOT IDENTITY, and that is the whole reason this exists. A checkout and the
    published wheel report the SAME `version('opyt')`, so `uvx --from opyt==0.1.0a5 opyt-worker`
    looks like the code in front of you and is not: on 2026-09-14 a worker installed from this
    worktree launched the PyPI build against the same store, with an identical version string and
    nothing anywhere saying the two differed. Two codebases on one database is a debugging
    nightmare that presents as "the fix didn't work".

    `source` when the running package sits in a tree with its own `pyproject.toml` — an editable
    install, i.e. a checkout. `published` otherwise. The launcher follows: a checkout launches its
    OWN console script, a published install goes through `uvx_command`.

    The Distributable invariant is intact and this is not the case it bans. What 2026-08-29 removed
    was a PUBLISHED install welding its config to a folder that could move; a source install is
    already a folder, and pointing its worker anywhere else is what produces the mismatch above.
    """
    import opyt_core

    root = Path(opyt_core.__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists():
        script = Path(sys.executable).with_name("opyt-worker")
        if script.exists():
            return {"kind": "source", "root": str(root), "launcher": [str(script)]}
        # An editable install whose console script is missing (a bare `pip install -e` gone
        # sideways). Run the module through the SAME interpreter rather than silently reaching
        # for a published build that is not this code.
        return {"kind": "source", "root": str(root),
                "launcher": [sys.executable, "-m", "pipeline.kb.rail_worker"]}
    return {"kind": "published", "root": str(root), "launcher": uvx_command("opyt-worker")}


def agent_plist() -> dict:
    """The LaunchAgent definition, as the dict `plistlib` will write.

    `KeepAlive` is unconditional, with a minute between attempts. Unconditional because the
    case worth restarting — the worker or a rail child dying mid-pass — is indistinguishable
    from the outside; a minute because the other repeatable exit is `WorkerAlreadyRunning`,
    when someone is running `opyt-worker` by hand. That retry is not wasted: the agent takes
    over the moment the hand-run worker stops. Its one line per minute goes to the worker log,
    which is where a user looking for "why is nothing ingesting" already is.
    """
    dist = running_distribution()
    plist: dict = {
        "Label": LABEL,
        # The SAME copy of opyt that is installing it — see `running_distribution`.
        "ProgramArguments": dist["launcher"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 60,
        "StandardOutPath": str(opyt_home() / WORKER_LOG),
        "StandardErrorPath": str(opyt_home() / WORKER_LOG),
    }
    # launchd starts an agent with almost no environment, so a data home chosen by this shell
    # would otherwise be lost and the agent would schedule work for `~/.opyt` instead. Pinned
    # only when it was chosen explicitly; the default stays derived at runtime.
    if os.environ.get("OPYT_HOME"):
        plist["EnvironmentVariables"] = {"OPYT_HOME": str(opyt_home())}
    return plist


def status() -> dict:
    """Is the resident worker actually there? `{"supported", "installed", "loaded", "ran"}`.

    ⚠️ ASK THIS BEFORE TELLING A USER THAT WORK CONTINUES ON ITS OWN. The worker is the ONLY
    thing that launches rails (`rail_worker.RAILS`), so every "we'll finish this in the
    background" sentence is a claim about this function. Those sentences were unconditional until
    2026-09-13 and false on any from-source install: probing David's machine that day found no
    LaunchAgent, no `launchctl` entry, no rail log, and two jobs sitting in `rail_jobs.db` with
    `started_at` NULL since 13:50 — while `oracle`'s ingest result told him a rate-limited Oracle
    "will continue automatically". A promise nothing keeps costs more than the missing feature,
    because the user stops waiting for something that never starts.

    `ran` is the separate, platform-independent evidence: a rail that has actually finished means
    SOMETHING drains the queue — including a worker run by hand (`opyt-worker`), which is the
    documented way to have one off macOS and which no plist check can see.

    Fail-safe in both directions it can fail: an unreadable plist, launchctl, or jobs database
    reports False rather than raising, so a probe that cannot answer under-promises instead of
    breaking the tool call it was asked from.
    """
    supported = sys.platform == "darwin"
    try:
        installed = PLIST_PATH.exists()
    except OSError:
        installed = False
    loaded = False
    if supported and installed:
        try:
            loaded = _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0
        except Exception:
            loaded = False
    # WHAT IT WOULD LAUNCH, and what an installed one DOES launch. A worker running different
    # code from the server that installed it is invisible without this, and it presents as a fix
    # that did not take.
    out = {"supported": supported, "installed": installed, "loaded": loaded, "ran": _any_rail_ran()}
    try:
        out["running_from"] = running_distribution()["kind"]
    except Exception:
        out["running_from"] = "unknown"
    try:
        if installed:
            out["agent_launches"] = plistlib.loads(PLIST_PATH.read_bytes())["ProgramArguments"]
    except Exception:
        pass
    return out


def _any_rail_ran() -> bool:
    """Has any rail job ever been claimed? Evidence that something drains the queue.

    Deliberately "ever", not "recently": this separates a store that has never had a worker from
    one whose worker is merely idle, and only the first needs saying out loud. A recency window
    would need a threshold nothing can defend, and would nag a user whose rails are simply not due.
    """
    try:
        from pipeline.kb.rail_jobs import RailJobStore
        return any(j.started_at is not None for j in RailJobStore().list_jobs())
    except Exception:
        return False


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def install(*, dry_run: bool = False) -> dict:
    """Write the agent and load it. Idempotent: an existing agent is replaced, not duplicated."""
    plist = agent_plist()
    if dry_run:
        return {"status": "DRY_RUN", "path": str(PLIST_PATH), "would_write": plist}

    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    opyt_home().mkdir(parents=True, exist_ok=True)      # launchd will not create the log's dir
    existed = PLIST_PATH.exists()
    PLIST_PATH.write_bytes(plistlib.dumps(plist))

    # Bootout first, unconditionally: bootstrapping over a loaded label fails, and a label that
    # is not loaded is not an error worth reporting.
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    loaded = _launchctl("bootstrap", _domain(), str(PLIST_PATH))
    if loaded.returncode != 0:
        return {"status": "ERROR_LOAD", "path": str(PLIST_PATH),
                "detail": (loaded.stderr or loaded.stdout).strip()}
    return {"status": "UPDATED" if existed else "INSTALLED", "path": str(PLIST_PATH),
            "log": str(opyt_home() / WORKER_LOG)}


def uninstall(*, dry_run: bool = False) -> dict:
    """Stop the agent and remove it. Leaves the jobs database — a reinstall resumes it."""
    if not PLIST_PATH.exists():
        return {"status": "NOT_PRESENT", "path": str(PLIST_PATH)}
    if dry_run:
        return {"status": "DRY_RUN_REMOVE", "path": str(PLIST_PATH)}
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    PLIST_PATH.unlink()
    return {"status": "REMOVED", "path": str(PLIST_PATH)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opyt-install-worker",
        description="Install (or remove) the OPYT rail worker as a macOS LaunchAgent.")
    ap.add_argument("--uninstall", action="store_true", help="remove the agent instead")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args(argv)

    if sys.platform != "darwin":
        print("This installs a macOS LaunchAgent. On Linux, run `opyt-worker` from a systemd "
              "user unit you manage; no unit is shipped, because none is verified yet.",
              file=sys.stderr)
        return 1

    try:
        res = uninstall(dry_run=args.dry_run) if args.uninstall else install(dry_run=args.dry_run)
    except FileNotFoundError as e:
        res = {"status": "ERROR", "detail": str(e)}

    line = f"[{res['status']}] {LABEL} -> {res.get('path', '')}"
    if res.get("log"):
        line += f"\n    logs: {res['log']}"
    if res.get("detail"):
        line += f"\n    {res['detail']}"
    if res.get("would_write"):
        line += "\n" + plistlib.dumps(res["would_write"]).decode()
    print(line)
    return 1 if str(res["status"]).startswith("ERROR") else 0


if __name__ == "__main__":
    raise SystemExit(main())
