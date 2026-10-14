"""
opyt_core/install_worker.py

Installs the resident rail worker as a macOS LaunchAgent, so a local OPYT keeps ingesting
without an MCP session open. It launches the `opyt-worker` entry point against the home's
durable jobs database.

    opyt-install-worker
    opyt-install-worker --dry-run
    opyt-install-worker --uninstall

macOS only: local installs are macOS-only, and off macOS this command says so and does nothing.

A LaunchAgent, not a LaunchDaemon: the worker runs as the user, reads that user's `~/.opyt`,
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

from opyt_core.launcher import running_distribution
from opyt_core.paths import opyt_home

LABEL = "com.useopyt.worker"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
WORKER_LOG = "worker.log"


def agent_plist(launcher: list[str], log: Path) -> dict:
    """The LaunchAgent definition, as the dict `plistlib` will write: `launcher` runs at login
    and its output goes to `log`.

    `KeepAlive` is unconditional, with a minute between attempts: a crash is indistinguishable
    from outside, and the other repeatable exit (`WorkerAlreadyRunning`, a hand-run worker) should
    be retried so the agent takes over when it stops.
    """
    plist: dict = {
        "Label": LABEL,
        "ProgramArguments": launcher,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 60,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }
    # No `EnvironmentVariables`, deliberately: a local install's data home is `~/.opyt`. Passing
    # `$OPYT_HOME` here would split the worker and the MCP server onto two different stores.
    return plist


def agent_state() -> dict:
    """The LaunchAgent as launchd sees it: `{"supported", "installed", "loaded", "disabled"}`,
    plus what it would launch (`running_from`) and, when installed, what it does launch
    (`agent_launches`). `pipeline.worker.liveness.status` adds whether anything is claiming the
    work; that is the question every promise of background work asks.

    Fail-safe: an unreadable plist or launchctl reports False rather than raising.
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
    # What it would launch and what an installed one does launch: a worker running different code
    # from the server that installed it is otherwise invisible.
    out = {"supported": supported, "installed": installed, "loaded": loaded,
           # Read-only; `repair()` is the verb that acts on it.
           "disabled": _disabled() if (supported and installed) else None}
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


def _disabled() -> bool | None:
    """Is the label in launchd's persistent disable list? `None` when it cannot be read.

    Separate from `loaded` because it is the one not-loaded reason that never clears on its own:
    the flag survives reinstalls and reboots, and every later `bootstrap` fails with errno 5.
    One click in Login Items sets it.

    `print-disabled` lists the label when enabled too, so match the `=> disabled` tail.
    """
    try:
        res = _launchctl("print-disabled", _domain())
        if res.returncode != 0:
            return None
        for line in res.stdout.splitlines():
            if LABEL in line:
                return line.strip().endswith("=> disabled")
        return False          # absent from the list = never disabled
    except Exception:
        return None


def repair(*, dry_run: bool = False) -> dict:
    """Clear a sticky launchd disable and load the agent. No-op unless that is the exact fault.

    Fires only when the agent is installed, not loaded, and measurably disabled, so it can never
    resurrect an agent a user removed (`--uninstall` deletes the plist). Re-enables silently.
    Called from `bootstrap.ensure_initialized` and `liveness.repair_or_report`.
    """
    if sys.platform != "darwin":
        return {"status": "UNSUPPORTED"}
    if not needs_repair(agent_state()):
        return {"status": "NO_ACTION"}
    if dry_run:
        return {"status": "DRY_RUN_REPAIR", "would_run": f"enable {_domain()}/{LABEL}"}
    loaded, detail = _load_agent()
    return {"status": "REPAIRED" if loaded else "ERROR_LOAD", "detail": detail or None}


def needs_repair(state: dict) -> bool:
    """Is this the one fault `repair` fixes: installed, not loaded, and measurably disabled?
    An unreadable disable list (`None`) is not a yes."""
    return state["installed"] and not state["loaded"] and state.get("disabled") is True


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _load_agent() -> tuple[bool, str]:
    """Enable the label, then bootstrap the plist. Returns (loaded, launchd's message).

    Enable first, unconditionally: a disabled label refuses to bootstrap, and `bootout` does not
    clear a disable. Note `enable` takes a target (domain/label); `bootstrap` takes a domain plus
    a path.
    """
    _launchctl("enable", f"{_domain()}/{LABEL}")
    res = _launchctl("bootstrap", _domain(), str(PLIST_PATH))
    return res.returncode == 0, (res.stderr or res.stdout).strip()


def install(*, dry_run: bool = False) -> dict:
    """Write the agent and load it. Idempotent: an existing agent is replaced, not duplicated."""
    # The same copy of opyt that is installing it. Raises FileNotFoundError, before anything is
    # written, when a published install has no uvx.
    log = opyt_home() / WORKER_LOG
    plist = agent_plist(running_distribution()["launcher"], log)
    if dry_run:
        return {"status": "DRY_RUN", "path": str(PLIST_PATH), "would_write": plist}

    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)       # launchd will not create the log's dir
    existed = PLIST_PATH.exists()
    PLIST_PATH.write_bytes(plistlib.dumps(plist))

    # Bootout first, unconditionally: bootstrapping over a loaded label fails, and a label that
    # is not loaded is not an error worth reporting.
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    loaded, detail = _load_agent()
    if not loaded:
        out = {"status": "ERROR_LOAD", "path": str(PLIST_PATH), "detail": detail}
        # `cause` lets a host say something actionable instead of relaying errno 5.
        if _disabled():
            out["cause"] = "launchd_disabled"
            out["remedy"] = f"launchctl enable {_domain()}/{LABEL}"
        return out
    return {"status": "UPDATED" if existed else "INSTALLED", "path": str(PLIST_PATH),
            "log": str(log)}


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
        print("opyt-install-worker runs on macOS only.", file=sys.stderr)
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
