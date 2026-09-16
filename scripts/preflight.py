#!/usr/bin/env python3
"""
scripts/preflight.py — is the thing I am about to test actually the code in front of me?

⚠️ WRITTEN AFTER A TEST RUN THAT ANSWERED "NO" IN THREE PLACES AT ONCE (2026-09-14). A full
onboarding was run to measure a build; the results were read as evidence about that build; and
underneath, all of this was true and none of it was visible:

  • seven `opyt-mcp` processes were alive from a DIFFERENT checkout than the one being edited;
  • the resident worker's plist launched `uvx --from opyt==0.1.0a5`, the PUBLISHED wheel — the
    same version string as the worktree and a different codebase, on the same database;
  • the client had not been restarted since the last commit, so the server in the conversation
    was older than the code on disk.

Every one of those turns a measurement into a guess, and the failure mode is the expensive one:
the run looks fine, the numbers look real, and the conclusion is about software nobody has.

So this answers ONE question — does everything that will execute during the next test resolve to
THIS commit? — and it is deliberately read-only. It changes nothing; it prints what is wrong and
the exact command that fixes it. A doctor that also operates is a doctor you stop reading.

    python scripts/preflight.py            # check
    python scripts/preflight.py --json     # same, machine-readable

Exit: 0 all clear, 1 something would be tested that is not this code.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OK, WARN, BAD = "ok", "warn", "bad"


def _run(*argv: str) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _check(name: str, level: str, detail: str, fix: str = "") -> dict:
    return {"name": name, "level": level, "detail": detail, "fix": fix}


def check_worktree() -> dict:
    """Which commit, and whether anything is uncommitted. INFORMATIONAL — a dirty tree is the
    normal state mid-change, and the editable install means those edits ARE what runs. It is
    reported because "which commit did we measure" is unanswerable afterwards otherwise."""
    head = _run("git", "-C", str(REPO), "log", "--oneline", "-1")
    dirty = [ln for ln in _run("git", "-C", str(REPO), "status", "--short").splitlines() if ln]
    detail = f"{head}" + (f"  ·  {len(dirty)} uncommitted file(s)" if dirty else "  ·  clean")
    return _check("worktree", OK, detail)


def check_this_venv() -> dict:
    """Does the venv beside this checkout import THIS checkout?

    An editable install makes source edits live, which is what makes a test meaningful at all. A
    NON-editable venv is the quiet failure: it imports a copy taken at install time, so every edit
    since is invisible and the run measures history.
    """
    py = REPO / "venv" / "bin" / "python"
    if not py.exists():
        return _check("venv", BAD, f"no venv at {py}", fix="python -m venv venv && venv/bin/pip install -e .")
    out = _run(str(py), "-c", "import opyt_core, sys; print(opyt_core.__file__)")
    if not out:
        return _check("venv", BAD, "venv cannot import opyt_core",
                      fix=f"{py} -m pip install -e {REPO}")
    resolved = Path(out).resolve()
    if REPO in resolved.parents:
        return _check("venv", OK, f"editable → {resolved.parent}")
    return _check("venv", BAD, f"imports {resolved} — NOT this checkout",
                  fix=f"{py} -m pip install -e {REPO}")


def _client_config() -> Path:
    return (Path.home() / "Library" / "Application Support" / "Claude"
            / "claude_desktop_config.json")


def check_client_config() -> dict:
    """Which binary the MCP client is configured to launch, and whether it is ours."""
    cfg = _client_config()
    if not cfg.exists():
        return _check("client config", WARN, f"not found at {cfg}")
    try:
        servers = (json.loads(cfg.read_text()).get("mcpServers") or {})
    except Exception as e:
        return _check("client config", BAD, f"unreadable: {type(e).__name__}: {e}")
    ours = {n: s for n, s in servers.items() if "opyt" in n.lower()}
    if not ours:
        return _check("client config", BAD, "no Opyt server configured")
    lines, bad = [], False
    for name, spec in ours.items():
        cmd = spec.get("command", "")
        here = REPO in Path(cmd).resolve().parents if cmd.startswith("/") else False
        lines.append(f"{name} → {cmd}" + ("" if here else "   ← NOT this checkout"))
        bad = bad or not here
    return _check("client config", BAD if bad else OK, "; ".join(lines),
                  fix=f'point "command" at {REPO}/venv/bin/opyt-mcp, then restart the client'
                      if bad else "")


def _proc_table() -> list[tuple[str, str, str]]:
    """(pid, start, command) for every live opyt-mcp / opyt-worker."""
    out = _run("ps", "-eo", "pid=,lstart=,command=")
    rows = []
    for ln in out.splitlines():
        if "opyt-mcp" not in ln and "opyt-worker" not in ln:
            continue
        if "disclaimer" in ln or "preflight.py" in ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        pid, rest = parts
        start, cmd = rest[:24].strip(), rest[24:].strip()
        rows.append((pid, start, cmd))
    return rows


def check_processes() -> dict:
    """Live servers, and which checkout each came from.

    ⚠️ THE ONE THAT IS EASIEST TO MISS. A stdio MCP server is spawned per client connection and
    OUTLIVES the conversation that made it; nothing reaps them. The measured run had nine alive
    across two checkouts, and the conversation was talking to exactly one of them without saying
    which. A stale server is not idle — it answers tool calls with old code.
    """
    rows = _proc_table()
    if not rows:
        return _check("live servers", OK, "none running (a fresh client start is a clean one)")
    foreign, mine = [], []
    for pid, start, cmd in rows:
        (mine if str(REPO) in cmd else foreign).append(f"pid {pid} · {start}")
    detail = f"{len(mine)} from this checkout, {len(foreign)} from elsewhere"
    if foreign:
        return _check("live servers", WARN,
                      detail + " — " + "; ".join(foreign[:4]),
                      fix="pkill -f opyt-mcp   # then restart the client")
    return _check("live servers", OK, detail)


def check_worker() -> dict:
    """The resident worker: installed? loaded? and — the part that bit — running WHICH build?"""
    sys.path.insert(0, str(REPO))
    try:
        from opyt_core import install_worker
        st = install_worker.status()
        would = install_worker.running_distribution()
    except Exception as e:
        return _check("worker", WARN, f"cannot probe: {type(e).__name__}: {e}")

    if not st["installed"]:
        return _check("worker", WARN,
                      f"not installed (would launch the {would['kind']} build). Scheduled "
                      f"refreshes and queued rail jobs will not run.",
                      fix="consent to recurring updates during onboarding — it installs itself")
    launches = st.get("agent_launches") or []
    here = any(str(REPO) in str(a) for a in launches)
    state = "loaded" if st["loaded"] else "installed but NOT loaded"
    if here:
        return _check("worker", OK if st["loaded"] else WARN, f"{state} → this checkout")
    return _check("worker", BAD,
                  f"{state} → {' '.join(str(a) for a in launches)} — a DIFFERENT build from the "
                  f"server you are testing, on the same store",
                  fix=f"{REPO}/venv/bin/opyt-install-worker   # rewrites it to this checkout")


def check_home() -> dict:
    """Which data home, how populated, and whether any rail job is wedged.

    A job with `started_at` set and `finished_at` NULL is a claim nobody released — the shape a
    killed or mis-installed worker leaves. It reads as "in progress" forever, and the next worker
    may skip it."""
    sys.path.insert(0, str(REPO))
    try:
        from opyt_core.paths import opyt_home
        home = opyt_home()
    except Exception as e:
        return _check("data home", WARN, f"cannot resolve: {type(e).__name__}: {e}")
    env = f" (OPYT_HOME={os.environ['OPYT_HOME']})" if os.environ.get("OPYT_HOME") else ""
    if not home.exists():
        return _check("data home", OK, f"{home}{env} — does not exist yet (a true cold start)")
    db = home / "opyt.db"
    size = f"{db.stat().st_size / 1e6:.0f} MB" if db.exists() else "no store yet"
    wedged = []
    try:
        from pipeline.kb.rail_jobs import RailJobStore
        wedged = [j.rail for j in RailJobStore().list_jobs()
                  if j.started_at is not None and j.finished_at is None]
    except Exception:
        pass
    if wedged:
        return _check("data home", WARN, f"{home}{env} · {size} · claimed-but-unfinished: "
                                         f"{', '.join(wedged)}",
                      fix="clear those rows before measuring, or a worker may skip them")
    return _check("data home", OK, f"{home}{env} · {size}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args(argv)

    checks = [check_worktree(), check_this_venv(), check_client_config(),
              check_processes(), check_worker(), check_home()]
    worst = BAD if any(c["level"] == BAD for c in checks) else (
        WARN if any(c["level"] == WARN for c in checks) else OK)

    if args.json:
        print(json.dumps({"status": worst, "repo": str(REPO), "checks": checks}, indent=2))
        return 0 if worst != BAD else 1

    mark = {OK: "  ok ", WARN: " warn", BAD: " BAD "}
    print(f"\nPREFLIGHT — {REPO}\n")
    for c in checks:
        print(f"[{mark[c['level']]}] {c['name']:<14} {c['detail']}")
        if c["fix"]:
            print(f"{'':>9}{'':<14} → {c['fix']}")
    print()
    if worst == BAD:
        print("NOT SAFE TO MEASURE: something that will run is not this code.\n")
    elif worst == WARN:
        print("Runnable, but read the warnings — each one can make a result mean "
              "something other than what it looks like.\n")
    else:
        print("Everything that will run resolves to this checkout.\n")
    return 0 if worst != BAD else 1


if __name__ == "__main__":
    raise SystemExit(main())
