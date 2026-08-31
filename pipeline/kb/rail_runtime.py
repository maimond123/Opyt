"""
pipeline/kb/rail_runtime.py — the mechanics every background rail shares.

A RAIL is a loop that runs detached on session open: `bookmark_catchup`, `curation_catchup`,
`probe_catchup`, `frontier_admit`, `frontier_execute`, `oracle_refresh`, `sitting_scheduler`.
This shares their near-identical spawn MECHANISM (`spawn_rail`, below), never their ownership —
the `atom-rail-not-welded-to-catchup` guard bans centralizing the spawners themselves, since a
past central runner took two rails down with a third. Each rail keeps its own entry point, env
vars, stamp, log, and `try/except` in `mcp_server/server.py`.

Each rail's `main()` stays unshared, deliberately: they take different CLI args and map
different status vocabularies onto the exit code.

"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from opyt_core.paths import opyt_path

# Every rail coalesces its spawn on this window unless it passes its own. One hour is the
# settled default; `oracle_refresh` uses 600s because its unit of work is far smaller.
COALESCE_DEFAULT = 3600.0


def load_rail_env() -> None:
    """Load credentials into THIS process — call it first in any rail's runner. A detached child
    inherits only whatever environment the MCP server itself had, so without this it sees none
    of the user's keys. `override=True` on the user file so a rotated key on disk beats a stale
    copy cached in the server's environment; the repo `.env` is a dev fallback and does not
    override. Fail-safe: a missing `dotenv` or unreadable file must not stop the rail."""
    try:
        from dotenv import load_dotenv
        load_dotenv(opyt_path(".env"), override=True)
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")     # dev fallback (repo root)
    except Exception:
        pass


def refusal_marker(rail: str) -> Path:
    """Where a rail records that its start gate turned a run away. ONE owner for a path with a
    writer here and a reader in `rail_budgets`, for the same reason `RAIL` is one constant rather
    than two matching literals: a drifted pair fails silently, with the writer filling one file
    while the reader watches another."""
    return opyt_path(f"{rail}_budget_refused")


def rail_budget_exhausted(rail: str, ceiling_usd: float) -> bool:
    """Has this rail's recorded spend today reached this rail's ceiling? Uses
    `spend_today_for_rail(rail)`, NEVER the global `spend_today()` — two rails sharing a pool
    silently let whichever ran first drain the other's allowance
    (`rail-ceilings-use-their-own-meter`). Reads the in-memory `_STATS`, not the on-disk
    `api_stats.json`, which is stale by exactly the spend a long-lived run just made.

    ON TRUE IT STAMPS `<rail>_budget_refused`, and that marker is the only record anywhere that a
    run was REFUSED — as opposed to a run that spent its ceiling on the way to FINISHING. The two
    look identical in the meter and mean opposite things, because every ceiling here is a START
    GATE: it is read once, before the run, and the run then walks to completion at whatever cost.
    So the run that crosses the ceiling is normally the run that did the work. This function is
    the one place that can tell them apart, because True here is turned straight into a
    `budget_paused` return by every caller. `rail_budgets.paused_today` reads the marker; see its
    module docstring for what went wrong when it read the meter instead.

    The stamp also crosses processes for free, which the meter cannot: the four detached rails
    flush their spend at exit, so the long-lived MCP server reads 0.0 for a child's rail in
    memory. A file on disk is visible to both.

    Fail-safe, in two directions. An unreadable meter must not block a legitimate run, so it
    returns False. A marker that cannot be written must not un-refuse a run that IS over its
    ceiling, so the failure is swallowed and True still stands."""
    try:
        from pipeline import llm_client
        if llm_client.spend_today_for_rail(rail) < ceiling_usd:
            return False
    except Exception:
        return False
    try:
        stamp = refusal_marker(rail)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()                 # mtime IS the fact, as with `<slug>_last_spawn`
    except Exception:
        pass
    return True


def models_unroutable(rail: str) -> str | None:
    """model_routing's preflight at the rail spend boundary — the reason this pass must SKIP,
    or None to proceed. Call it after `load_rail_env()` (the catalog fetch needs the key).

    Blocks ONLY on `ok: False` — a registered model with zero surviving providers and no live
    fallback. `fragile` and `unknown` log and proceed: preflight must never be the reason a run
    cannot start when the answer is uncertain. Fail-safe: a preflight that throws proceeds —
    an outage in the outage-detector is not an outage."""
    try:
        from pipeline import model_routing
        from pipeline.ingestion.utils import log
        rep = model_routing.preflight()
        if rep["dead"] or rep["fragile"] or rep["unknown"]:
            log(f"[{rail}] {model_routing.format_report(rep)}")
        if not rep["ok"]:
            dead = ", ".join(f"{m} ({why})" for m, why in rep["dead"])
            return (f"unroutable under the active deny-list: {dead}. No provider survives and "
                    f"no fallback is declared — every call would 404. Edit the OpenRouter "
                    f"deny-list (or the model) and re-run; the `oracle` screen's "
                    f"`model_routing` notice shows what survives.")
        return None
    except Exception:
        return None


def spawn_rail(module: str, *, slug: str, force: bool = False,
               coalesce: float = COALESCE_DEFAULT) -> bool:
    """Fire one pass of `module` as a detached, non-blocking child. True iff a child started.

    `slug` is the rail's PUBLIC name and derives all four of its artifacts (env-var disable/
    coalesce-override/log-path plus the on-disk stamp and log file), so a rail declares it once
    rather than spelling out four strings that must agree. Not always the module name —
    `pipeline.kb.probe_catchup` spawns under `candidate_probe` — since the slug is what a human
    greps for, so it follows the log file, not the import path.

    Two orderings are load-bearing: the log fd is closed in the parent only AFTER `Popen` (the
    child already dup'd it; not closing leaks an fd per session), and the stamp is touched only
    AFTER `Popen` succeeds (stamping first would burn the whole coalesce window on a failed fork).

    Fail-safe: ANY failure returns False rather than raising.
    """
    if os.environ.get(f"OPYT_NO_{slug.upper()}") == "1":
        return False
    logf = None
    try:
        stamp = Path(os.environ.get(f"OPYT_{slug.upper()}_STAMP",
                                    opyt_path(f"{slug}_last_spawn")))
        window = float(os.environ.get(f"OPYT_{slug.upper()}_COALESCE", coalesce))
        if not force and stamp.exists() and (time.time() - stamp.stat().st_mtime) < window:
            return False
        log_path = Path(os.environ.get(f"OPYT_{slug.upper()}_LOG", opyt_path(f"{slug}.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "a")
        subprocess.Popen(
            [sys.executable, "-m", module, "--once"] + (["--force"] if force else []),
            stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,   # NEVER inherit: stdout is JSON-RPC
            start_new_session=True,                               # detach; outlive the server
            cwd=str(Path(__file__).resolve().parents[2]),         # repo root, so -m resolves
        )
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
        return True
    except Exception:
        return False
    finally:
        if logf is not None:
            logf.close()
