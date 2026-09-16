"""
pipeline/kb/rail_runtime.py — the preflight every background rail runs before it works.

A RAIL is a bounded `--once` pass the resident worker launches: `bookmark_catchup`,
`curation_catchup`, `substack_saved_catchup`, `probe_catchup`, `push_catchup`, `frontier_admit`,
`frontier_execute`, `oracle_refresh`, `sitting_scheduler` — nine. `pipeline/kb/rail_worker.py`
owns the registry of their commands and is the ONLY thing that launches them;
`pipeline/kb/rail_jobs.py` owns the durable row that says which one runs next.

This module owns neither. It holds the two things a rail child must do for ITSELF, in its own
process, before it spends anything: load the user's keys, and refuse the pass when no model it
needs is routable. Both belong here rather than in the worker because the worker supervises many
homes and each child's answer is its own.
"""

from __future__ import annotations

from pathlib import Path

from opyt_core.paths import opyt_path


def load_rail_env() -> None:
    """Load credentials into THIS process — call it first in any rail's runner. A rail child
    inherits only the environment the worker itself had, so without this it sees none of the
    user's keys. `override=True` on the user file so a rotated key on disk beats a stale copy
    cached in the parent's environment; the repo `.env` is a dev fallback and does not override.
    Fail-safe: a missing `dotenv` or unreadable file must not stop the rail."""
    try:
        from dotenv import load_dotenv
        load_dotenv(opyt_path(".env"), override=True)
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")     # dev fallback (repo root)
    except Exception:
        pass


def models_unroutable(rail: str) -> str | None:
    """model_routing's preflight at the rail request boundary — the reason this pass must SKIP,
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
