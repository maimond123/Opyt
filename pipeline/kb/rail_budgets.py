"""
pipeline/kb/rail_budgets.py — which paid rails are refusing runs today, and why.

Why this exists. Each rail's daily ceiling is correct on its own and INVISIBLE on its own. When
`bookmark_catchup` hits $1.00 it returns `budget_paused` into `~/.opyt/bookmark_catchup.log`, which
nothing reads — so the rail goes quiet, the corpus stops growing, and the only symptom the user
ever sees is that search results feel stale. That is the frozen-Oracle failure shape: a rail
refusing correctly, in private. A ceiling that pauses a rail invisibly is barely better than no
ceiling at all, because the expensive part was never the money — it was the days spent not knowing.

WHAT COUNTS AS PAUSED, and the 2026-08-30 correction. This module first derived the answer purely
from `api_stats.json`: over its ceiling meant paused. That is false, and measurably so. Every
ceiling here is a START GATE — `rail_budget_exhausted` is read once before a run, and the run then
walks to completion at whatever it costs (see `bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD`). So
the run that crosses the ceiling is normally the run that DID THE WORK, and "spent >= ceiling" is
not a weak proxy for "blocked" — it is anti-correlated with it. Measured: across 68 recorded runs
on a real install, `budget_paused` had never once occurred, while a brand-new user who consented to
a one-time $8.03 bookmark backlog import got told, minutes after it drained completely, that
"nothing new is being brought in ... these results may be missing recent material."

So the fact this module needs is not a dollar figure. It is whether a run was actually REFUSED, and
`rail_budget_exhausted` stamps `<rail>_budget_refused` at exactly that moment. The meter still owns
the money and still supplies `spent_usd`; the marker owns "a run was turned away". Those are two
different facts, not two homes for one — nothing has to reconcile them. mtime is the whole record
(the same convention as `<slug>_last_spawn`), and a marker left from yesterday simply stops
matching at UTC midnight, so nothing has to clean it up either.

A registry, and that word is a warning in this repo. The list below is the fifth thing that has to
learn about a new paid rail (label, ceiling, gate, decorator, this). A rail missing from it is not
broken — it still gates itself correctly — it is merely SILENT, which is the failure this module
exists to fix. `tests/kb/test_rail_spend_attribution.py` pins both the membership and the one
deliberate exclusion.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pipeline import llm_client
from pipeline.kb import rail_runtime


def _paid_rails() -> list[tuple[str, float, str]]:
    """(rail label, daily ceiling, human name) for every rail that spends money and gates on it.

    `curation_catchup` is deliberately absent. Its four collectors are free, so it has no
    ceiling and can never be paused. Adding it would make `budget_paused` mean two different
    things — "this rail spent its money" and "this rail that spends no money is off".

    Imported lazily so importing this module stays cheap for the `search` path, which reaches it
    on every call and must not pay for four rail modules to find out nothing is paused."""
    from . import bookmark_catchup, frontier_execute, hopper, oracle_refresh
    return [
        (bookmark_catchup.RAIL, bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD, "bookmark catch-up"),
        (oracle_refresh.RAIL, oracle_refresh.ORACLE_REFRESH_DAILY_USD, "Oracle refresh"),
        (frontier_execute.RAIL, frontier_execute.FRONTIER_EXECUTE_DAILY_USD, "Frontier execution"),
        (hopper.RAIL, hopper.HOPPER_DAILY_USD, "saving links with hopper"),
    ]


def _refused_today(rail: str) -> bool:
    """Did this rail's start gate actually turn a run away today? See the module docstring for why
    this is the question rather than "is it over its ceiling". UTC, matching the meter's own day
    boundary, so the pause and the spend figure beside it always describe the same day."""
    try:
        # `rail_runtime.refusal_marker`, never a from-import of it: a from-import binds the
        # function at import time, and the test harness redirects this path by patching the
        # attribute on its owning module. Same trap the `_stats_file` fixture documents.
        mtime = rail_runtime.refusal_marker(rail).stat().st_mtime
    except OSError:
        return False                  # never stamped, or no home to read — not refused
    return (datetime.fromtimestamp(mtime, timezone.utc).date()
            == datetime.now(timezone.utc).date())


def paused_today() -> list[dict]:
    """Every paid rail that REFUSED a run today because it was over its own daily ceiling.

    Empty is the normal answer, and callers gate on that — a quiet day says nothing at all. An
    unconditional block would train the reader to skip the one call where it matters, which is the
    same argument the freshness notice already makes for itself. Since the correction above, empty
    is also the answer on the far more common day when a rail spent its whole ceiling and finished:
    that rail did its job, and saying it is "paused" made a completed import look like a failure.

    `spent_usd` reads `llm_client.spend_today_by_rail()` — the CROSS-PROCESS reader — not the
    gates' own `spend_today_for_rail`. The rails run as detached children and this runs in the
    long-lived MCP server, which never sees a child's spend in memory.

    Fail-safe in the direction that matters: a hiccup reports NO pause, never a fabricated one. A
    false pause would send the user chasing a rail that is running fine — which is exactly what
    happened for as long as this read the meter."""
    try:
        refused = [(rail, ceiling, label) for rail, ceiling, label in _paid_rails()
                   if _refused_today(rail)]
        if not refused:
            return []                 # the common path: no meter read at all
        spent = llm_client.spend_today_by_rail()
        return [{"rail": rail, "label": label,
                 "spent_usd": round(spent.get(rail, 0.0), 4), "ceiling_usd": ceiling}
                for rail, ceiling, label in refused]
    except Exception:
        return []
