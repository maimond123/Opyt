"""
pipeline/oracles/milestones.py — where each pick stands, read from the store.

A pick is a confirmed Oracle. What it has reached is computed, never stored: the scrapers' own rows
already say it, and a stored copy would be a second thing that can disagree with them.

  | milestone      | reached when                                                              |
  |----------------|---------------------------------------------------------------------------|
  | `newest_posts` | every X pair the scrapers walk has had its first page. None: no X pair     |
  | `archives`     | discovery has looked for their other sources (its links pass, which runs  |
  |                | once X reaches its window), every source it added had its first pull, and |
  |                | every pair beyond X was pulled for the latest ask or gave up. Before       |
  |                | discovery has looked, "nothing owed" only means nothing is known yet       |
  |                | None: no ask on record                                                     |
  | `reach`        | how far back X goes (`covered_from`) against what the walk is owed         |
  |                | (`windows.target`), `reached` once no X pair owes history. None: no X       |
  | `stuck`        | a cause the user can act on, and the step that clears it                   |

`stuck`, first match wins, and only while the pick still owes something:
  • `not_running`      — nothing is claiming the work (`liveness.repair_or_report_cached`),
                         with its cause and step. Asked only when no scraper has beaten for two
                         minutes;
  • `x_session`        — X needs reconnecting;
  • `model:<service>`  — the model provider refuses, so nothing can be saved;
  • `needs_topics`     — a research venue waits for the subjects to pull (`asks.py`);
  • `discovery_failed` — finding their other sources failed ten times in a row;
  • `unsaved`          — pages keep coming back but will not save (`retries.stuck`).
An outage, X's or a web platform's, is never `stuck`: the user can do nothing about it. A web
platform's pause lives only in the scraper's memory, so a paused platform reads as "not yet".

An Oracle with no ask (confirmed before the scrapers) is owed what the walk owes any pair: six
months of X (`windows.target`). Its `newest_posts` and `reach` count; `archives` is None,
because nothing was asked of its web sources.

Since you last looked: `milestones_shown` remembers which reached milestones the user has been
told about (`told.unshown`, `told.mark_shown`), so each is said once: by `progress`, or by the rider on the
next call (`told.completion_notice`). A new ask makes `archives` (and `reach`, when it asked
for more X) owed again, and `told.reopen` lets them be said again. Only picks with an ask are announced.

`outstanding` is the one answer to "what has the user been promised that hasn't landed", for
`search` and `onboard`: these milestones, the thread tails and bookmark conversations still owed,
and the saved-posts imports begun and not landed (`curation/imports.imports_outstanding`).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime

from pipeline.curation import imports
from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.oracles import roster
from pipeline.oracles import windows
from pipeline.oracles.asks import Ask, asks_by_head
from pipeline.oracles.found import list_found
from pipeline.timeparse import parse_ts, utc_iso, utc_now
from pipeline.worker import liveness
from pipeline.worker.quota_loop import loop_state
from pipeline.worker.quota_loop import retries
from pipeline.oracles.sources.threads import state as thread_state
from pipeline.oracles.sources.timeline.walks import QUOTA as X
from pipeline.oracles.sources.web.archives import WEB_TYPES
from pipeline.oracles.sources.web.platform import QUOTA as WEB
from pipeline.oracles.sources import pair_rows


def _when(instant: str | None) -> str | None:
    """A stored ask instant as the user reads it: `all` for everything, None for nothing."""
    return "all" if instant == windows.ALL_TIME else instant


def _picks(conn) -> list[tuple[str, str | None, Ask | None]]:
    """`(head, name, ask)` for every confirmed Oracle: asked ones in ask order, the rest after."""
    asks = asks_by_head(conn)
    seen: dict[str, tuple[str, str | None, Ask | None]] = {}
    for o in roster.list_oracles(conn):
        head = schema.current_canonical(conn, o["canonical_id"])
        seen.setdefault(head, (head, o["name"], asks.get(head)))
    return sorted(seen.values(),
                  key=lambda p: p[2].position if p[2] is not None else float("inf"))


def _pick(head: str, name: str | None, ask: Ask | None, rows: list, mine: list,
          now: datetime) -> dict:
    xs = [r for r in rows if r.source_type == "x"]
    webs = [r for r in rows if r.source_type in WEB_TYPES]

    newest = all(r.last_pulled_at for r in xs) if xs else None
    reach = None
    if xs:
        covered = [t for t in (parse_ts(r.covered_from) for r in xs) if t is not None]
        reach = {"covered_from": utc_iso(min(covered)) if covered else None,
                 "asked_from": utc_iso(min(windows.target(r, ask, now) for r in xs)),
                 "reached": not any(windows.owes_history(r, ask, now) for r in xs)}
    archives = None
    if ask is not None:
        archives = (ask.discovered_at is not None and not ask.rediscover
                    and all(not windows.owed_on_ask(r, ask) or retries.given_up(r.failures)
                            for r in webs)
                    and not any(windows.waits_for_an_ask(r, ask) for r in webs)
                    and all(f.pulled_at is not None or retries.given_up(f.failures)
                            for f in mine))

    pick = {"name": name or head, "canonical_id": head, "newest_posts": newest,
            "archives": archives, "reach": reach,
            "asked": None if ask is None else {"x": _when(ask.x_from), "web": _when(ask.web_from),
                                                "papers": _when(ask.scholar_from)}}
    return pick


def _owes_x(pick: dict) -> bool:
    reach = pick["reach"]
    return pick["newest_posts"] is False or (reach is not None and not reach["reached"])


def stuck(head: str, ask: Ask | None, rows: list, mine: list, scrapers: dict,
          not_running: dict | None, *, owes_x: bool, owes_web: bool) -> dict | None:
    """Why a pick that still owes something is stuck, or None: the first cause that matches.
    `not_running` is the worker's liveness report, gathered only when no scraper has beaten
    for two minutes (None otherwise).

    | cause                                         | result                     |
    |-----------------------------------------------|----------------------------|
    | the liveness report says not running          | `not_running`, cause, step |
    | the owed loop paused on the X session         | `x_session`                |
    | the owed loop paused on the model provider    | `model:<service>`          |
    | a venue with no subjects                      | `needs_topics`             |
    | discovery given up                            | `discovery_failed`         |
    | a page that would not save                    | `unsaved`                  |
    | none                                          | None                       |
    """
    xs = [r for r in rows if r.source_type == "x"]
    webs = [r for r in rows if r.source_type in WEB_TYPES]
    if not_running is not None and not_running.get("ok") is False:
        return {"cause": "not_running", "detail": not_running.get("cause"),
                "step": not_running.get("step")}
    for quota, owed in ((X, owes_x), (WEB, owes_web)):
        s = scrapers.get(quota)
        if not owed or s is None or s["state"] != "paused":
            continue
        if s["cause"] == "x_session":
            return {"cause": "x_session", "step": "onboard(source='x')"}
        if (s["cause"] or "").startswith("model:"):
            return {"cause": s["cause"]}
    if ask is not None and any(windows.waits_for_an_ask(r, ask) and r.source_type == "openalex"
                               for r in webs):
        return {"cause": "needs_topics",
                "step": f"add_oracle('{head}') for the subject list, then "
                        f"oracle(action='ingest', canonical_ids=['{head}'], scholar_topics=[...])"}
    if (ask is not None and (ask.discovered_at is None or ask.rediscover)
            and retries.given_up(ask.discovery_failures)):
        return {"cause": "discovery_failed", "detail": ask.discovery_error,
                "step": f"oracle(action='ingest', canonical_ids=['{head}'], force=True)"}
    for item in (*xs, *webs, *mine):
        if retries.stuck(item.last_status, item.failures):
            return {"cause": "unsaved", "detail": item.last_error}
    return None


def for_picks(conn: sqlite3.Connection, canonical_ids=None, *,
              now: datetime | None = None) -> list[dict]:
    """Each pick's milestones (the module docstring), every confirmed Oracle unless
    `canonical_ids` narrows them. A store without the scrapers' tables reads as nothing yet."""
    now = now or utc_now()
    try:
        picks = _picks(conn)
        found = list_found(conn)
        scrapers = {q: loop_state.read_state(q) for q in (X, WEB)}
        walked = pair_rows.walked(conn, ("x", *WEB_TYPES))   # read once: this runs on every search
    except sqlite3.OperationalError:
        return []
    if canonical_ids is not None:
        want = {schema.current_canonical(conn, c) for c in canonical_ids}
        picks = [p for p in picks if p[0] in want]
    rows, mine = defaultdict(list), defaultdict(list)
    for r in walked:
        rows[r.canonical_id].append(r)
    for f in found:
        mine[schema.current_canonical(conn, f.canonical_id)].append(f)
    results = [_pick(head, name, ask, rows[head], mine[head], now) for head, name, ask in picks]
    owing = [(_owes_x(p), p["archives"] is False) for p in results]
    beat_at = max((s["beat_at"] for s in scrapers.values() if s is not None), default=None)
    report = None
    if any(x or web for x, web in owing) and not loop_state.is_beating(beat_at, now.timestamp()):
        report = liveness.repair_or_report_cached()
    for pick, (head, _name, ask), (owes_x, owes_web) in zip(results, picks, owing):
        if (owes_x or owes_web) and (why := stuck(head, ask, rows[head], mine[head], scrapers,
                                                  report, owes_x=owes_x,
                                                  owes_web=owes_web)) is not None:
            pick["stuck"] = why
    unfinished = defaultdict(list)
    try:
        for row in conn.execute(
                "SELECT w.canonical_id,u.status,u.failures,u.last_error FROM archive_walks w "
                "JOIN archive_units u ON u.walk=w.key AND u.generation=w.generation "
                "WHERE w.completed=0 AND u.done=0"):
            unfinished[schema.current_canonical(conn, row["canonical_id"])].append(row)
    except sqlite3.OperationalError:
        pass
    for pick in results:
        work = unfinished[pick["canonical_id"]]
        if work and pick["archives"] is not None:
            pick["archives"] = False
        if "stuck" not in pick:
            failed = next((r for r in work if retries.stuck(r["status"], r["failures"])), None)
            if failed:
                pick["stuck"] = {"cause": "unsaved", "detail": failed["last_error"]}
    return results


def owes(pick: dict) -> bool:
    """Something of this pick's is still to come."""
    reach = pick.get("reach")
    return (pick.get("newest_posts") is False or pick.get("archives") is False
            or (reach is not None and not reach["reached"]))


def reached(pick: dict) -> list[str]:
    """The milestones this pick has reached, by name; `stuck` when it is stuck."""
    out = [m for m in ("newest_posts", "archives") if pick.get(m)]
    if (pick.get("reach") or {}).get("reached"):
        out.append("reach")
    if pick.get("stuck"):
        out.append("stuck")
    return out


def waiting_on(conn: sqlite3.Connection, picks: list[dict], *,
               now: datetime | None = None) -> dict | None:
    """When X will next move, while the timeline waits for X's window or for X to answer again
    and a pick still owes X. `{source, until}`, or None. Not `stuck`: nobody can hurry it."""
    now = now or utc_now()
    if not any(p["newest_posts"] is False or (p["reach"] is not None and not p["reach"]["reached"])
               for p in picks):
        return None
    try:
        s = loop_state.read_state(X)
    except sqlite3.OperationalError:
        return None
    if (s is None or s["state"] not in ("waiting", "paused") or s["cause"] == "x_session"
            or (s["cause"] or "").startswith("model:")
            or (s["retry_at"] or 0) <= now.timestamp()):
        return None
    return {"source": "x", "until": utc_iso(datetime.fromtimestamp(s["retry_at"], now.tzinfo))}


# ── what is still owed, anywhere ──────────────────────────────────────────────────
def outstanding(conn: sqlite3.Connection) -> list[str]:
    """Short codes for what the user was promised and has not landed, in this order:
    `newest_posts`, `archives`, `reach` (some pick still owes it), `thread_tails` (Oracle threads
    still owed their later posts), `bookmark_threads` (saved posts still owed their conversation),
    then the saved-posts imports begun and not landed (`bookmarks_import`, `substack_saved_import`).
    Each part is fail-safe on its own: what cannot be read is left out, never raised."""
    out: list[str] = []
    try:
        picks = for_picks(conn)
    except Exception as exc:
        log(f"[scrapers] outstanding: milestones unreadable: {type(exc).__name__}: {exc}")
        picks = []
    if any(p["newest_posts"] is False for p in picks):
        out.append("newest_posts")
    if any(p["archives"] is False for p in picks):
        out.append("archives")
    if any(p["reach"] is not None and not p["reach"]["reached"] for p in picks):
        out.append("reach")

    def counted(code: str, count) -> None:
        try:
            if count() > 0:
                out.append(code)
        except sqlite3.OperationalError:
            pass                           # a store too young for the table: nothing owed yet
        except Exception as exc:
            log(f"[scrapers] outstanding: {code} unreadable: {type(exc).__name__}: {exc}")

    counted("thread_tails", lambda: len(thread_state.owed(conn, "tails")))
    counted("bookmark_threads", lambda: len(thread_state.owed(conn, "bookmarks")))
    try:
        out += [c for c in imports.imports_outstanding() if c not in out]
    except Exception as exc:
        log(f"[scrapers] outstanding: imports unreadable: {type(exc).__name__}: {exc}")
    return out
