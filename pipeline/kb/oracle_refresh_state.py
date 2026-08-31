"""
pipeline/kb/oracle_refresh_state.py — the freshness registry under the Oracle refresh loop.

One row per (Oracle × source), recording WHEN we last pulled that pair and how far
(`cursor_ts`). Nothing reads this on the query path; it exists only so a background loop can
answer "who has gone stale" without re-deriving it from the corpus every session.

Why a registry and not `SELECT DISTINCT source_type, source_url FROM atoms`: every adapter
writes the individual PERMALINK into `atoms.source_url` (`ingest_x_footprint`, `ingest_substack`,
`ingest_blog`, `ingest_github` all do), so that query yields hundreds of tweet URLs per Oracle and
hands the loop a list of permalinks to treat as feeds. The ROOTS already live in `entities`, typed
by prefix (`x:user:{id}` | `substack:{h}` | `blog:{host}` | `github:{owner}`), so the seed reads
entities via `schema.entities_for_canonical` — which also re-anchors a drifted cluster head.

Design invariants:
  • DERIVABLE — every field is rebuildable from `atoms` + `entities` + `oracles`. It is a cache,
    not a source of truth, so a dropped row self-heals on the next `seed_from_entities`.
  • Flat per-type TTL, no adaptive cadence — polling more often doesn't make a person post more,
    so cadence buys freshness, not savings; blog is the one source whose TTL is long because it
    pays a fixed LLM triage per refresh.
  • Breaker state is not here — it lives in the `circuit_breaker` table, keyed by service string.
  • `source_key` is ADAPTER-READY, never a permalink — the exact identifier the source's adapter
    takes (`x` → bare handle, `substack`/`blog` → the home URL, `github` → the owner login).

"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from pipeline import jitter
# Re-exported deliberately, not merely used: `oracle_refresh.py` reads it as `st.parse_ts`.
from pipeline.timeparse import parse_ts, utc_now

from . import schema

# ── Flat per-type TTLs (hours) ──────────────────────────────────────────────────
# x       — billed per tweet RETURNED, so the month's total does not move with poll frequency;
#           short is nearly free and buys the freshest channel.
# substack— a newsletter is bursty and low-frequency; the listing check itself is free.
# blog    — a FIXED discovery + LLM-triage cost per refresh even after the `known_urls` seam, so
#           this is the one type whose spend scales directly with how often we poll.
# github  — repos change slowly; ~2 API calls per refresh once the `pushed_at` gate is in.
FLAT_TTL_HOURS: dict[str, float] = {"x": 72.0, "substack": 168.0, "blog": 336.0, "github": 336.0}
DEFAULT_TTL_HOURS = 168.0          # an unknown source_type falls back to a week

# ±10% per-pair spread on the flat TTL. Every pair an `add_oracle` registers inherits the SAME
# `oracles.ingest_to`, so without this they all fall due in the same second — and the clustering
# re-forms every cycle rather than decaying, because a batch refreshed together gets stamped
# together. That is phase-locking, and at a roster large enough that one tick's due set exceeds
# what `max_pairs` can drain, every cycle then starts with a burst and a permanent backlog.
#
# Derived from the pair key, never drawn per call. A `random()` inside `is_stale` would make
# staleness nondeterministic — the exact property the repeat-run harness verifies (5 consecutive
# no-op runs, identical). Hashing the key keeps `is_stale` a pure function of stored state, so a
# pair's TTL is the same on every call, in every process, forever.
TTL_JITTER = 0.10

SUPPORTED_SOURCES: tuple[str, ...] = ("x", "substack", "blog", "github")

_DDL = """
CREATE TABLE IF NOT EXISTS oracle_sources (
  canonical_id   TEXT NOT NULL,
  source_type    TEXT NOT NULL,   -- 'x' | 'substack' | 'blog' | 'github'
  source_key     TEXT NOT NULL,   -- handle / pub url / blog url / gh owner — NOT a permalink
  status         TEXT NOT NULL DEFAULT 'trusted',
  added_at       TEXT NOT NULL DEFAULT (datetime('now')),
  last_pulled_at TEXT,            -- NULL = never refreshed → infinitely stale
  cursor_ts      TEXT,            -- MAX(when_ts) over this pair's atoms
  last_status    TEXT,
  PRIMARY KEY (canonical_id, source_type, source_key)
);
CREATE INDEX IF NOT EXISTS idx_oracle_sources_type ON oracle_sources(source_type);
"""

_GITHUB_OWNER_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/?(?:[?#]|$)",
                              re.I)


# ── time helpers ────────────────────────────────────────────────────────────────
# `parse_ts` is imported from `pipeline.timeparse` (above), not defined here. This body USED to
# be a local copy, and it had already drifted from the radar one it was copied from — it gained
# a `str()` coercion the original never got. Two "now" spellings stay local: they are one line
# each and share no failure mode with a parse.
def _now() -> str:
    # Full precision, deliberately unchanged: this stamp is already stored in
    # `collector_runs` / `oracle_sources` / `sync_dedup` at microsecond width, and
    # narrowing it would make new rows sort against old ones on a shared prefix.
    # `utc_iso()` (seconds) is the format for NEW stamps. See the audit's open
    # question on unifying stored-stamp precision.
    return utc_now().isoformat()


# ── the row ─────────────────────────────────────────────────────────────────────
@dataclass
class SourceRow:
    canonical_id: str
    source_type: str
    source_key: str
    status: str = "trusted"
    added_at: str | None = None
    last_pulled_at: str | None = None
    cursor_ts: str | None = None
    last_status: str | None = None
    name: str | None = None          # display name, joined from `oracles` — not stored here

    @property
    def pair(self) -> tuple[str, str, str]:
        return (self.canonical_id, self.source_type, self.source_key)


def _row_to_source(row: sqlite3.Row) -> SourceRow:
    keys = row.keys()
    return SourceRow(
        canonical_id=row["canonical_id"],
        source_type=row["source_type"],
        source_key=row["source_key"],
        status=row["status"],
        added_at=row["added_at"],
        last_pulled_at=row["last_pulled_at"],
        cursor_ts=row["cursor_ts"],
        last_status=row["last_status"],
        name=row["name"] if "name" in keys else None,
    )


# ── connection + schema ─────────────────────────────────────────────────────────
def init_state_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every writable open, and called by every public writer here — a
    caller may hand us a plain `schema.connect()` that has never seen this table."""
    conn.executescript(_DDL)
    conn.commit()


def connect(db_path=None, *, read_only: bool = False) -> sqlite3.Connection:
    """The atom-KB store with `oracle_sources` guaranteed present. Reuses `schema.connect`
    (WAL + busy_timeout + row_factory + `$OPYT_HOME`) and layers this table on top. Read-only
    opens skip DDL, matching `schema.connect`'s contract."""
    conn = schema.connect(db_path, read_only=read_only)
    if not read_only:
        init_state_schema(conn)
    return conn


# ── pure TTL math ───────────────────────────────────────────────────────────────
def ttl_hours(source_type: str) -> float:
    """The base TTL for a source type, before per-pair jitter. `pair_ttl_hours` is what gates."""
    return FLAT_TTL_HOURS.get(source_type, DEFAULT_TTL_HOURS)


def jitter_factor(canonical_id: str, source_type: str, source_key: str) -> float:
    """A stable multiplier in [1-TTL_JITTER, 1+TTL_JITTER], derived from the pair key alone.

    The body lives in `pipeline/jitter.py` (shared with the candidate probe); the NUL-joined pair
    key stays local since it's this table's own identity.
"""
    return jitter.stable_factor(f"{canonical_id}\x00{source_type}\x00{source_key}", TTL_JITTER)


def pair_ttl_hours(row: SourceRow) -> float:
    """This pair's effective TTL: the flat per-type base, spread by its own stable jitter."""
    return ttl_hours(row.source_type) * jitter_factor(
        row.canonical_id, row.source_type, row.source_key)


def is_stale(row: SourceRow, now: datetime | None = None) -> bool:
    """Is this pair due? Never-pulled (or an unparseable stamp) is always stale; otherwise stale
    iff the hours since the last pull meet or exceed this pair's effective TTL."""
    now = now or utc_now()
    last = parse_ts(row.last_pulled_at)
    if last is None:
        return True
    return (now - last).total_seconds() / 3600.0 >= pair_ttl_hours(row)


def staleness_hours(row: SourceRow, now: datetime | None = None) -> float:
    """How overdue a pair is, in hours PAST its TTL. Never-pulled sorts first (infinite). Used
    only to ORDER the run so an interrupted pass drains the worst backlog — `is_stale` gates."""
    now = now or utc_now()
    last = parse_ts(row.last_pulled_at)
    if last is None:
        return float("inf")
    return (now - last).total_seconds() / 3600.0 - pair_ttl_hours(row)


# ── cursor ──────────────────────────────────────────────────────────────────────
def latest_atom_ts(conn: sqlite3.Connection, source_type: str, who_ids) -> str | None:
    """`MAX(when_ts)` over this pair's atoms — the corpus-derived cursor bookmark.

    A MAX over the whole resolved cluster, not over one identifier: the atom rail keys people on
    `who_id`, and one person is several per-platform ids. The pre-atom-KB implementation this
    replaced keyed on a single actor string, which is why it was rewritten rather than ported."""
    ids = [i for i in (who_ids or []) if i]
    if not ids:
        return None
    placeholders = ", ".join("?" for _ in ids)
    row = conn.execute(
        f"SELECT MAX(when_ts) AS m FROM atoms WHERE source_type=? "
        f"AND who_id IN ({placeholders}) AND when_ts IS NOT NULL AND when_ts != ''",
        [source_type, *ids],
    ).fetchone()
    return row["m"] if row else None


# ── persistence ─────────────────────────────────────────────────────────────────
def upsert_source(conn: sqlite3.Connection, row: SourceRow) -> None:
    """Register a pair, PRESERVING any freshness already recorded for it.

    `last_pulled_at` / `cursor_ts` COALESCE onto the stored value, which is what makes
    `seed_from_entities` idempotent AND safe to re-run after every ingest: re-seeding a pair that
    the loop already refreshed must never rewind it back to the onboarding coverage marker."""
    init_state_schema(conn)
    conn.execute(
        "INSERT INTO oracle_sources "
        "(canonical_id, source_type, source_key, status, added_at, last_pulled_at, "
        " cursor_ts, last_status) "
        "VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')), ?, ?, ?) "
        "ON CONFLICT(canonical_id, source_type, source_key) DO UPDATE SET "
        "  status=excluded.status, "
        "  last_pulled_at=COALESCE(oracle_sources.last_pulled_at, excluded.last_pulled_at), "
        "  cursor_ts=COALESCE(oracle_sources.cursor_ts, excluded.cursor_ts)",
        (row.canonical_id, row.source_type, row.source_key, row.status, row.added_at,
         row.last_pulled_at, row.cursor_ts, row.last_status),
    )
    conn.commit()


def record_pull(conn: sqlite3.Connection, row: SourceRow, *, last_status: str,
                cursor_ts: str | None = None, stamp: bool = True,
                now: str | None = None) -> None:
    """Persist ONE pair's outcome.

    `stamp` is the load-bearing argument, not a convenience. A pull that SUCCEEDED — even with
    nothing new — is a real observation, so it stamps `last_pulled_at` and the flat TTL restarts
    from now. A pull that was BLOCKED (a Cloudflare shell, a provider serving an empty 200) wrote
    nothing and marked nothing seen, so it must NOT stamp: a host that stopped us is not an author
    who went quiet, and stamping would let one bad night buy a full TTL of silence."""
    init_state_schema(conn)
    conn.execute(
        "UPDATE oracle_sources SET last_status=?, "
        "  cursor_ts=COALESCE(?, cursor_ts), "
        "  last_pulled_at=CASE WHEN ? THEN ? ELSE last_pulled_at END "
        "WHERE canonical_id=? AND source_type=? AND source_key=?",
        (last_status, cursor_ts, 1 if stamp else 0, now or _now(),
         row.canonical_id, row.source_type, row.source_key),
    )
    conn.commit()
    row.last_status = last_status
    if cursor_ts:
        row.cursor_ts = cursor_ts
    if stamp:
        row.last_pulled_at = now or _now()


def list_sources(conn: sqlite3.Connection, canonical_ids=None) -> list[SourceRow]:
    """Every registered pair (optionally scoped to some Oracles), with the Oracle's display name
    joined on so a report can name a person without a second query."""
    init_state_schema(conn)
    sql = ("SELECT s.canonical_id, s.source_type, s.source_key, s.status, s.added_at, "
           "       s.last_pulled_at, s.cursor_ts, s.last_status, o.name AS name "
           "FROM oracle_sources s LEFT JOIN oracles o ON o.canonical_id = s.canonical_id")
    params: list = []
    if canonical_ids:
        ids = list(canonical_ids)
        sql += f" WHERE s.canonical_id IN ({', '.join('?' for _ in ids)})"
        params = ids
    sql += " ORDER BY s.canonical_id, s.source_type, s.source_key"
    return [_row_to_source(r) for r in conn.execute(sql, params)]


# ── seeding: entities → pairs ───────────────────────────────────────────────────
def _links(raw) -> list:
    """An entity's `identity_links` (a JSON string, a list, or a bare URL) → a list of strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, str)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw.startswith("http") else []
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, str)]
        if isinstance(parsed, str):
            return [parsed]
    return []


def _first_http(links) -> str | None:
    for u in _links(links):
        if u.startswith("http"):
            return u
    return None


def github_owners_from_links(links) -> list[str]:
    """GitHub owner logins named in an entity's `identity_links`.

    Closes a recorded gap: a GitHub node can carry an attested identity link without folding into
    the Oracle's canonical cluster, so seeding only from `github:*` members would silently leave
    those Oracles without a GitHub row."""
    out: list[str] = []
    for u in _links(links):
        m = _GITHUB_OWNER_RE.search(u)
        if m and m.group(1).lower() not in ("orgs", "settings", "about", "features"):
            out.append(m.group(1))
    return out


def _field(member, key: str):
    """Read one column from a member, whether it arrived as a `sqlite3.Row` or a plain dict.
    Rows are what `entities_for_canonical` returns; dicts are what a caller hand-builds."""
    if isinstance(member, dict):
        return member.get(key)
    return member[key] if key in member.keys() else None


def pair_from_member(member) -> tuple[str, str] | None:
    """One cluster member entity → its `(source_type, source_key)` pair, or None when the member
    carries no pullable root (an `org:`/`paper-authors:` node, or an X entity whose handle we
    never stored — the adapter pulls `from:handle`, so a bare numeric id is not enough).

    `source_key` is what the ADAPTER takes, deliberately: `sync_x_footprint(handle=)`,
    `sync_substack_footprint(publication_url=)`, `sync_blog_footprint(blog_url=)`,
    `sync_github(handles=[owner])`. Storing the entity id instead would make every dispatch
    re-derive a URL, in four places, from a shape that differs per platform."""
    eid = _field(member, "entity_id") or ""
    links = _field(member, "identity_links")

    if eid.startswith("x:user:"):
        profile = _field(member, "profile")
        try:
            parsed = json.loads(profile) if isinstance(profile, str) else (profile or {})
        except (ValueError, TypeError):
            parsed = {}
        handle = ((parsed or {}).get("handle") or "").strip().lstrip("@")
        return ("x", handle) if handle else None

    if eid.startswith("substack:"):
        url = _first_http(links)
        if not url:
            tail = eid.split("substack:", 1)[1].strip()
            if not tail or tail == "unknown":
                return None
            # `substack_entity_id` keys on the author HANDLE when it has one and on the host
            # otherwise, so a dot is the only signal telling the two apart.
            url = f"https://{tail}" if "." in tail else f"https://{tail}.substack.com"
        return ("substack", url)

    if eid.startswith("blog:"):
        url = _first_http(links)
        if not url:
            host = eid.split("blog:", 1)[1].strip()
            if not host or host == "unknown":
                return None
            url = f"https://{host}"
        return ("blog", url)

    if eid.startswith("github:"):
        owner = eid.split("github:", 1)[1].strip()
        # Entity ids are `github:{owner}`; `github:{owner}/{name}` is an ATOM id (and a `forked`
        # edge target). A slash here means we were handed the wrong kind of id — skip, never
        # register a repo as if it were a feed.
        return ("github", owner) if owner and "/" not in owner else None

    return None


def pairs_for_oracle(conn: sqlite3.Connection, canonical_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Every pullable `(source_type, source_key)` for one Oracle, plus the `who_id`s its atoms may
    carry (the cluster members, widened by any GitHub owner found only in a member's links)."""
    members = schema.entities_for_canonical(conn, canonical_id)
    pairs: list[tuple[str, str]] = []
    who_ids: list[str] = []
    for m in members:
        who_ids.append(_field(m, "entity_id"))
        p = pair_from_member(m)
        if p and p[0] in SUPPORTED_SOURCES and p[1]:
            pairs.append(p)
        for owner in github_owners_from_links(_field(m, "identity_links")):
            pairs.append(("github", owner))
            who_ids.append(f"github:{owner}")
    # De-dupe, preserving order — an X member and a blog member can both link the same GitHub.
    seen: set = set()
    deduped = [p for p in pairs if not (p in seen or seen.add(p))]
    return deduped, list(dict.fromkeys(who_ids))


def seed_from_entities(conn: sqlite3.Connection, canonical_ids=None) -> dict:
    """Register every confirmed Oracle's pullable sources. Idempotent — safe after every ingest.

    A NEW pair is seeded with:
      • `last_pulled_at` = the Oracle's `oracles.ingest_to` coverage marker, so a freshly-onboarded
        Oracle is not immediately re-pulled (the onboarding pull IS the first pull);
      • `cursor_ts`      = `MAX(when_ts)` over the pair's atoms, so even an Oracle onboarded before
        `set_oracle_window` existed (its `ingest_to` is NULL) still has a sane window to pull from.
    An EXISTING pair keeps whatever the loop has since recorded — see `upsert_source`."""
    init_state_schema(conn)
    rows = schema.list_oracles(conn)
    if canonical_ids:
        want = {schema.current_canonical(conn, c) for c in canonical_ids}
        rows = [o for o in rows if schema.current_canonical(conn, o["canonical_id"]) in want]

    seeded = 0
    for o in rows:
        cid = schema.current_canonical(conn, o["canonical_id"])
        pairs, who_ids = pairs_for_oracle(conn, cid)
        for stype, key in pairs:
            upsert_source(conn, SourceRow(
                canonical_id=cid, source_type=stype, source_key=key, status="trusted",
                last_pulled_at=o["ingest_to"],
                cursor_ts=latest_atom_ts(conn, stype, who_ids),
            ))
            seeded += 1
    return {"oracles": len(rows), "pairs": seeded}
