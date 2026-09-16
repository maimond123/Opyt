"""
pipeline/kb/oracle_refresh_state.py — the freshness registry under the Oracle refresh loop.

One row per (Oracle × source), recording WHEN we last pulled that pair and how much of the person
we hold — forward to `cursor_ts` (the newest atom) and back to `covered_from` (the oldest instant
a pull has reached). Nothing reads this on the query path; it exists so a background loop can
answer "who has gone stale" and "who is thinnest" without re-deriving either from the corpus.

This row is the ONLY record of what a pull covered. It is written by whoever performed the pull,
never inferred from an Oracle-level column — see `seed_from_entities` for the defect that rule
exists to prevent.

Why a registry and not `SELECT DISTINCT source_type, source_url FROM atoms`: every adapter
writes the individual PERMALINK into `atoms.source_url` (`ingest_x_footprint`, `ingest_substack`,
`ingest_blog`, `ingest_github` all do), so that query yields hundreds of tweet URLs per Oracle and
hands the loop a list of permalinks to treat as feeds. The ROOTS already live in `entities`, typed
by prefix (`x:user:{id}` | `substack:{h}` | `blog:{host}` | `github:{owner}`), so the seed reads
entities via `schema.entities_for_canonical` — which also re-anchors a drifted cluster head.

Design invariants:
  • DERIVABLE, with ONE exception. `cursor_ts` and the pair's identity rebuild from `atoms` +
    `entities`, so a dropped row self-heals on the next `seed_from_entities`. `last_pulled_at`
    and `covered_from` do NOT: they record an ATTEMPT, and an attempt that returned nothing
    leaves no trace in the corpus to rebuild from. A dropped row therefore self-heals into a
    re-pull, which is the safe direction, not into a coverage claim.
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

# ±10% per-pair spread on the flat TTL. Every pair one `add_oracle` stamps is stamped inside the
# same second, so without this they all fall due in the same second too — and the clustering
# re-forms every cycle rather than decaying, because a batch refreshed together gets stamped
# together. That is phase-locking, and at a roster large enough that one tick's due set exceeds
# what `max_pairs` can drain, every cycle then starts with a burst and a permanent backlog.
#
# Derived from the pair key, never drawn per call. A `random()` inside `is_stale` would make
# staleness nondeterministic — the exact property the repeat-run harness verifies (5 consecutive
# no-op runs, identical). Hashing the key keeps `is_stale` a pure function of stored state, so a
# pair's TTL is the same on every call, in every process, forever.
TTL_JITTER = 0.10

SUPPORTED_SOURCES: tuple[str, ...] = ("x", "substack", "blog", "github", "openalex")

_DDL = """
CREATE TABLE IF NOT EXISTS oracle_sources (
  canonical_id   TEXT NOT NULL,
  source_type    TEXT NOT NULL,   -- 'x' | 'substack' | 'blog' | 'github' | 'openalex'
  source_key     TEXT NOT NULL,   -- handle / pub url / blog url / gh owner / openalex author or
                                  -- source id — NOT a permalink
  status         TEXT NOT NULL DEFAULT 'trusted',
  added_at       TEXT NOT NULL DEFAULT (datetime('now')),
  last_pulled_at TEXT,            -- NULL = never refreshed → infinitely stale
  cursor_ts      TEXT,            -- MAX(when_ts) over this pair's atoms
  covered_from   TEXT,            -- oldest instant this pair has been pulled BACK to (widen-only)
  topic_filter   TEXT,            -- openalex only: '|'-joined topic ids; NULL = unfiltered
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
    covered_from: str | None = None
    # OpenAlex only. The '|'-joined topic ids the ongoing stream is narrowed to, NULL = all of it.
    # This is the ONE home of that decision: `oracle_refresh` re-pulls a scholar pair forever off
    # `source_key` alone, so a filter that lived only on the ingest call would let the stream
    # widen back to unfiltered on the next refresh, silently and within one TTL.
    topic_filter: str | None = None
    last_status: str | None = None
    name: str | None = None          # display name, joined from `oracles` — not stored here

def _row_to_source(row: sqlite3.Row) -> SourceRow:
    return SourceRow(
        canonical_id=row["canonical_id"],
        source_type=row["source_type"],
        source_key=row["source_key"],
        status=row["status"],
        added_at=row["added_at"],
        last_pulled_at=row["last_pulled_at"],
        cursor_ts=row["cursor_ts"],
        covered_from=row["covered_from"],
        topic_filter=row["topic_filter"],
        last_status=row["last_status"],
        name=row["name"],
    )


# ── connection + schema ─────────────────────────────────────────────────────────
def init_state_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every writable open, and called by every public writer here — a
    caller may hand us a plain `schema.connect()` that has never seen this table.

    `CREATE TABLE IF NOT EXISTS` does NOT add a column to a table that already exists, so a new
    column needs the explicit ALTER below. It is not a separate migration hook on purpose: every
    writer here already calls this function, so there is exactly one place a store can be behind."""
    conn.executescript(_DDL)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(oracle_sources)")}
    for col in ("covered_from", "topic_filter"):
        if col not in cols:
            conn.execute(f"ALTER TABLE oracle_sources ADD COLUMN {col} TEXT")
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

    `last_pulled_at` / `cursor_ts` / `covered_from` / `topic_filter` COALESCE onto the stored
    value, which is what makes `seed_from_entities` idempotent AND safe to re-run after every
    ingest: re-seeding a pair the loop already refreshed must never rewind what it recorded.

    `topic_filter` is in that list for a sharper reason than the others. `seed_from_entities`
    rebuilds every pair from `entities`, which carry no topic selection, so it necessarily
    re-seeds with None — and it runs after EVERY ingest. Overwriting here would erase the user's
    topic choice on the next top-up. `set_topic_filter` is the only writer that can change it,
    which is what makes changing it deliberate."""
    init_state_schema(conn)
    conn.execute(
        "INSERT INTO oracle_sources "
        "(canonical_id, source_type, source_key, status, added_at, last_pulled_at, "
        " cursor_ts, covered_from, topic_filter, last_status) "
        "VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')), ?, ?, ?, ?, ?) "
        "ON CONFLICT(canonical_id, source_type, source_key) DO UPDATE SET "
        "  status=excluded.status, "
        "  last_pulled_at=COALESCE(oracle_sources.last_pulled_at, excluded.last_pulled_at), "
        "  cursor_ts=COALESCE(oracle_sources.cursor_ts, excluded.cursor_ts), "
        "  covered_from=COALESCE(oracle_sources.covered_from, excluded.covered_from), "
        "  topic_filter=COALESCE(oracle_sources.topic_filter, excluded.topic_filter)",
        (row.canonical_id, row.source_type, row.source_key, row.status, row.added_at,
         row.last_pulled_at, row.cursor_ts, row.covered_from, row.topic_filter, row.last_status),
    )
    conn.commit()


def record_pull(conn: sqlite3.Connection, row: SourceRow, *, last_status: str,
                cursor_ts: str | None = None, covered_from: str | None = None,
                stamp: bool = True, now: str | None = None) -> None:
    """Persist ONE pair's outcome.

    `stamp` is the load-bearing argument, not a convenience. A pull that SUCCEEDED — even with
    nothing new — is a real observation, so it stamps `last_pulled_at` and the flat TTL restarts
    from now. A pull that was BLOCKED (a Cloudflare shell, a provider serving an empty 200) wrote
    nothing and marked nothing seen, so it must NOT stamp: a host that stopped us is not an author
    who went quiet, and stamping would let one bad night buy a full TTL of silence.

    `covered_from` is the BACKWARD frontier — the oldest instant this pull reached — and it only
    ever WIDENS (takes the MIN). `last_pulled_at`/`cursor_ts` answer "how current are we"; this
    answers "how far back do we go", which is the question `oracle_refresh.deepen_target` walks
    and which no other column records. Passing None means "this pull had no lower bound to
    report" and leaves the stored value alone — so a stored NULL means exactly one thing, no
    lower bound recorded, and never doubles as "unbounded".

    A caller that deliberately NARROWS a window (re-ingesting a full-archive blog with
    `web_lookback='1yr'`) therefore leaves the frontier where the wider pull put it. No caller
    does that today; if one appears, the cost is a redundant re-pull, which is the safe direction."""
    init_state_schema(conn)
    conn.execute(
        "UPDATE oracle_sources SET last_status=?, "
        "  cursor_ts=COALESCE(?, cursor_ts), "
        # Positional `?` throughout, so `covered_from` is bound three times rather than named
        # `?1`. Mixing numbered and unnumbered placeholders in one statement re-numbers every
        # later `?` off the highest index used so far, which silently shifts the whole binding.
        "  covered_from=CASE WHEN ? IS NULL THEN covered_from "
        "                    ELSE MIN(COALESCE(covered_from, ?), ?) END, "
        "  last_pulled_at=CASE WHEN ? THEN ? ELSE last_pulled_at END "
        "WHERE canonical_id=? AND source_type=? AND source_key=?",
        (last_status, cursor_ts, covered_from, covered_from, covered_from,
         1 if stamp else 0, now or _now(),
         row.canonical_id, row.source_type, row.source_key),
    )
    conn.commit()
    row.last_status = last_status
    if cursor_ts:
        row.cursor_ts = cursor_ts
    if covered_from and (row.covered_from is None or covered_from < row.covered_from):
        row.covered_from = covered_from
    if stamp:
        row.last_pulled_at = now or _now()


# An OpenAlex topic id. The shape is fixed (`T` + digits) and this regex is the ONE place a
# caller-supplied id is checked before it reaches a URL — see `set_topic_filter`.
_TOPIC_ID_RE = re.compile(r"^T\d{1,9}$")


def set_topic_filter(conn: sqlite3.Connection, canonical_id: str, source_key: str,
                     topics) -> str | None:
    """Narrow one OpenAlex pair to a set of topics, and return what is now stored.

    `topics` is a list of OpenAlex topic ids. An EMPTY list clears the filter; `None` is not a
    value this function takes, because "leave it alone" is expressed by not calling it. That split
    is load-bearing: an ordinary top-up (`oracle(action='ingest')` with no topics named) must not
    touch a stored selection, and the user needs a way back to the whole corpus.

    THE TRUST BOUNDARY for topic ids. Everything downstream — `works_filter`, the count calls, the
    refresh pull — treats the stored string as a filter fragment and concatenates it into a URL,
    so it is validated here, once, and trusted after. An id that is not `T…` is dropped rather
    than refused: the list comes from a host echoing back ids we handed it, and one garbled entry
    should narrow to the rest, not fail the whole ingest.

    Raises nothing and writes nothing for a pair that does not exist — a caller registers the pair
    first (`upsert_source`), the same order `record_pull` requires.
    """
    init_state_schema(conn)
    clean = [t for t in (str(x).strip() for x in (topics or [])) if _TOPIC_ID_RE.match(t)]
    stored = "|".join(dict.fromkeys(clean)) or None
    conn.execute(
        "UPDATE oracle_sources SET topic_filter=? "
        "WHERE canonical_id=? AND source_type='openalex' AND source_key=?",
        (stored, canonical_id, source_key))
    conn.commit()
    return stored


def topic_filter_for(conn: sqlite3.Connection, canonical_id: str,
                     source_key: str) -> str | None:
    """The stored topic filter for one OpenAlex pair, or None when it is unfiltered or unknown.

    The single read behind BOTH pulls — the first backlog and every later refresh. Neither is
    handed a topic list by its caller; both look it up here, so the backlog and the ongoing stream
    cannot disagree about what the user chose."""
    init_state_schema(conn)
    row = conn.execute(
        "SELECT topic_filter FROM oracle_sources "
        "WHERE canonical_id=? AND source_type='openalex' AND source_key=?",
        (canonical_id, source_key)).fetchone()
    return (row["topic_filter"] or None) if row else None


def list_sources(conn: sqlite3.Connection, canonical_ids=None) -> list[SourceRow]:
    """Every registered pair (optionally scoped to some Oracles), with the Oracle's display name
    joined on so a report can name a person without a second query."""
    init_state_schema(conn)
    sql = ("SELECT s.canonical_id, s.source_type, s.source_key, s.status, s.added_at, "
           "       s.last_pulled_at, s.cursor_ts, s.covered_from, s.topic_filter, "
           "       s.last_status, o.name AS name "
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


def pair_from_member(member: sqlite3.Row) -> tuple[str, str] | None:
    """One cluster member entity → its `(source_type, source_key)` pair, or None when the member
    carries no pullable root (an `org:`/`paper-authors:` node, a `scholar:` node — a Semantic
    Scholar author id has no works feed OPYT can pull — or an X entity whose handle we never
    stored, since the adapter pulls `from:handle` and a bare numeric id is not enough).

    A pair carries no topic filter: that is a USER decision and lives on the `oracle_sources` row,
    which is why it cannot be derived from an entity here.

    `source_key` is what the ADAPTER takes, deliberately: `sync_x_footprint(handle=)`,
    `sync_substack_footprint(publication_url=)`, `sync_blog_footprint(blog_url=)`,
    `sync_github(handles=[owner])`, `sync_scholar_footprint(openalex_id=)`. Storing the entity id
    instead would make every dispatch re-derive a key, in five places, from a shape that differs
    per platform."""
    eid = member["entity_id"] or ""
    links = member["identity_links"]

    if eid.startswith("x:user:"):
        profile = member["profile"]
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

    if eid.startswith("openalex:"):
        # The BARE OpenAlex id, because that is what `works_filter` takes — the same rule every
        # branch here follows: `source_key` is what the ADAPTER takes. An `A…` is an author and an
        # `S…` is a source (a journal, a preprint repository, a venue); ONE pair shape covers both
        # because the id's own prefix is what picks the `/works` field to filter on.
        openalex_id = eid.split("openalex:", 1)[1].strip()
        return ("openalex", openalex_id) if openalex_id else None

    if eid.startswith("github:"):
        owner = eid.split("github:", 1)[1].strip()
        # Entity ids are `github:{owner}`; `github:{owner}/{name}` is an ATOM id (and a `forked`
        # edge target). A slash here means we were handed the wrong kind of id — skip, never
        # register a repo as if it were a feed.
        return ("github", owner) if owner and "/" not in owner else None

    return None


def _pairs_with_entities(conn: sqlite3.Connection,
                         canonical_id: str) -> tuple[list[tuple[str, tuple[str, str]]], list[str]]:
    """The one walk of an Oracle's cluster: `[(entity_id, (source_type, source_key))]` plus every
    `who_id` its atoms may carry. `pairs_for_oracle` and `pairs_by_entity` are two views of this,
    so the GitHub-from-links widening cannot be present in one and missing from the other."""
    members = schema.entities_for_canonical(conn, canonical_id)
    found: list[tuple[str, tuple[str, str]]] = []
    who_ids: list[str] = []
    for m in members:
        eid = m["entity_id"]
        who_ids.append(eid)
        p = pair_from_member(m)
        if p and p[0] in SUPPORTED_SOURCES and p[1]:
            found.append((eid, p))
        for owner in github_owners_from_links(m["identity_links"]):
            # Keyed by the owner's OWN entity id, not the linking member's — the same GitHub can be
            # linked from an X member and a blog member, and both must resolve to one pair.
            found.append((f"github:{owner}", ("github", owner)))
            who_ids.append(f"github:{owner}")
    return found, list(dict.fromkeys(who_ids))


def _host(url: str) -> str:
    """A url's host, lowercased and stripped of `www.` — the identity of a PUBLICATION.

    `www.` is the whole reason this exists: one cluster carries `https://www.hyperdimensional.co`
    on its substack member and `https://hyperdimensional.co` on its blog member, and those are one
    publication by every measure except string equality.
    """
    m = re.match(r"^\s*(?:https?://)?([^/?#]+)", url or "", re.I)
    return re.sub(r"^www\.", "", m.group(1).lower()) if m else ""


# What a footprint pull writes. The OTHER entry modes are the reason this constant exists — see
# `_holds_atoms`.
FOOTPRINT_ENTRY_MODE = "oracle-footprint"


def _holds_atoms(conn: sqlite3.Connection, source_type: str, who_ids) -> bool:
    """Has a FOOTPRINT PULL of `source_type` ever landed atoms for this cluster?

    ⚠️ `entry_mode` IS THE WHOLE QUESTION, and leaving it out made this answer yes for the wrong
    reason (caught 2026-09-15 on a live store). The question being asked is "does this adapter
    work for this publication" — and Dean W. Ball's cluster held exactly ONE substack atom, put
    there by the SAVED-POSTS import because the user had bookmarked one of his essays. That is
    evidence about the user's reading, not about the adapter: his Substack footprint source is
    refused by the single-author eligibility gate and has never returned anything. Counting the
    saved post handed his publication to the adapter that cannot pull it.
    """
    ids = [i for i in (who_ids or []) if i]
    if not ids:
        return False
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"SELECT 1 FROM atoms WHERE source_type=? AND entry_mode=? "
        f"AND who_id IN ({placeholders}) LIMIT 1",
        [source_type, FOOTPRINT_ENTRY_MODE, *ids]).fetchone() is not None


def _one_adapter_per_publication(conn: sqlite3.Connection, pairs: list[tuple[str, str]],
                                 who_ids) -> list[tuple[str, str]]:
    """One publication, one adapter — drop the losing pair when `substack` and `blog` both cover
    the same host.

    ⚠️ THE DUPLICATE-CORPUS DEFECT, measured 2026-09-14. A Substack on a custom domain is TWO
    cluster members — `substack:{host}` minted by the Substack collector and `blog:{host}` minted
    by blog discovery — and `resolve_entities` correctly merges them into one person. Both stayed
    pullable, so both adapters walked the same publication: Dwarkesh Patel's archive landed 180
    times as `blog:` atoms AND 180 times as `substack:` atoms, same `source_url` on every pair,
    8,771 duplicate chunks and ~14MB of duplicate text in the embedding index. Nothing errors —
    the atom-id namespaces differ, so neither adapter's content hash can see the other's copy.

    ⚠️ WHICH ONE WINS IS DECIDED BY THE CORPUS, NOT BY A PREFERENCE, and THAT is the 2026-09-15
    correction. The first version of this always kept substack, on the measurement that it
    returned 0.9% more chunks and keys atoms on Substack's own post id. Then a real run produced
    the case that rule gets wrong: Dean W. Ball's `hyperdimensional.co` essays pull fine from the
    BLOG adapter (116 atoms) while his Substack is refused by the single-author eligibility gate,
    because the publication is named "Hyperdimensional" and its author is not. Preferring substack
    there retired the source that worked in favour of one that returns `skipped` forever — his
    essays were in the store with nothing registered that could ever refresh them.

    So: whichever adapter this cluster ALREADY HOLDS ATOMS FROM wins, and substack only wins the
    open case where neither has pulled yet. That rule is stable (the winner keeps winning, so the
    choice does not oscillate between passes) and it is self-correcting (a source the gate refuses
    never accumulates atoms, so it never takes the publication from one that works).
    """
    hosts = {}
    for stype, key in pairs:
        if stype in ("substack", "blog"):
            hosts.setdefault(_host(key), set()).add(stype)
    contested = {h for h, kinds in hosts.items() if {"substack", "blog"} <= kinds}
    if not contested:
        return pairs

    # One query per adapter, not per host — a cluster's atoms are keyed by who_id, not by url.
    blog_has = _holds_atoms(conn, "blog", who_ids)
    substack_has = _holds_atoms(conn, "substack", who_ids)
    loser = "substack" if (blog_has and not substack_has) else "blog"
    return [(stype, key) for stype, key in pairs
            if not (stype == loser and _host(key) in contested)]


def pairs_for_oracle(conn: sqlite3.Connection, canonical_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Every pullable `(source_type, source_key)` for one Oracle, plus the `who_id`s its atoms may
    carry (the cluster members, widened by any GitHub owner found only in a member's links).

    ONE ADAPTER PER PUBLICATION — see `_one_adapter_per_publication` for the corpus this doubled
    before the collapse existed."""
    found, who_ids = _pairs_with_entities(conn, canonical_id)
    # De-dupe, preserving order — an X member and a blog member can both link the same GitHub.
    seen: set = set()
    deduped = [p for _eid, p in found if not (p in seen or seen.add(p))]
    return _one_adapter_per_publication(conn, deduped, who_ids), who_ids


def pairs_by_entity(conn: sqlite3.Connection, canonical_id: str) -> dict[str, tuple[str, str]]:
    """`entity_id -> (source_type, source_key)` for one Oracle's registered pairs.

    The join a CALLER needs to attribute an adapter run to the row that records it. An ingest
    reports its outcomes per URL; `derive.blog_entity_id` / `derive.substack_entity_id` /
    `x:user:{rest_id}` turn a URL back into the same entity id this walk keys on, so the two sides
    meet on a DERIVED id rather than on two spellings of a URL that only usually agree."""
    found, _who_ids = _pairs_with_entities(conn, canonical_id)
    return dict(found)


def seed_from_entities(conn: sqlite3.Connection, canonical_ids=None, *, include_x: bool = True) -> dict:
    """Register every confirmed Oracle's enabled pullable sources. Idempotent after every ingest.

    `include_x` is false while OPYT has no managed X session. Discovery can still identify an
    Oracle's X profile, but that identity is not permission to schedule a session-backed pull.
    Existing X rows are intentionally preserved: they are the record of a past, connected pull
    and return to the registry when the user reconnects.

    Seeding registers ROWS. It does NOT claim coverage: a new pair gets `last_pulled_at = NULL`
    and only `cursor_ts` = `MAX(when_ts)` over the pair's atoms, which is corpus-derived and
    therefore cannot over-claim. An EXISTING pair keeps whatever the loop has since recorded —
    see `upsert_source`.

    ⚠️ `last_pulled_at` USED to be seeded from the Oracle's `oracles.ingest_to` marker, and that
    was the defect this file's registry was supposed to prevent. `ingest_to` was written
    unconditionally at the end of an onboarding ingest, including on a run whose X pull raised, so
    a person with zero X atoms got a row claiming a fresh X pull — and `upsert_source`'s COALESCE
    then made that claim permanent. The pull that actually happened is recorded by whoever
    performed it, via `record_pull`; nothing infers it from an Oracle-level column any more.
    An unstamped pair is re-pulled a little early, which dedup absorbs; the reverse mistake loses
    content silently and forever."""
    init_state_schema(conn)
    rows = schema.list_oracles(conn)
    if canonical_ids:
        want = {schema.current_canonical(conn, c) for c in canonical_ids}
        rows = [o for o in rows if schema.current_canonical(conn, o["canonical_id"]) in want]

    seeded = retired = adopted = 0
    for o in rows:
        cid = schema.current_canonical(conn, o["canonical_id"])
        pairs, who_ids = pairs_for_oracle(conn, cid)
        for stype, key in pairs:
            if stype == "x" and not include_x:
                continue
            upsert_source(conn, SourceRow(
                canonical_id=cid, source_type=stype, source_key=key, status="trusted",
                cursor_ts=latest_atom_ts(conn, stype, who_ids),
            ))
            seeded += 1
        adopted += adopt_orphaned_sources(conn, cid)
        retired += retire_superseded(conn, cid, pairs)
    return {"oracles": len(rows), "pairs": seeded, "retired": retired, "adopted": adopted}


def adopt_orphaned_sources(conn: sqlite3.Connection, canonical_id: str) -> int:
    """Re-point rows stranded under a PRE-MERGE id of this same Oracle, preserving pull history.

    ⚠️ THE SAME PAIR, REGISTERED TWICE, PULLED TWICE. `seed_from_entities` registers under the
    CURRENT head; a row written before a footprint merge moved that head keeps the old id, and
    nothing reconciles them. `refresh_all` reads every row in the table with no join to `oracles`,
    so both get walked each cycle. Measured on a live store: `substack:www.dwarkesh.com` and
    `blog:dwarkesh.com` each carried `substack https://www.dwarkesh.com` and `x dwarkesh_sp` — one
    publication and one timeline, four rows, double the requests against the API whose rate limit
    is already what defers this user's backlog.

    HISTORY WINS OVER RECENCY. When both ids carry the pair, the surviving row is the one with the
    later `last_pulled_at`: an orphan is usually the one that DID the pulling (it predates the
    merge) while the head's row was seeded fresh with NULL. Keeping the head's empty row would
    re-walk an archive already held — which dedup absorbs, but only after paying for the walk.
    """
    init_state_schema(conn)
    adopted = 0
    for row in conn.execute("SELECT * FROM oracle_sources WHERE canonical_id != ?",
                            (canonical_id,)).fetchall():
        if schema.current_canonical(conn, row["canonical_id"]) != canonical_id:
            continue
        here = conn.execute(
            "SELECT last_pulled_at FROM oracle_sources "
            "WHERE canonical_id=? AND source_type=? AND source_key=?",
            (canonical_id, row["source_type"], row["source_key"])).fetchone()
        if here is None:
            conn.execute("UPDATE oracle_sources SET canonical_id=? "
                         "WHERE canonical_id=? AND source_type=? AND source_key=?",
                         (canonical_id, row["canonical_id"], row["source_type"],
                          row["source_key"]))
        else:
            if (row["last_pulled_at"] or "") > (here["last_pulled_at"] or ""):
                conn.execute(
                    "UPDATE oracle_sources SET last_pulled_at=?, cursor_ts=?, covered_from=?, "
                    "last_status=? WHERE canonical_id=? AND source_type=? AND source_key=?",
                    (row["last_pulled_at"], row["cursor_ts"], row["covered_from"],
                     row["last_status"], canonical_id, row["source_type"], row["source_key"]))
            conn.execute("DELETE FROM oracle_sources "
                         "WHERE canonical_id=? AND source_type=? AND source_key=?",
                         (row["canonical_id"], row["source_type"], row["source_key"]))
        adopted += 1
    if adopted:
        conn.commit()
    return adopted


def retire_superseded(conn: sqlite3.Connection, canonical_id: str,
                      pairs: list[tuple[str, str]]) -> int:
    """Delete this Oracle's registered website rows that `pairs_for_oracle` no longer produces,
    for the ONE reason it stops producing one: a `substack`/`blog` pair lost the contest for a
    host the other one also covers (`_one_adapter_per_publication`).

    ⚠️ REGISTRATION IS NOT THE LOOP'S INPUT — `refresh_all` reads `oracle_sources` rows, not
    `pairs_for_oracle`. So collapsing the pair stops a NEW duplicate and does nothing about an
    existing one: every store onboarded before the collapse keeps its losing row and keeps
    re-walking a publication the winner already covers, forever. Dwarkesh Patel's archive was 180
    atoms deep on both adapters when this was found.

    NARROW ON PURPOSE, and narrower than "any row not in `pairs`" — which would be wrong twice
    over: `seed_from_entities` deliberately preserves X rows while no session is connected (they
    are the record of a past pull), and a pair that momentarily fails to derive would drop a
    source along with its whole pull history. Only a CONTESTED host qualifies, meaning one where
    both adapters are registered and the winner is therefore already covering it.

    The atoms already written are NOT deleted here. Removing content is a user's decision and
    belongs to `forget`, not to a registry pass that runs unattended before every refresh.
    """
    live = {(stype, _host(key)) for stype, key in pairs if stype in ("substack", "blog")}
    rows = conn.execute(
        "SELECT source_type, source_key FROM oracle_sources "
        "WHERE canonical_id=? AND source_type IN ('substack','blog')",
        (canonical_id,)).fetchall()

    registered_kinds: dict = {}
    for r in rows:
        registered_kinds.setdefault(_host(r["source_key"]), set()).add(r["source_type"])

    retired = 0
    for r in rows:
        host = _host(r["source_key"])
        contested = {"substack", "blog"} <= (registered_kinds.get(host) or set())
        if not contested or (r["source_type"], host) in live:
            continue
        conn.execute("DELETE FROM oracle_sources "
                     "WHERE canonical_id=? AND source_type=? AND source_key=?",
                     (canonical_id, r["source_type"], r["source_key"]))
        retired += 1
    if retired:
        conn.commit()
    return retired
