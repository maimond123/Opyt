"""service/db.py — opens `service.db` with its one schema applied.

`service.db` holds who may call this service and how often; kept separate from `opyt.db` (the
peers registry) so token material never travels with a knowledge-base store. Thirteen tables,
each written by one module: `tokens`, `owner_claims` (`tokens.py`), `owner_uploads`
(`publications.py`), `grant_codes`, `agent_grants` (`grants.py`), `usage_daily` (`usage.py`),
`accounts`, `account_links` (`accounts.py`), `account_peers` (`account_peers.py`), `pages`,
`page_comments` (`pages/store.py`), `line_calls`, `line_seen` (`line.py`). Deleting a library removes its rows from all of them at
once (`deletion.py`).

A new table appears by itself on the next open; a new column on an existing table never does
(`CREATE TABLE IF NOT EXISTS`), so it is listed in `_ADDED` too. Every caller opens and closes its own
connection: a `sqlite3` connection is bound to the thread that made it.
"""

from __future__ import annotations

import sqlite3

from opyt_core.paths import opyt_path

_DDL = """
CREATE TABLE IF NOT EXISTS tokens (
  token_sha256 TEXT PRIMARY KEY,   -- never the token itself: the DB holds no usable credential
  owner        TEXT NOT NULL,      -- the ONE knowledge base this token can reach
  role         TEXT NOT NULL,      -- 'owner' (upload/grant/revoke) | 'reader' (query) |
                                   -- 'agent' (query, and act as the owner on the web tools)
  label        TEXT,               -- shown in the owner's revoke list
  install_id   TEXT,               -- the reader's stable per-install id. No accounts.
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tokens_owner ON tokens(owner);

-- A routing key is claimed until its library is deleted.
-- Nothing in `tokens` stops two owner tokens sharing a key, and the served file is
-- `exports/<owner>.db` — so a second token for one key would silently replace that knowledge
-- base, and every reader's saved peer row would start answering with somebody else's atoms, with
-- no error at any layer. Releasing a key when its last token is revoked reopens the same hole one
-- step later, so the claim outlives the token. Deleting the whole library (`deletion.py`) is the
-- one release: every token for the key goes in the same step, so no saved peer row can reach
-- whoever is assigned the key next.
--
-- The claim is not a one-way door: `register_owner` assigns `secrets.token_hex(6)`, which nobody
-- wants and nobody competes for, and `NameClaimed` is that function's collision signal rather
-- than a refusal a person meets.
CREATE TABLE IF NOT EXISTS owner_claims (
  owner        TEXT PRIMARY KEY,
  token_sha256 TEXT NOT NULL,     -- the one token allowed to publish under this name
  claimed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
-- The claims seed for owner tokens that predate this table is `_CLAIM_SEED`, below.

-- How much disk each knowledge base costs, and since when. A table and not two columns on
-- `owner_claims`: `_DDL` runs `CREATE TABLE IF NOT EXISTS` on every connect,
-- so a column added to an existing table silently never appears.
--
-- Recorded from the first upload because neither fact can be backfilled. `first_published_at` is
-- gone the moment it is not written, and `bytes` after the fact is a directory listing that says
-- nothing about who published when. Publishing is self-service, so nothing
-- vets owners: "see who is eating the disk" has to be a query, and it is what the removal
-- workflow reads. They are also the prerequisites for pricing.
CREATE TABLE IF NOT EXISTS owner_uploads (
  owner              TEXT PRIMARY KEY,
  bytes              INTEGER NOT NULL DEFAULT 0,   -- the served export's size; 0 = unpublished
  -- Demand term: total reads at the moment of that upload. A watermark on a monotonic
  -- counter, which is what makes the push gate genuinely self-quieting — a read makes demand
  -- true, the push consumes it by moving this, and it goes false again. `usage_daily` is
  -- day-granular by construction (the timestamp is not a column, deliberately), so comparing
  -- against a DATE instead would leave demand true for the rest of the calendar day and re-push
  -- on every session that ingested anything.
  reads_at_upload    INTEGER NOT NULL DEFAULT 0,
  first_published_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_published_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS grant_codes (
  code_sha256 TEXT PRIMARY KEY,
  owner       TEXT NOT NULL,
  label       TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  redeemed_at TEXT                 -- non-NULL = dead. one-time, decided by a row count.
);

-- WHICH INVITES BUY AN AGENT KEY. A row here makes its `grant_codes` row an agent invite: the
-- code redeems into an `agent` token, not a `reader` one. A table and not a column on
-- `grant_codes`, per the rule above. `owner` is here so deleting a library finds its rows.
CREATE TABLE IF NOT EXISTS agent_grants (
  code_sha256 TEXT PRIMARY KEY,
  owner       TEXT NOT NULL
);

-- Daily counts, not an event log. The intrusive part of a request log is the JOIN — who read
-- from whom and when, at full resolution — not the facts. One row per (day, owner, reader,
-- tool) keeps every metric the stats page reads and drops the trace. The query is not a column
-- and must never become one.
CREATE TABLE IF NOT EXISTS usage_daily (
  day          TEXT NOT NULL,          -- YYYY-MM-DD (UTC), deliberately not a timestamp
  owner        TEXT NOT NULL,
  reader       TEXT NOT NULL,          -- the token HASH; resolves to a label only via tokens
  tool         TEXT NOT NULL,          -- search | open | aggregate
  n            INTEGER NOT NULL DEFAULT 0,
  zero_results INTEGER NOT NULL DEFAULT 0,   -- searches that returned nothing; 0 for other tools
  PRIMARY KEY (day, owner, reader, tool)
);

-- WHICH LIBRARY A GOOGLE ACCOUNT READS through the connector. `google_sub` is Google's permanent
-- id for the account, the only claim kept: no email, no name. Written when a builder holding the
-- library's owner token completes a Google sign-in, so both halves are proven. A later link from
-- the same account replaces the row.
CREATE TABLE IF NOT EXISTS accounts (
  google_sub TEXT PRIMARY KEY,
  owner      TEXT NOT NULL,
  linked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- THE LIBRARIES SHARED WITH A GOOGLE ACCOUNT, read through the connector (`account_peers.py`).
-- A row is one invite the account redeemed. `token_sha256` is the reader token the invite
-- bought, so the owner sees and revokes this reader like any other; the token itself is not
-- kept. A row whose token is gone has been cut off.
CREATE TABLE IF NOT EXISTS account_peers (
  google_sub   TEXT NOT NULL,
  name         TEXT NOT NULL,       -- what the caller passes as kb=
  owner        TEXT NOT NULL,
  token_sha256 TEXT NOT NULL,
  added_at     TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (google_sub, name)
);

-- A GOOGLE SIGN-IN IN FLIGHT, one per library, deleted when it ends. `device_code` is Google's
-- and useless without this service's client secret; it lives here, not on the builder, so a
-- builder that restarts between the two calls still finishes.
CREATE TABLE IF NOT EXISTS account_links (
  owner            TEXT PRIMARY KEY,
  device_code      TEXT NOT NULL,
  user_code        TEXT NOT NULL,
  verification_url TEXT NOT NULL,
  expires_at       REAL NOT NULL      -- unix seconds, on Google's clock
);

-- A PAGE OF SOURCES, readable by anyone holding its link until it expires (`pages/store.py`:
-- so many days after it was last opened). The id is the link's only secret. `items` is what the page is drawn from, per request, so a design change
-- reaches pages already made; a paper's fetched PDF sits beside it on disk (`pages/<id>/`).
CREATE TABLE IF NOT EXISTS pages (
  id      TEXT PRIMARY KEY,   -- 128 random bits
  owner   TEXT NOT NULL,      -- the library that made it; only its owner token deletes it
  title   TEXT NOT NULL,
  note    TEXT,
  answer  TEXT,               -- markdown: the host's answer, drawn above the sources
  items   TEXT NOT NULL,      -- JSON list, one object per source, as `pages/render.py` takes it
  made    REAL NOT NULL,      -- unix seconds
  expires REAL NOT NULL       -- unix seconds; pushed out each time the page is opened
);
CREATE INDEX IF NOT EXISTS idx_pages_expires ON pages(expires);

-- A COMMENT on a page of sources, written by anyone holding the page's link, and deleted with its
-- page. The comment that starts a thread says where it is attached: `item`, and for a passage its
-- `quote` with the text around it, which is how the page finds the passage again. A reply names
-- its `thread`. Only `key_sha256`'s key, which the poster's browser holds, deletes it sooner.
CREATE TABLE IF NOT EXISTS page_comments (
  id         TEXT PRIMARY KEY,   -- random
  page_id    TEXT NOT NULL,
  thread     TEXT NOT NULL,      -- the id of the comment that started the thread: its own, for that one
  item       INTEGER,            -- which source on the page, counted from 1; NULL on a reply
  quote      TEXT,               -- the passage commented on; NULL for a whole source, and on a reply
  prefix     TEXT,               -- the source's text just before the passage
  suffix     TEXT,               -- and just after it
  name       TEXT NOT NULL,      -- as the commenter typed it
  body       TEXT NOT NULL,
  made       REAL NOT NULL,      -- unix seconds
  key_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_page_comments_page ON page_comments(page_id, made);

-- A TOOL CALL WAITING FOR THE OWNER'S MAC. A web tool that changes the library cannot write the
-- served copy, so it leaves the local tool's arguments here; the Mac takes them over the line,
-- runs the tool and posts its answer (`line.py`). `args` holds only what the tool needs: a link,
-- a topic, an item id. Deleted a week after `asked`, whatever its state.
CREATE TABLE IF NOT EXISTS line_calls (
  id           TEXT PRIMARY KEY,   -- 128 random bits
  owner        TEXT NOT NULL,      -- the library whose Mac runs it
  tool         TEXT NOT NULL,
  args         TEXT NOT NULL,      -- JSON object, the local tool's own arguments
  state        TEXT NOT NULL,      -- waiting | running | done | failed; `line.py` lists the moves
  answer       TEXT,               -- JSON object, the Mac's own answer once done or failed
  asked        REAL NOT NULL,      -- unix seconds
  waited_until REAL NOT NULL,      -- unix seconds: its asker waits for the answer until then
  taken        REAL,               -- unix seconds: when the Mac last took it
  told         INTEGER NOT NULL DEFAULT 0   -- 1 once a caller has been given how it ended
);
CREATE INDEX IF NOT EXISTS idx_line_calls_owner ON line_calls(owner, asked);

-- WHEN EACH LIBRARY'S MAC LAST ASKED FOR A CALL. While that is recent the line is held, and a web
-- tool waits for the Mac's answer instead of leaving the call for later.
CREATE TABLE IF NOT EXISTS line_seen (
  owner TEXT PRIMARY KEY,
  seen  REAL NOT NULL              -- unix seconds
);
"""


# Every owner token's name gets a claim, oldest token first; checked on every connect, but it
# writes only when `_UNCLAIMED` finds one missing, since an INSERT takes the write lock regardless.
_UNCLAIMED = ("SELECT 1 FROM tokens t WHERE t.role = 'owner' AND NOT EXISTS "
              "(SELECT 1 FROM owner_claims c WHERE c.owner = t.owner) LIMIT 1")
_CLAIM_SEED = ("INSERT OR IGNORE INTO owner_claims (owner, token_sha256) "
               "SELECT owner, token_sha256 FROM tokens WHERE role = 'owner' ORDER BY created_at")


# Columns added to a table after it shipped: `_DDL` holds them for a new database, and these add
# them to one made before.
_ADDED = (("pages", "answer", "TEXT"),)


def _add_columns(conn: sqlite3.Connection) -> None:
    for table, column, kind in _ADDED:
        have = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")


def db_path():
    """Where `service.db` lives — beside the peers registry, under the service's own home."""
    return opyt_path("service.db")


def connect() -> sqlite3.Connection:
    """An open `service.db` with the DDL applied, owned by the calling thread."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)          # waits up to 5 s for a lock rather than failing at once
    conn.row_factory = sqlite3.Row
    # WAL, so a read never waits on a writer. Once set (it lives in the file) this takes no lock.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)              # CREATE … IF NOT EXISTS takes no lock once tables exist
    _add_columns(conn)
    if conn.execute(_UNCLAIMED).fetchone() is not None:
        conn.execute(_CLAIM_SEED)
    conn.commit()
    return conn
