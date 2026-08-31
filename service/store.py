"""service/store.py — `service.db`: who may call this service, and how often they did.

One sentence, one module. This owns credentials and audit. It is a SEPARATE file from
`opyt.db` on purpose: `opyt.db` is the service's peers registry — the record of whose knowledge
base is served and where its export file sits — and a knowledge-base store that also holds
bearer-token material is one backup or one accidental `build_export` away from leaking it. Two
files, two sentences, no overlap.

Four tables and nothing else:

  • `tokens` — a credential's HASH, never the credential. A row is what makes a request
    possible, so REVOCATION IS A ROW DELETE and takes effect on the very next request. There is
    no refresh cycle, no expiry to tune, and no window during which a revoked reader still works,
    because this service keeps state and a token is a row in it.
  • `owner_claims` — one row per published NAME, held forever. The name is an address: it is the
    served file (`exports/<owner>.db`) and the `kb=` every reader saved, so it maps to exactly
    one publishing token and is never released — not even when that token is revoked, because a
    re-claimed name would answer those readers with somebody else's atoms. Rotation repoints the
    claim (`mint_token(..., reclaim=True)`); nothing deletes it.
  • `grant_codes` — one-time exchange codes. The owner mints one and sends it however they like;
    it buys exactly one reader token and then it is dead. What crosses a chat window is therefore
    not a standing credential.
  • `usage_daily` — one row per (day, owner, reader, tool), counting reads. NOT an event log:
    the intrusive part of a request log is the join — who read from whom and when, at full
    resolution — not the facts, so the day is the finest time this file records and a second
    read on the same day increments a counter instead of appending a row. THE QUERY IS NOT A
    COLUMN AND MUST NEVER BECOME ONE (R10). See `service/app.py` for what "blind" does and does
    not mean, and `TELEMETRY.md` for the published version of this paragraph.

Two roles, `owner` and `reader`, because both have a live caller on day one — the owner uploads
and grants, the reader queries. This is not a permissions model: there is no roles table, no
scopes, and no sharing pane. A third role would need a third caller first.

ADDING A TABLE HERE IS FREE; ADDING A COLUMN IS NOT, AND FAILS SILENTLY. `_DDL` is
`CREATE TABLE IF NOT EXISTS` run by `connect()` on every open, so a new TABLE appears by itself —
but SQLite skips the whole statement for one that already exists, so a new COLUMN never appears
and the next query dies on "no such column". Copy `pipeline/kb/schema.py:_ensure_column` when the
first column is actually needed; do not add the helper before then.

EVERY FUNCTION OPENS AND CLOSES ITS OWN CONNECTION, exactly as `pipeline/kb/peers.py` does. The
first version took a `conn` parameter and the server supplied one through a FastAPI yield
dependency — which creates it in one worker thread and runs the handler in another, and
`sqlite3` refuses a connection used off the thread that made it. Measured: fine at any sequential
rate, because the threadpool reuses one thread, and a 500 on the first genuinely concurrent
request. Owning the connection here removes the handoff instead of permitting it with
`check_same_thread=False`, which would have kept the sharing and silenced the warning about it.

Design record: docs/plans/2026-08-26-foreign-kb-service-phase3.md §3.2 / §3.4 / §3.6.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import string

from opyt_core.paths import opyt_path

# A URL-safe 256-bit secret. Long enough that guessing is not a threat model, so nothing here
# rate-limits an auth attempt — the counting path ships (`usage_daily`), the enforcement waits
# for a stranger to actually have access.
_TOKEN_BYTES = 32

# A GRANT CODE IS THE ONE CREDENTIAL A PERSON RETYPES INTO A SHELL — `opyt-redeem <url> <code>` —
# so its alphabet is letters and digits only, where a token's is `secrets.token_urlsafe`'s
# `-`/`_` included. That is not cosmetic: `token_urlsafe` starts a code with `-` about one time in
# 64, and argparse reads a leading `-` as an option, so roughly 1.6% of codes were unusable and
# answered with a usage error naming the wrong problem. Found as a 1-in-64 test flake, 2026-08-27.
# 43 characters of this alphabet is ~256 bits, the same as the tokens.
_CODE_ALPHABET = string.ascii_letters + string.digits
_CODE_LEN = 43

ROLES = ("owner", "reader")

_DDL = """
CREATE TABLE IF NOT EXISTS tokens (
  token_sha256 TEXT PRIMARY KEY,   -- never the token itself: the DB holds no usable credential
  owner        TEXT NOT NULL,      -- the ONE knowledge base this token can reach
  role         TEXT NOT NULL,      -- 'owner' (upload/grant/revoke) | 'reader' (query)
  label        TEXT,               -- shown in the owner's revoke list
  install_id   TEXT,               -- R7: the reader's stable per-install id. No accounts.
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tokens_owner ON tokens(owner);

-- A NAME IS CLAIMED FOREVER (2026-08-28). Nothing in `tokens` stops two owner tokens sharing a
-- name, and the served file is `exports/<owner>.db` — so a second 'dave' would silently replace
-- the first dave's knowledge base, and every reader's saved peer row would start answering with
-- the second dave's atoms, with no error at any layer. Releasing a name when its last token is
-- revoked reopens the same hole one step later, so the claim outlives the token.
CREATE TABLE IF NOT EXISTS owner_claims (
  owner        TEXT PRIMARY KEY,
  token_sha256 TEXT NOT NULL,     -- the one token allowed to publish under this name
  claimed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
-- The migration for a database whose owner tokens predate the table: each claims its own name,
-- oldest token first. Re-run on every connect, it also restores a hand-deleted claim, so "every
-- owner token's name is claimed" is structural rather than operational.
INSERT OR IGNORE INTO owner_claims (owner, token_sha256)
  SELECT owner, token_sha256 FROM tokens WHERE role = 'owner' ORDER BY created_at;

CREATE TABLE IF NOT EXISTS grant_codes (
  code_sha256 TEXT PRIMARY KEY,
  owner       TEXT NOT NULL,
  label       TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  redeemed_at TEXT                 -- non-NULL = dead. R5: one-time, decided by a row count.
);

-- The tombstone for the event log this replaced (2026-08-27). It never held a real row — the
-- service had not been deployed — so this drops it from dev databases and stops it coming back.
DROP TABLE IF EXISTS exchanges;

-- Daily counts, not an event log. The intrusive part of a request log is the JOIN — who read
-- from whom and when, at full resolution — not the facts. One row per (day, owner, reader,
-- tool) keeps every metric the stats page reads and drops the trace. THE QUERY IS NOT A COLUMN
-- AND MUST NEVER BECOME ONE (R10).
CREATE TABLE IF NOT EXISTS usage_daily (
  day          TEXT NOT NULL,          -- YYYY-MM-DD (UTC), deliberately not a timestamp
  owner        TEXT NOT NULL,
  reader       TEXT NOT NULL,          -- the token HASH; resolves to a label only via tokens
  tool         TEXT NOT NULL,          -- search | open | aggregate
  n            INTEGER NOT NULL DEFAULT 0,
  zero_results INTEGER NOT NULL DEFAULT 0,   -- searches that returned nothing; 0 for other tools
  PRIMARY KEY (day, owner, reader, tool)
);
"""


def db_path():
    """Where `service.db` lives — beside the peers registry, under the service's own home."""
    return opyt_path("service.db")


def connect() -> sqlite3.Connection:
    """An open `service.db` with the DDL applied, owned by the CALLING thread.

    Public because tests and an operator standing the service up mint the first owner token with
    it. Request handling never holds one across a call — see the module docstring."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


def token_hash(token: str) -> str:
    """The one hashing rule, stated once. Every read and write of a credential goes through it,
    so "the database holds no usable credential" is a property of this function rather than of
    each call site remembering."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── tokens ───────────────────────────────────────────────────────────────────────

class NameClaimed(RuntimeError):
    """An owner name that is already published under. One exception for every refusal shape —
    already claimed, or reclaiming past a live token — because the caller's move is the same
    either way: revoke first, or pick another name."""


def _claim_name(conn: sqlite3.Connection, owner: str, token_sha256: str, *, reclaim: bool) -> None:
    """Take or repoint the name's claim, inside the mint's own transaction.

    The PRIMARY KEY decides the fresh-claim race — the same discipline as `redeem_grant`'s row
    count: two simultaneous mints both INSERT and the loser gets the constraint, not a window.
    A reclaim additionally requires the live token to be gone, so a claim can never move while
    its holder still works; rotating a leaked owner token is revoke, then `reclaim=True`."""
    if not reclaim:
        try:
            conn.execute("INSERT INTO owner_claims (owner, token_sha256) VALUES (?, ?)",
                         (owner, token_sha256))
        except sqlite3.IntegrityError:
            raise NameClaimed(
                f"'{owner}' is already claimed, and a name is never released — it is the address "
                f"in every reader's peer row. To rotate this owner's token: revoke it, then mint "
                f"again with reclaim=True.") from None
        return
    live = conn.execute("SELECT 1 FROM tokens WHERE owner = ? AND role = 'owner' LIMIT 1",
                        (owner,)).fetchone()
    if live is not None:
        raise NameClaimed(f"'{owner}' still has a live owner token — revoke it before reclaiming.")
    conn.execute("INSERT INTO owner_claims (owner, token_sha256) VALUES (?, ?) "
                 "ON CONFLICT(owner) DO UPDATE SET token_sha256 = excluded.token_sha256",
                 (owner, token_sha256))


def mint_token(owner: str, role: str, *, label: str | None = None,
               install_id: str | None = None, reclaim: bool = False) -> str:
    """Create a token for `owner` and return it ONCE, in the clear. Only its hash is stored.

    The clear text is returned rather than stored because there is no second occasion to hand it
    over: a service that could re-show a token is a service that holds one.

    An OWNER mint also takes the name's claim, and refuses (`NameClaimed`) when the name is
    already held — including by a token that was since revoked. `reclaim=True` is the rotation
    path: only valid once no live owner token remains, it moves the claim to the new token while
    the readers' grants survive."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    conn = connect()
    try:
        if role == "owner":
            _claim_name(conn, owner, token_hash(token), reclaim=reclaim)
        conn.execute(
            "INSERT INTO tokens (token_sha256, owner, role, label, install_id) VALUES (?,?,?,?,?)",
            (token_hash(token), owner, role, label, install_id),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def claim_holder(owner: str) -> str | None:
    """The hash of the one token allowed to publish under this name — what the upload boundary
    compares against. None only for a name no owner token has ever been minted for; the seed in
    `_DDL` claims every pre-existing one on connect."""
    conn = connect()
    try:
        row = conn.execute("SELECT token_sha256 FROM owner_claims WHERE owner = ?",
                           (owner,)).fetchone()
    finally:
        conn.close()
    return row["token_sha256"] if row is not None else None


def resolve_token(token: str) -> dict | None:
    """The row a bearer token names, or None. The `owner` column IS the scope — a reader token
    can only reach the knowledge base it was minted against, so there is nothing else to check.

    Returns a plain dict, not a `sqlite3.Row`: a Row is tied to the connection that produced it,
    and this one is closed before the caller sees the result."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT token_sha256, owner, role, label, install_id, created_at FROM tokens "
            "WHERE token_sha256 = ?", (token_hash(token),)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def list_tokens(owner: str) -> list[dict]:
    """Every token issued for this knowledge base, oldest first — the owner's revoke list.
    Carries the hash, because that is the handle `revoke` takes and the only one that exists."""
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT token_sha256, role, label, install_id, created_at FROM tokens "
            "WHERE owner = ? ORDER BY created_at, token_sha256", (owner,))]
    finally:
        conn.close()


def revoke(owner: str, token_sha256: str) -> bool:
    """Delete one token. Returns whether a row was there.

    Scoped to `owner` in the WHERE clause, not checked beforehand: an owner naming another
    owner's token hash deletes nothing, and learns nothing from the answer either."""
    conn = connect()
    try:
        n = conn.execute("DELETE FROM tokens WHERE owner = ? AND token_sha256 = ?",
                         (owner, token_sha256)).rowcount
        conn.commit()
    finally:
        conn.close()
    return n > 0


# ── grant codes ──────────────────────────────────────────────────────────────────

def mint_grant(owner: str, *, label: str | None = None) -> str:
    """Mint a one-time code and return it once. Same discipline as a token: only the hash lands.

    The alphabet is narrower than a token's, and `_CODE_ALPHABET` says why."""
    code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
    conn = connect()
    try:
        conn.execute("INSERT INTO grant_codes (code_sha256, owner, label) VALUES (?,?,?)",
                     (token_hash(code), owner, label))
        conn.commit()
    finally:
        conn.close()
    return code


class GrantUnavailable(RuntimeError):
    """A code that is not a code, or is one that has already been spent. ONE type for both,
    because telling the two apart is information a stranger holding a wrong code should not get,
    and the holder of a real code does the same thing with either answer: ask for a new one."""


def redeem_grant(code: str, install_id: str | None = None) -> tuple[str, str]:
    """Exchange a code for a reader token, exactly once. Returns `(owner, token)`.

    THE ROW COUNT DECIDES. The claim is a single conditional UPDATE — `redeemed_at IS NULL` is
    part of the WHERE, so SQLite's own write lock serializes two simultaneous redeems and the
    loser updates zero rows. A read-then-write would have a window between the check and the
    claim where both callers see NULL; there is no such window here, so the one-time property
    does not depend on there being one process."""
    h = token_hash(code)
    conn = connect()
    try:
        cur = conn.execute(
            "UPDATE grant_codes SET redeemed_at = datetime('now') "
            "WHERE code_sha256 = ? AND redeemed_at IS NULL", (h,))
        if cur.rowcount != 1:
            conn.commit()
            raise GrantUnavailable("that code is not valid, or has already been used.")
        row = conn.execute("SELECT owner, label FROM grant_codes WHERE code_sha256 = ?",
                           (h,)).fetchone()
        owner, label = row["owner"], row["label"]
        conn.commit()
    finally:
        conn.close()
    # After the claim is committed, so a crash between the two leaves the code SPENT and no token
    # minted. That direction is the safe one: the owner mints another code, where the reverse
    # would leave a live token nobody knows about.
    return owner, mint_token(owner, "reader", label=label, install_id=install_id)


# ── usage ────────────────────────────────────────────────────────────────────────

def record_usage(owner: str, reader_sha256: str, tool: str, *, zero_results: bool = False) -> None:
    """One increment per served request, aggregated to the day. Four values, none of them the
    query, and no record of WHEN within the day.

    Called AFTER the read runs, so the count is of reads that happened rather than of reads that
    were attempted — this is the meter a rate-limit policy will read, and a failed request that
    returned nothing should not spend somebody's allowance.

    The upsert is what makes this a counter rather than a log: the primary key is the whole
    grain, so the second read of the day has nowhere to land except on the first one's row."""
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO usage_daily (day, owner, reader, tool, n, zero_results) "
            "VALUES (date('now'), ?, ?, ?, 1, ?) "
            "ON CONFLICT(day, owner, reader, tool) "
            "DO UPDATE SET n = n + 1, zero_results = zero_results + excluded.zero_results",
            (owner, reader_sha256, tool, 1 if zero_results else 0),
        )
        conn.commit()
    finally:
        conn.close()


def usage_total(owner: str, reader_sha256: str | None = None) -> int:
    """How many reads this knowledge base has served, optionally for one reader."""
    conn = connect()
    try:
        if reader_sha256 is None:
            return conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily WHERE owner = ?",
                                (owner,)).fetchone()[0]
        return conn.execute(
            "SELECT COALESCE(SUM(n), 0) FROM usage_daily WHERE owner = ? AND reader = ?",
            (owner, reader_sha256)).fetchone()[0]
    finally:
        conn.close()


def stats_rollup() -> dict:
    """Every number the public page shows, in one open. Roll-up only: no owner, no reader, no
    label appears in the result — that is what makes the page publishable rather than an
    operator dashboard behind a credential.

    `zero_result_rate` is None rather than 0.0 when nothing has been searched, because "no
    searches yet" and "every search found something" are different facts and a page that
    printed 0% for both would be lying about one of them.

    Deliberately NOT here: `kbs_published`. This module owns `service.db` and nothing else, and
    the peers registry is a different file — `service/app.py` composes the two."""
    conn = connect()
    try:
        total = conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily").fetchone()[0]
        recent = conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily "
                              "WHERE day >= date('now', '-30 days')").fetchone()[0]
        by_tool = {r["tool"]: r["n"] for r in conn.execute(
            "SELECT tool, SUM(n) AS n FROM usage_daily GROUP BY tool")}
        readers_30d = conn.execute("SELECT COUNT(DISTINCT reader) FROM usage_daily "
                                   "WHERE day >= date('now', '-30 days')").fetchone()[0]
        readers = conn.execute("SELECT COUNT(*) FROM tokens WHERE role = 'reader'").fetchone()[0]
        minted = conn.execute("SELECT COUNT(*) FROM grant_codes").fetchone()[0]
        redeemed = conn.execute(
            "SELECT COUNT(redeemed_at) FROM grant_codes").fetchone()[0]
        searches, zeroes = conn.execute(
            "SELECT COALESCE(SUM(n), 0), COALESCE(SUM(zero_results), 0) FROM usage_daily "
            "WHERE tool = 'search'").fetchone()
    finally:
        conn.close()
    return {
        "reads_total": total,
        "reads_30d": recent,
        "reads_by_tool": by_tool,
        "active_readers_30d": readers_30d,
        "readers_total": readers,
        "codes_minted": minted,
        "codes_redeemed": redeemed,
        "zero_result_rate": (zeroes / searches) if searches else None,
    }
