"""The subtractive migrations — the create-copy-drop-rename ones, tested for the first time.

Four of them rebuild a table to drop a column: `_drop_sittings_lam`, `_drop_oracle_ingest_window`,
`_drop_frontier_pair_counters` and `_drop_entities_kind`. Until 2026-09-05 not one had a test, and
they run on every writable `connect()` — so a mistake in one is a store that will not OPEN, on a
machine whose owner did nothing but leave it closed for a fortnight.

`_drop_entities_kind` is the one covered here, because it is the one that changed. It replaced a
bare `ALTER TABLE entities DROP COLUMN kind`, which needs SQLite >= 3.35 and so contradicted the
distributability invariant the sibling migrations' own docstrings state.

WHAT IS DELIBERATELY NOT ASSERTED: the statement text. `test_schema_backfill_guards.py` records
why — a statement's spelling is a proxy for the thing that actually hurts, and an earlier draft
that spied on `conn.execute` flagged three statements that were fine. These tests assert the
BEHAVIOUR: an old-shaped store converges, keeps its rows, and converging twice costs nothing.

The ban on writing `ALTER TABLE ... DROP COLUMN` in shipping code is enforced by
`test_no_shipping_code_executes_ALTER_TABLE_DROP_COLUMN` below, which says at length why it is a
TEST and not a `.guards.py` rule. This header credited a rule named `no-alter-table-drop-column`
until 2026-09-06; no such rule has ever existed. `7d2ab03a` drafted it, confirmed it firing on the
four migration docstrings that exist to teach the ban, and withdrew it in the same commit — the
header is a fossil of the draft, and it contradicted its own file.
"""
from __future__ import annotations

import ast
import pathlib
import sqlite3

import pytest

from pipeline.kb import schema

# The `entities` shape as it stood before 2026-08-23, when `kind` was still written by 14 ingest
# sites. Spelled out rather than sliced from `_DDL`, because the point of the fixture is to be the
# OLD schema — deriving it from the current one would make the test pass by construction.
_OLD_ENTITIES = """
CREATE TABLE entities (
  entity_id      TEXT PRIMARY KEY,
  name           TEXT,
  kind           TEXT,
  identity_links TEXT,
  canonical_id   TEXT,
  profile        TEXT
);
"""

_OLD_READER_COST_TABLES = """
CREATE TABLE frontier_reader_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generator TEXT,
  sitting_id TEXT,
  lens TEXT,
  ran_at TEXT NOT NULL,
  window_from TEXT, window_to TEXT,
  atoms_read INTEGER, consensus TEXT,
  model TEXT, in_tokens INTEGER, out_tokens INTEGER, cost_usd REAL,
  emitted INTEGER, new INTEGER, refreshed INTEGER, marked_dormant INTEGER,
  kept INTEGER, dropped INTEGER, unverdicted INTEGER, middle_share REAL,
  status TEXT NOT NULL, reason TEXT
);
CREATE TABLE sitting_lens_outputs (
  sitting_id TEXT NOT NULL,
  lens TEXT NOT NULL,
  output TEXT NOT NULL,
  model TEXT, in_tokens INTEGER, out_tokens INTEGER, cost_usd REAL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (sitting_id, lens)
);
"""


@pytest.fixture()
def old_store(kb_home, tmp_path):
    """A store carrying `entities.kind`, with rows whose OTHER columns hold data — an empty table
    would migrate cleanly even if the copy step dropped every row."""
    path = tmp_path / "old.db"
    c = sqlite3.connect(str(path))
    c.executescript(_OLD_ENTITIES)
    c.executemany(
        "INSERT INTO entities (entity_id, name, kind, identity_links, canonical_id, profile) "
        "VALUES (?,?,?,?,?,?)",
        [("x:1", "Ada", "person", '["github:ada"]', "x:1", '{"classified_kind": "person"}'),
         ("x:2", "Acme", "org", "[]", "x:2", None),
         ("x:3", "Unset", None, None, None, None)])
    c.commit()
    c.close()
    return path


def test_an_old_store_loses_entities_kind_and_keeps_every_row(old_store):
    """The whole point of a create-copy-drop-rename: the column goes and the DATA does not."""
    c = schema.connect(old_store)
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(entities)")]
        assert "kind" not in cols
        assert {"entity_id", "name", "identity_links", "canonical_id", "profile"} <= set(cols)

        rows = {r["entity_id"]: r for r in c.execute("SELECT * FROM entities")}
        assert set(rows) == {"x:1", "x:2", "x:3"}
        assert rows["x:1"]["name"] == "Ada"
        assert rows["x:1"]["identity_links"] == '["github:ada"]'
        assert rows["x:1"]["profile"] == '{"classified_kind": "person"}'
        assert rows["x:2"]["canonical_id"] == "x:2"
        assert rows["x:3"]["name"] == "Unset"
    finally:
        c.close()


def test_the_index_survives_the_rebuild(old_store):
    """The rebuild DROPs the table, so every index on it goes too. `idx_entities_canonical` is
    recreated further down `init_kb_schema` with `IF NOT EXISTS` — which only works because the
    rebuild runs BEFORE it. Reversing that order leaves the store unindexed and nothing says so.
    """
    c = schema.connect(old_store)
    try:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='entities'")}
        assert "idx_entities_canonical" in names
    finally:
        c.close()


def test_an_index_declared_in_the_ddl_survives_the_rebuild(kb_home, tmp_path):
    """The half `init_kb_schema` cannot cover. `_DDL` runs at the TOP of that function, so an index
    declared beside its CREATE TABLE is created and then dropped with the table — and nothing reads
    an index to notice it is missing. `idx_sittings_read` is that case: `init_kb_schema` re-creates
    the other three `sittings` indexes further down and not that one, which is why
    `_SITTINGS_INDEXES` existed as a third hand-kept copy of what `_DDL` already said.

    `lam` is re-added rather than the old table being spelled out, because what is under test is
    index survival across a rebuild, not the 2026-08-25 `sittings` shape."""
    path = tmp_path / "lam.db"
    c = schema.connect(path)
    c.execute("ALTER TABLE sittings ADD COLUMN lam REAL")
    c.commit()
    c.close()

    c = schema.connect(path)                        # `_drop_sittings_lam` fires here
    try:
        assert "lam" not in [r[1] for r in c.execute("PRAGMA table_info(sittings)")]
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sittings'")}
        assert "idx_sittings_read" in names
        # The three `init_kb_schema` puts back are the other half of the same guarantee.
        assert {"idx_sittings_continues", "idx_sittings_parent", "idx_sittings_region"} <= names
    finally:
        c.close()


def test_converging_twice_changes_nothing(old_store):
    """Idempotency is the read guard, not luck: the second pass finds no `kind` and returns before
    it opens a transaction. A migration that re-ran would rebuild the table on every connect, which
    is the lock-contention shape `test_schema_backfill_guards.py` exists to prevent."""
    c = schema.connect(old_store)
    before = c.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    c.close()

    c = schema.connect(old_store)          # a second full init_kb_schema
    try:
        assert c.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == before
        assert "kind" not in [r[1] for r in c.execute("PRAGMA table_info(entities)")]
    finally:
        c.close()


def test_a_store_that_never_had_the_column_is_untouched(kb_home, tmp_path):
    """Fail-safe: a fresh store has no `kind` and the migration must not care. The guard is
    `if not cols or "kind" not in cols`, and the `not cols` half covers a table that does not
    exist yet — `PRAGMA table_info` on an absent table returns no rows rather than raising."""
    fresh = schema.connect(tmp_path / "fresh.db")
    try:
        assert "kind" not in [r[1] for r in fresh.execute("PRAGMA table_info(entities)")]
    finally:
        fresh.close()


def test_an_old_store_loses_reader_cost_columns_and_keeps_its_receipts(kb_home, tmp_path):
    """Dollar accounting is removed without losing the response metadata it used to sit beside.

    This store is TWO drops behind on `frontier_reader_runs`: the fixture carries `cost_usd` and
    the never-written `window_from`/`window_to`, which went 2026-09-06. Both migrations run on the
    same connect, in sequence, and the receipt columns between them survive."""
    path = tmp_path / "old-reader-costs.db"
    old = sqlite3.connect(path)
    old.executescript(_OLD_READER_COST_TABLES)
    old.execute(
        "INSERT INTO frontier_reader_runs (generator, ran_at, model, in_tokens, out_tokens, "
        "cost_usd, status) VALUES ('sitting:x', '2026-09-05T00:00:00Z', 'm', 10, 2, .01, 'ok')"
    )
    old.execute(
        "INSERT INTO sitting_lens_outputs (sitting_id, lens, output, model, in_tokens, "
        "out_tokens, cost_usd, created_at) VALUES ('s', 'briefing', 'out', 'm', 10, 2, .01, "
        "'2026-09-05T00:00:00Z')"
    )
    old.commit()
    old.close()

    conn = schema.connect(path)
    try:
        for table in ("frontier_reader_runs", "sitting_lens_outputs"):
            assert "cost_usd" not in [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        run_cols = [r[1] for r in conn.execute("PRAGMA table_info(frontier_reader_runs)")]
        assert "window_from" not in run_cols and "window_to" not in run_cols
        # And the index `_DDL` declares for this table is back — dropping the table dropped it,
        # twice, and `init_kb_schema` re-creates it below only because someone once noticed.
        assert "idx_frontier_runs_status_time" in {r[1] for r in conn.execute(
            "PRAGMA index_list(frontier_reader_runs)")}
        assert tuple(conn.execute(
            "SELECT model, in_tokens, out_tokens FROM frontier_reader_runs"
        ).fetchone()) == ("m", 10, 2)
        assert tuple(conn.execute(
            "SELECT model, in_tokens, out_tokens FROM sitting_lens_outputs"
        ).fetchone()) == ("m", 10, 2)
    finally:
        conn.close()


# ── the portability ban ──────────────────────────────────────────────────────────

_SHIPPED = ("opyt_core", "mcp_server", "pipeline", "service")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every Constant that IS a docstring. These are the strings that TEACH the rule —
    four migrations in schema.py explain why not to write the banned statement — and a checker
    that cannot tell them from an executed one is a checker nobody keeps."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def test_no_shipping_code_executes_ALTER_TABLE_DROP_COLUMN():
    """`ALTER TABLE ... DROP COLUMN` needs SQLite >= 3.35 (2021-03), and the distributability
    invariant forbids assuming a version of anything on the user's machine. `init_kb_schema` runs
    on EVERY writable `connect()`, so on an older SQLite such a statement does not return a wrong
    answer — the store does not OPEN, and every rail dies on the same line.

    One lived at `schema.py:1222` from 2026-08-23 to 2026-09-05, two hundred lines below the
    docstring stating the rule it broke. It was the ONLY line in the repository imposing that
    floor: swept 2026-09-05, the highest version-gated feature anywhere else is
    `ON CONFLICT ... DO UPDATE` (3.24, 2018) and FTS5 (3.9, 2015). Use the portable
    create-copy-drop-rename — `schema._drop_entities_kind` is the pattern.

    THIS IS A TEST AND NOT A `.guards.py` RULE, deliberately. The `str_contains` checker walks
    every string Constant (`scripts/guard.py::_strings_containing`), docstrings included, so it
    flags the four migration docstrings that exist to teach this. `guard.py`'s own header says
    the point of AST over grep is "zero false positives from prose, which is what keeps a guard
    trusted (a noisy guard gets disabled)" — a rule that fires on its own explanation fails that
    test. Excluding docstrings is what makes the check honest, and this is where that can be done.
    """
    repo = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for pkg in _SHIPPED:
        for py in sorted((repo / pkg).rglob("*.py")):
            tree = ast.parse(py.read_text())
            docs = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and id(node) not in docs and "DROP COLUMN" in node.value):
                    offenders.append(f"{py.relative_to(repo)}:{node.lineno}")
    assert not offenders, (
        "portable create-copy-drop-rename required (see schema._drop_entities_kind): "
        + ", ".join(offenders))


# ── `_rebuild_without`: the three defects the five copies all shared ───────────────
# Every subtractive migration was its own create-copy-drop-rename body, and all five carried the
# same three faults. These test the collapsed helper directly, because the faults are properties
# of the MECHANICS rather than of any one column's removal.

_OLD_ENTITIES_WITH_A_RETIRED_COLUMN = """
CREATE TABLE entities (
  entity_id      TEXT PRIMARY KEY,
  name           TEXT,
  kind           TEXT,
  legacy_note    TEXT,
  identity_links TEXT,
  canonical_id   TEXT,
  profile        TEXT
);
"""


@pytest.fixture()
def store_two_drops_behind(kb_home, tmp_path):
    """A store carrying `kind` AND a column the current `_DDL` no longer declares.

    This is the shape a SECOND subtractive migration creates. It is not hypothetical: removing
    `sittings.read_status` from `_DDL` puts every store last written before 2026-08-25 — the ones
    that still have `lam` — into exactly this state.
    """
    path = tmp_path / "two-behind.db"
    c = sqlite3.connect(str(path))
    c.executescript(_OLD_ENTITIES_WITH_A_RETIRED_COLUMN)
    c.execute("INSERT INTO entities (entity_id, name, kind, legacy_note, identity_links) "
              "VALUES (?,?,?,?,?)", ("x:1", "Ada", "person", "gone", '["github:ada"]'))
    c.commit()
    c.close()
    return path


def test_a_store_two_drops_behind_still_opens(store_two_drops_behind):
    """THE MEASURED BRICK. Every copy built its column list from `PRAGMA table_info` on the OLD
    table while slicing the replacement DDL from the CURRENT `_DDL`, so a column the DDL had since
    dropped went into the INSERT's column list and not into the new table:
    `OperationalError: table _entities_new has no column named legacy_note`.

    `init_kb_schema` runs on every writable `connect()`, so that is not a failed migration — it is
    a store that never opens again, for someone who did nothing but leave it closed. Intersecting
    with the NEW table's own `table_info` is what fixes it, in one place, for every future drop.
    """
    c = schema.connect(store_two_drops_behind)
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(entities)")]
        assert "kind" not in cols and "legacy_note" not in cols
        row = c.execute("SELECT * FROM entities").fetchone()
        assert row["name"] == "Ada" and row["identity_links"] == '["github:ada"]'
    finally:
        c.close()


def test_an_orphan_temp_table_from_an_interrupted_run_does_not_brick_the_store(old_store):
    """The second half of the same failure. The old slice replaced
    `CREATE TABLE IF NOT EXISTS entities (` with a BARE `CREATE TABLE _entities_new (`, so a
    retry after any interruption raised `table _entities_new already exists` inside
    `init_kb_schema` — on every connect, with no repair path.

    Dropping the temp name rather than restoring `IF NOT EXISTS`: an orphan left by an older
    `_DDL` has the wrong shape, and `IF NOT EXISTS` would silently copy into it.
    """
    c = sqlite3.connect(str(old_store))
    c.executescript("CREATE TABLE _entities_new (entity_id TEXT, stale_shape TEXT)")
    c.commit()
    c.close()

    c = schema.connect(old_store)
    try:
        assert "kind" not in [r[1] for r in c.execute("PRAGMA table_info(entities)")]
        assert c.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 3
    finally:
        c.close()


def test_the_rebuild_is_one_transaction(old_store):
    """`conn.execute("BEGIN")` followed by `conn.executescript(ddl)` was not a transaction at all:
    `executescript` issues an implicit COMMIT, so the BEGIN was committed away and every later
    statement ran in autocommit. Measured — `in_transaction` went True then False across the call,
    and the orphan `_entities_new` survived a simulated crash.

    A failure anywhere inside must therefore leave the store exactly as it was, which is what this
    asserts by failing the last statement on purpose.
    """
    c = sqlite3.connect(str(old_store))          # RAW: `schema.connect` would migrate it first
    try:
        with pytest.raises(sqlite3.OperationalError):
            schema._rebuild_without(c, "entities", "kind",
                                    after=("SELECT no_such_column FROM entities",))
        assert "kind" in [r[1] for r in c.execute("PRAGMA table_info(entities)")]
        assert c.execute("SELECT name FROM sqlite_master WHERE name='_entities_new'"
                         ).fetchone() is None
        assert c.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 3
    finally:
        c.close()
