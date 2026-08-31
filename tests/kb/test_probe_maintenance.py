"""The store-wide maintenance tools must see BOTH vector-bearing tables.

`kb_meta` is a claim about how to decode every blob in the file — its dim, its strip, and its
WIDTH. Until 2026-08-11 exactly one table held vectors, so a tool that walked `chunks` and then
stamped was telling the truth. `probe_chunks` makes that false, and these tests are the only place
that can prove it: the tools live on main, the second table does not.

Two failures, and the second is the worse one:

  restrip → SILENT STALENESS. Probe vectors keep an older strip while kb_meta says otherwise.
            Degraded retrieval, recoverable by re-running.
  dtype   → A WIDTH LIE. Probe blobs keep the old width while kb_meta advertises the new one, and
            readers take the width from kb_meta and reshape. That is garbage, not drift, and
            nothing catches it — `assert_model` compares kb_meta to the EMBEDDER, never to the
            blobs. `convert_chunk_storage_dtype`'s docstring promises the two never disagree.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.kb import probe_store, schema
from pipeline.kb.embed import convert_chunk_storage_dtype, stored_dtype
from pipeline.kb.ingest_common import AtomSink


def _seed(conn, embedder, *, trusted: int = 2, probe: int = 2) -> None:
    """A store with vectors in BOTH tables — the shape every one of these tests needs."""
    t = AtomSink(conn, embedder)
    for i in range(trusted):
        t.submit({"atom_id": f"x:{i}", "source_type": "x", "who_id": "x:user:9",
                  "description": "d", "raw_hash": f"h{i}"}, f"a trusted agent framework {i}")
    t.close()
    p = AtomSink(conn, embedder, writer=probe_store.write_probe_atom)
    for i in range(probe):
        p.submit({"atom_id": f"xprobe:{i}", "source_type": "x", "who_id": "x:user:11",
                  "description": "d", "raw_hash": f"p{i}"}, f"a candidate crypto rollup {i}")
    p.close()


# ── the width lie ─────────────────────────────────────────────────────────────

def test_dtype_conversion_covers_probe_vectors(kb_home, fake_embedder):
    conn = schema.connect()
    _seed(conn, fake_embedder)
    start = stored_dtype(conn)
    target = "float32" if start != "float32" else "float16"

    n = convert_chunk_storage_dtype(conn, target)
    assert stored_dtype(conn) == target

    # EVERY blob must now decode at the advertised width with no remainder. A probe row left at the
    # old width is not a smaller number — `np.frombuffer` either raises or silently reinterprets.
    width = np.dtype(target).itemsize
    for table in ("chunks", "probe_chunks"):
        for (blob,) in conn.execute(f"SELECT vector FROM {table} WHERE vector IS NOT NULL"):
            assert len(blob) % width == 0, f"{table} blob is not {target}-aligned"
    trusted = conn.execute("SELECT COUNT(*) FROM chunks WHERE vector IS NOT NULL").fetchone()[0]
    probe = conn.execute("SELECT COUNT(*) FROM probe_chunks WHERE vector IS NOT NULL").fetchone()[0]
    assert n == trusted + probe, "the converter reported fewer rows than it must have rewritten"
    assert probe > 0, "fixture failure — the test proves nothing without probe vectors"
    conn.close()


def test_dtype_conversion_leaves_a_probe_less_store_untouched(kb_home, fake_embedder):
    """A store that never probed must come out byte-identical. Creating empty probe tables as a
    side effect of a dtype conversion is a schema change nobody asked for, on the one path whose
    job is to leave the store consistent."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=0)
    conn.execute("DROP TABLE IF EXISTS probe_chunks")
    conn.execute("DROP TABLE IF EXISTS probe_atoms")
    conn.commit()

    target = "float32" if stored_dtype(conn) != "float32" else "float16"
    convert_chunk_storage_dtype(conn, target)
    assert not probe_store.probe_tables_exist(conn)
    conn.close()


# ── the silent staleness ──────────────────────────────────────────────────────

def test_probe_rows_are_reported_stale_when_the_strip_moves(kb_home, fake_embedder):
    conn = schema.connect()
    _seed(conn, fake_embedder)
    # Simulate vectors written by an older strip: blank the recorded surface.
    conn.execute("UPDATE probe_chunks SET embed_text = ''")
    conn.commit()

    s = probe_store.restrip_probe_rows(conn, None, profile="scaffolding", apply=False)
    assert s["stale"] > 0 and s["embedded"] == 0     # surveyed, nothing spent
    conn.close()


def test_restripping_probe_rows_rewrites_surface_and_vector(kb_home, fake_embedder):
    conn = schema.connect()
    _seed(conn, fake_embedder)
    conn.execute("UPDATE probe_chunks SET embed_text = ''")
    conn.commit()

    s = probe_store.restrip_probe_rows(conn, fake_embedder, profile="scaffolding", apply=True)
    assert s["embedded"] == s["stale"] > 0 and s["failed"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM probe_chunks WHERE embed_text = ''").fetchone()[0] == 0
    # Idempotent: a second pass finds nothing.
    assert probe_store.restrip_probe_rows(
        conn, fake_embedder, profile="scaffolding", apply=False)["stale"] == 0
    conn.close()


def test_a_failed_probe_group_writes_nothing_and_stays_stale(kb_home, fake_embedder):
    """Fail-safe granularity, mirroring the trusted pass: a group whose embed fails is skipped
    WHOLE. Partially-written rows would look current while holding old-surface vectors — the one
    outcome worse than doing nothing."""
    from tests.kb.conftest import RecordingEmbedder

    conn = schema.connect()
    _seed(conn, fake_embedder)
    conn.execute("UPDATE probe_chunks SET embed_text = ''")
    conn.commit()
    before = [r[0] for r in conn.execute("SELECT vector FROM probe_chunks ORDER BY chunk_id")]

    s = probe_store.restrip_probe_rows(conn, RecordingEmbedder(poison="candidate"),
                                       profile="scaffolding", apply=True)
    assert s["failed"] == s["stale"] > 0 and s["embedded"] == 0
    after = [r[0] for r in conn.execute("SELECT vector FROM probe_chunks ORDER BY chunk_id")]
    assert after == before                                     # no partial write
    assert conn.execute(
        "SELECT COUNT(*) FROM probe_chunks WHERE embed_text = ''").fetchone()[0] > 0  # still stale
    conn.close()


def test_restrip_migrates_a_probe_store_that_predates_embed_text(kb_home, fake_embedder):
    """CAUGHT LIVE. The first 25-account run built `probe_chunks` before the column existed, so the
    very store this hook was written for raised `no such column: embed_text` on its first real
    invocation. Tables-exist means migrate; only a probe-LESS store is left untouched."""
    conn = schema.connect()
    _seed(conn, fake_embedder)
    conn.execute("ALTER TABLE probe_chunks DROP COLUMN embed_text")   # the pre-column shape
    conn.commit()
    assert "embed_text" not in {r[1] for r in conn.execute("PRAGMA table_info(probe_chunks)")}

    s = probe_store.restrip_probe_rows(conn, None, profile="scaffolding", apply=False)
    assert s["stale"] > 0                                             # migrated, then surveyed
    conn.close()


@pytest.mark.parametrize("read", [
    lambda c: probe_store.count_probe_atoms(c),
    lambda c: probe_store.load_probe_hashes(c),
    lambda c: probe_store.probed_who_ids(c),
    lambda c: probe_store.pull_states(c),
    lambda c: probe_store.fresh_who_ids(c, ttl_days=30),
    lambda c: probe_store.restrip_probe_rows(c, None, profile="scaffolding"),
    lambda c: probe_store.convert_probe_chunk_dtype(c, np.dtype("float16"), np.dtype("float32")),
])
def test_no_read_ever_creates_the_probe_store(kb_home, read):
    """MEASURED, not theoretical: the first version of these readers called `init_probe_schema` to
    guarantee the tables they were about to query, and one `count_probe_atoms()` on a never-probed
    store created NINE tables (three of ours plus FTS5's six shadow tables). So merely ASKING "does
    this store hold candidate content" wrote a schema into every store that asked — including an
    archived store someone opened only to survey it, which no re-run undoes.

    Mirrors `schema.connect(read_only=True)`, which skips DDL so retrieval degrades to "no such
    table" rather than creating one. Only the WRITE paths may bring the store into existence."""
    conn = schema.connect()
    before = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'probe%'")}
    assert before == set()

    read(conn)                                   # a question, not a statement

    after = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'probe%'")}
    assert after == set(), f"a read created {sorted(after)}"
    conn.close()


def test_a_write_does_create_the_probe_store(kb_home, fake_embedder):
    """The other half — the guard above must not make the store uncreatable."""
    conn = schema.connect()
    assert not probe_store.probe_tables_exist(conn)
    probe_store.record_pull(conn, "x:user:11", probe_store.STATUS_EMPTY)
    assert probe_store.probe_tables_exist(conn)
    assert probe_store.pull_states(conn)["x:user:11"]["status"] == probe_store.STATUS_EMPTY
    conn.close()


def test_probe_hooks_no_op_without_a_probe_store(kb_home, fake_embedder):
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=0)
    conn.execute("DROP TABLE IF EXISTS probe_chunks")
    conn.commit()
    assert probe_store.restrip_probe_rows(conn, None, profile="scaffolding")["stale"] == 0
    assert probe_store.convert_probe_chunk_dtype(conn, np.dtype("float16"),
                                                 np.dtype("float32")) == 0
    conn.close()


# ── the gate: stamping is an assertion about the WHOLE file ───────────────────
#
# Driven through the REAL script, not through a re-stated predicate. An earlier version of this
# test asserted `not (a or b)` against an inline truth table — it would have passed with the gate
# deleted, which makes it a test of arithmetic rather than of the code.

def _strip_version_in(conn) -> str:
    from pipeline.kb.embed import read_kb_meta
    return (read_kb_meta(conn) or {}).get("strip_version") or ""


def _run_cli(monkeypatch, embedder, *args) -> int:
    import pipeline.kb.embed as embed_mod
    import scripts.restrip_embed_surface as restrip_cli

    monkeypatch.setattr(embed_mod, "get_kb_embedder", lambda *a, **k: embedder)
    return restrip_cli._cli(list(args))


def test_a_clean_trusted_pass_does_not_certify_a_failed_probe_pass(kb_home, fake_embedder,
                                                                   monkeypatch):
    """THE regression, in its exact shape. The trusted pass has nothing to do — so under the old
    one-population gate the script took its "nothing stale" branch and stamped, certifying probe
    vectors it had never looked at."""
    from tests.kb.conftest import RecordingEmbedder

    conn = schema.connect()
    _seed(conn, fake_embedder)                       # written at DEFAULT_PROFILE = scaffolding
    conn.execute("UPDATE kb_meta SET strip_version = 'stale-version'")
    conn.execute("UPDATE probe_chunks SET embed_text = ''")     # only the PROBE side is stale
    conn.commit()
    conn.close()

    _run_cli(monkeypatch, RecordingEmbedder(poison="candidate"), "--apply",
             "--profile", "scaffolding")

    conn = schema.connect()
    assert _strip_version_in(conn) == "stale-version", (
        "kb_meta was stamped while probe vectors were left on the old surface")
    conn.close()


def test_both_passes_clean_does_stamp(kb_home, fake_embedder, monkeypatch):
    """The gate must not be a permanent refusal — the ordinary path still certifies the store."""
    from pipeline.kb.embed_surface import DEFAULT_PROFILE, strip_version

    conn = schema.connect()
    _seed(conn, fake_embedder)
    conn.execute("UPDATE kb_meta SET strip_version = 'stale-version'")
    conn.execute("UPDATE probe_chunks SET embed_text = ''")
    conn.commit()
    conn.close()

    _run_cli(monkeypatch, fake_embedder, "--apply", "--profile", DEFAULT_PROFILE)

    conn = schema.connect()
    assert _strip_version_in(conn) == strip_version(DEFAULT_PROFILE)
    assert conn.execute(
        "SELECT COUNT(*) FROM probe_chunks WHERE embed_text = ''").fetchone()[0] == 0
    conn.close()


# ── the guard's advice must actually resolve the guard ─────────────────────────

def test_the_strip_guard_recommends_a_command_that_satisfies_it(kb_home, fake_embedder,
                                                                monkeypatch):
    """`assert_strip_version` refuses the write and prints the fix. Following that fix must WORK.

    It did not. The message said `restrip_embed_surface.py --apply`, whose `--profile` defaulted to
    `full` — a profile no store is ever built at. An operator following the message re-embedded
    every chunk onto the full surface (which moves the geometry `sitting_builder`'s floors are
    calibrated to), and landed back on the same refusal, because a strip_version carries its
    profile as a suffix and `2026-08-11.x` never equals `2026-08-11.x+scaffolding`.

    Asserted as a ROUND TRIP rather than as a string match: pull the profile out of the message the
    guard actually raises, and check that running at that profile stamps the identity the guard
    actually demands. A test pinning the literal text would still pass with the two sides
    disagreeing, which is the entire bug."""
    import re

    import scripts.restrip_embed_surface as restrip_cli
    from pipeline.kb.embed import SubspaceError, assert_strip_version
    from pipeline.kb.embed_surface import DEFAULT_PROFILE, strip_version

    conn = schema.connect()
    _seed(conn, fake_embedder)
    conn.execute("UPDATE kb_meta SET strip_version = 'some-older-generation'")
    conn.commit()

    with pytest.raises(SubspaceError) as e:
        assert_strip_version(conn)
    conn.close()

    m = re.search(r"--profile\s+(\S+)", str(e.value))
    assert m, f"the guard must NAME the profile, or its advice is ambiguous: {e.value}"
    advised = m.group(1)

    # the round trip: what the advice stamps == what the guard demands
    assert strip_version(advised) == strip_version(DEFAULT_PROFILE), (
        f"the guard advises --profile {advised!r}, which stamps "
        f"{strip_version(advised)!r}, but the guard demands "
        f"{strip_version(DEFAULT_PROFILE)!r} — following the message leaves the store refused")

    # …and the message stays runnable when the operator drops the flag, because the script's own
    # default is the same profile. Driven through a real no-argument run rather than by reading the
    # parser's default off argparse: what matters is the surface the run MEASURES AGAINST, and that
    # is the thing a future refactor could break while leaving the default string intact.
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        restrip_cli._cli([])                                   # no --profile, no --apply: dry run
    printed = out.getvalue()

    m = re.search(r"this build=(\S+)", printed)
    assert m, f"restrip must report the surface it measured against:\n{printed}"
    assert m.group(1) == strip_version(DEFAULT_PROFILE), (
        f"a no-flag restrip run measured against {m.group(1)!r}, but stores are built at "
        f"{strip_version(DEFAULT_PROFILE)!r} — the default surveys a surface no store has")
