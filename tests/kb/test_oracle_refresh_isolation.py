"""Atom-rail isolation for the Oracle refresh loop.

`pipeline/kb/` has no imports from `pipeline/radar/` or any vault module. The radar rail had an
equivalent refresh loop, which made an import the tempting shortcut and the wrong move: that rail
read `radar_atoms`, a pre-atom-KB table, so importing it would weld this loop's lifetime to a
legacy module.

The `atom-rail-not-welded-to-catchup` and `atom-rail-not-welded-to-radar` guards cover
`mcp_server.catchup` and `pipeline.radar` at the AST level. This covers the rest by grep, and pins
the tables the loop is allowed to write.

The scan covers every live package, not just `pipeline/kb/`: a module outside it once imported a
timestamp parse from the dead rail, which is exactly the weld this file forbids. A scan scoped to
one package proves a property about that package, not about the rail. Keep this scan on the rail,
not on a directory.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NEW = ("pipeline/oracles/coverage.py", "pipeline/oracles/freshness.py", "pipeline/oracles/registry.py", "pipeline/oracles/pairs.py")

# The live atom rail, by package. Both must stay free of dead-rail imports; see the note above on
# why `pipeline/artifacts/` is not optional here.
_LIVE_PACKAGES = ("pipeline/kb", "pipeline/artifacts")

# Vault-rail + pre-atom-KB modules. `pipeline.ask` is the deleted vault retrieval rail;
# `mcp_server.catchup` is a deleted vault catch-up job, kept in this list because a deleted module
# is exactly what someone re-creates; `pipeline.radar` is pre-atom-KB.
_FORBIDDEN = re.compile(
    r"\b(?:from|import)\s+(pipeline\.radar|pipeline\.ask|pipeline\.processing|"
    r"mcp_server\.catchup|opyt_core\.actions|gui\.indexer)\b")

# Vault / pre-atom-KB tables. The refresh loop writes atoms, bodies, edges, entities,
# oracle_sources, circuit_breaker and sync_lock — none of these.
_VAULT_TABLES = re.compile(r"\b(notes_fts|note_embeddings|radar_atoms|note_topics|daily_activity)\b"
                           r"|\b(?:FROM|INTO|JOIN|UPDATE)\s+(?:notes|topics|taxonomy)\b", re.I)


@pytest.mark.parametrize("name", _NEW)
def test_new_modules_import_nothing_from_a_dead_rail(name):
    hit = _FORBIDDEN.search((_ROOT / name).read_text())
    assert hit is None, f"{name} imports {hit.group(1)!r} — see this module's docstring"


def _query_text(path: Path) -> str:
    """Every string literal in the file except docstrings, joined.

    SQL lives in string literals, so a scan that strips all strings would catch nothing — but both
    new modules name `radar_atoms` / `notes_fts` in their prose, on purpose, to record which tables
    must never be touched and why. A naive grep fails on exactly the comments that document the
    invariant, and the obvious fix (delete the prose) is backwards. Docstrings out, SQL in."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docs.add(id(body[0].value))
    return "\n".join(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and id(n) not in docs)


@pytest.mark.parametrize("name", _NEW)
def test_new_modules_query_no_vault_or_radar_table(name):
    hit = _VAULT_TABLES.search(_query_text(_ROOT / name))
    assert hit is None, f"{name} names the retired table {hit.group(0)!r}"


def test_the_table_scan_would_actually_catch_a_real_query(tmp_path):
    """A negative assertion with no positive control is an assertion that always passes."""
    bad = tmp_path / "bad.py"
    bad.write_text('"""A docstring mentioning notes_fts."""\nx = 1\n')
    assert _VAULT_TABLES.search(_query_text(bad)) is None        # prose → correctly ignored
    bad.write_text('x = conn.execute("SELECT 1 FROM notes")\n')
    assert _VAULT_TABLES.search(_query_text(bad)) is not None    # a real query → caught


@pytest.mark.parametrize("package", _LIVE_PACKAGES)
def test_whole_live_package_still_has_no_dead_rail_imports(package):
    """The property this change had to preserve, not just introduce for two files."""
    offenders = [f"{package}/{p.name}" for p in sorted((_ROOT / package).glob("*.py"))
                 if _FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_the_import_scan_would_actually_catch_a_real_import(tmp_path):
    """Same reasoning as the table-scan control below: a negative assertion with no positive
    control is an assertion that always passes. This one is not theoretical — the exact line it
    reconstructs is one that `pipeline/artifacts/policy.py` once carried, and that a kb-only scan
    missed."""
    ok = tmp_path / "ok.py"
    ok.write_text('"""Prose naming pipeline.radar, which must NOT trip the scan."""\nx = 1\n')
    assert _FORBIDDEN.search(ok.read_text()) is None
    bad = tmp_path / "bad.py"
    bad.write_text("from pipeline.radar.refresh_state import _parse_ts\n")
    assert _FORBIDDEN.search(bad.read_text()) is not None


def test_the_guard_rules_are_registered():
    """`.guards.py` is the enforcement; a test that they exist keeps a silent deletion visible.

    Skips where there is no `.guards.py` at all — the public tree excludes it as author
    discipline, and an assertion about rules that cannot exist there is vacuous, not failing.
    Same shape as test_migration_guards.py's skip on a checkout with no mainline ref."""
    if not (_ROOT / ".guards.py").exists():
        pytest.skip("no .guards.py in this checkout (excluded from the public tree)")
    src = (_ROOT / ".guards.py").read_text()
    assert "atom-rail-not-welded-to-catchup" in src
    assert "'modules': {'mcp_server.catchup'}" in src
    assert "atom-rail-not-welded-to-radar" in src
    assert "'modules': {'pipeline.radar'}" in src


def test_the_shared_timestamp_parse_is_neutral_not_forked():
    """`parse_ts` lives in a neutral module both rails may import, and NOT as a private copy.

    Separate copies of this parse drift apart (one gained a `str()` coercion the others lacked). The
    shared module only holds while there is one
    body: an inline re-fork in either live package would satisfy every other test in this file,
    since a copy imports nothing."""
    assert (_ROOT / "pipeline" / "timeparse.py").exists()
    for package in _LIVE_PACKAGES:
        for path in sorted((_ROOT / package).glob("*.py")):
            tree = ast.parse(path.read_text())
            forked = [n.name for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name in ("parse_ts", "_parse_ts")]
            assert forked == [], f"{package}/{path.name} re-forks {forked} — import it instead"


def test_refresh_infra_deps_are_generic_not_vault():
    """circuit_breaker / sync_lock / dedup_store are shared infra; each may reach only
    `pipeline.sqlite_db` and `pipeline.loops_db`, which is why the loop is allowed to depend on them."""
    root = Path(__file__).resolve().parents[2] / "pipeline"
    for mod in ("circuit_breaker.py", "sync_lock.py", "dedup_store.py"):
        assert _FORBIDDEN.search((root / mod).read_text()) is None, mod
