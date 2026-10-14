"""Every write to opyt.db goes through `write_tx` (pipeline/kb/writes.py).

A bare `conn.commit()` gets none of what `write_tx` gives: `BEGIN IMMEDIATE`, the 30 s retry past
`busy_timeout`, the slow-hold log, and joining a caller's transaction instead of committing it
early. So in the server and pipeline code a `.commit()` may appear only:
  • in `writes.py` itself;
  • in a module whose connection is another database file (its own lock, its own rules);
  • in DDL setup, next to `executescript`, which commits by itself and cannot run inside a
    `write_tx` anyway.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("pipeline", "mcp_server", "opyt_core")

OTHER_DATABASES = {
    "pipeline/kb/writes.py",                  # the one way to write the store
    "pipeline/circuit_breaker.py",            # circuit_breaker.db
    "pipeline/platforms/x/rate.py",           # x_rate_meter.db
    "pipeline/worker/job_queue.py",           # rail_jobs.db
    "pipeline/sync_lock.py",                  # loops.db
    "pipeline/loops_db.py",                   # loops.db
    "pipeline/platforms/browser/cookies.py",  # a copy of Chrome's Cookies file
    "pipeline/sharing/export.py",             # the export's own file
    "pipeline/sharing/fingerprint.py",        # a copy's digest stamp, in that copy
    "pipeline/sharing/delta.py",              # the delta's own file
}


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Attribute, ast.Name))
            and (n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id) == name]


def _is_ddl(fn: ast.AST) -> bool:
    """A setup function: it runs `executescript`, or calls an `init_*schema` that does."""
    return bool(_calls(fn, "executescript")) or any(
        isinstance(c.func, (ast.Attribute, ast.Name))
        and (c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id).startswith("init_")
        and (c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id).endswith("schema")
        for c in ast.walk(fn) if isinstance(c, ast.Call))


def test_no_bare_commit_on_the_store():
    bare = []
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in OTHER_DATABASES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or _is_ddl(fn):
                    continue
                # Only the commits directly in this function, not in a nested one it defines.
                nested = {id(c) for inner in ast.walk(fn)
                          if inner is not fn and isinstance(inner, (ast.FunctionDef,
                                                                    ast.AsyncFunctionDef))
                          for c in _calls(inner, "commit")}
                bare += [f"{rel}:{c.lineno}" for c in _calls(fn, "commit")
                         if id(c) not in nested and not c.args]
    assert bare == [], ("write the store with `with write_tx(conn, site=...)`, not a bare "
                        f"commit: {bare}")
