"""`ALTER TABLE ... DROP COLUMN` stays out of shipping code: it needs SQLite >= 3.35.

The store's DDL describes only the current shape, so today nothing drops a column at runtime. The
ban outlives that: the first migration written for a released store must use the portable
create-copy-drop-rename, and this is what catches it if it does not.
"""
from __future__ import annotations

import ast
import pathlib

_SHIPPED = ("opyt_core", "mcp_server", "pipeline", "service")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every Constant that IS a docstring. These are the strings that TEACH the rule, and
    a checker that cannot tell them from an executed one is a checker nobody keeps."""
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
    on every writable `connect()`, so on an older SQLite such a statement does not return a wrong
    answer — the store does not open, and every rail dies on the same line.

    It is the only construct imposing that floor: the highest version-gated feature anywhere else
    is `ON CONFLICT ... DO UPDATE` (3.24, 2018) and FTS5 (3.9, 2015). Use the portable
    create-copy-drop-rename instead (create the new table, copy, drop, rename).

    This is a test rather than a string-grep guard, deliberately: a grep flags the docstrings that
    exist to teach this rule. Walking the AST and excluding docstrings is what makes the check
    honest, and this is where that can be done.
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
        "portable create-copy-drop-rename required: "
        + ", ".join(offenders))
