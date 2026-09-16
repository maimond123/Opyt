"""Structural guards for B3's retired admission-state surface."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(), filename=relative_path)


def test_retired_admission_state_surface_stays_deleted():
    curation = _tree("pipeline/kb/curation_state.py")
    assert {node.name for node in ast.walk(curation) if isinstance(node, ast.FunctionDef)} \
        .isdisjoint({"connect", "is_stale"})

    refresh = _tree("pipeline/kb/oracle_refresh_state.py")
    source_row = next(node for node in refresh.body
                      if isinstance(node, ast.ClassDef) and node.name == "SourceRow")
    assert all(not isinstance(node, ast.FunctionDef) or node.name != "pair"
               for node in source_row.body)


def test_source_rows_are_consumed_as_the_schema_selects_them():
    refresh = _tree("pipeline/kb/oracle_refresh_state.py")
    row_loader = next(node for node in refresh.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_row_to_source")
    assert not any(isinstance(node, ast.If) for node in ast.walk(row_loader))
