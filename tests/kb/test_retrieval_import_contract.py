"""The retrieval unit's import-direction contract (S3, S5, S7) — the "Allowed imports" table of
docs/plans/2026-10-06-restructure-retrieval.md (§2), checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down and the unit has no
    cycle; retrieval depends on library, model-access (through `embed`) and two `opyt_core`
    foundations only, never on oracles, research, frontier, curation, sharing, worker or the
    server;
  • the modules the restructure deleted stay deleted;
  • imports sit at the top of the file, and `pipeline/retrieval/__init__.py` stays empty.

Other units import retrieval freely (it sits below them), so unlike frontier's contract there is no
list of importers. Relative imports are resolved, as in tests/kb/test_frontier_import_contract.py,
which this mirrors. Changing the table is changing the design: do it with the plan, not to make a
red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")

R = "pipeline.retrieval."
_SCHEMA, _DERIVE = "pipeline.kb.schema", "pipeline.kb.derive"
ALLOWED: dict[str, set[str]] = {
    # ── leaves ──
    "pipeline/retrieval/cards.py": set(),
    "pipeline/retrieval/rank.py": set(),
    "pipeline/retrieval/excerpt.py": set(),
    "pipeline/retrieval/corpus_census.py": {R + "rank"},
    "pipeline/retrieval/peers.py": {_SCHEMA, "pipeline.kb.writes"},
    "pipeline/retrieval/new_material.py": {"opyt_core.paths"},
    "pipeline/retrieval/filters.py": {_DERIVE, _SCHEMA},
    "pipeline/retrieval/who.py": {_DERIVE, _SCHEMA},
    # ── the engine and the notices ──
    "pipeline/retrieval/engine.py": {R + "filters", R + "rank", R + "excerpt"},
    "pipeline/retrieval/notices.py": {R + "peers", R + "engine", _DERIVE, _SCHEMA},
    # ── the two knowledge bases ──
    "pipeline/retrieval/local.py": {R + "engine", R + "filters", R + "who", R + "notices",
                                    R + "cards", R + "corpus_census", R + "peers", _SCHEMA,
                                    "pipeline.kb.paper_text"},
    "pipeline/retrieval/served.py": {R + "peers", "opyt_core.service_http"},
    # ── the four reads ──
    "pipeline/retrieval/reads.py": {R + "local", R + "served", R + "peers", R + "notices",
                                    R + "filters", _SCHEMA},
}
INIT = "pipeline/retrieval/__init__.py"
# Deleted by the restructure (2026-10-06); their code lives in pipeline/retrieval/.
RETIRED = ["opyt_core/kb.py", "opyt_core/kb_remote.py", "pipeline/kb/retrieve.py",
           "pipeline/rank.py", "pipeline/kb/peers.py", "pipeline/kb/corpus_census.py",
           "pipeline/kb/new_material.py"]


def _module_of(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def _imported(rel: str, tree: ast.AST) -> list[tuple[int, str]]:
    """Every repository module `rel` imports, as `(line, dotted name)`, relative ones resolved."""
    package = _module_of(rel).rsplit(".", 1)[0]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                anchor = ".".join(parts[:len(parts) - node.level + 1])
                base = f"{anchor}.{base}" if base else anchor
            for a in node.names:
                full = f"{base}.{a.name}"
                is_module = (ROOT / (full.replace(".", "/") + ".py")).exists()
                out.append((node.lineno, full if is_module else base))
    return [(ln, m) for ln, m in out if m.split(".")[0] in INTERNAL]


def _parse(rel: str) -> ast.AST:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in ALLOWED:
        for fn in ast.walk(_parse(rel)):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bad.append(f"{rel}:{node.lineno} imports inside {fn.name}()")
    return bad


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_every_module_of_the_unit_has_a_row():
    unit = {str(p.relative_to(ROOT)) for p in (ROOT / "pipeline/retrieval").rglob("*.py")} - {INIT}
    assert sorted(unit - set(ALLOWED)) == []


def test_the_modules_the_restructure_deleted_stay_deleted():
    assert [rel for rel in RETIRED if (ROOT / rel).exists()] == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_the_package_init_stays_empty():
    assert (ROOT / INIT).read_text(encoding="utf-8").strip() == "", f"{INIT} must stay empty"
