"""The oracles unit's import-direction contract (S3, S5, S7) — the "Allowed imports" table of
docs/plans/2026-10-05-restructure-oracles.md (§2), checked against the code.

The unit is drawn in tiers, not one row per module: the foundation (the stores an Oracle is kept
in), the jobs, `discovery/`, `sources/` (the scrapers) and the flows. What a reader relies on, and
what an edit can break without failing any behaviour test:
  • arrows inside the unit point down a tier, the foundation's own arrows are the ones listed,
    and the unit has no cycle;
  • nothing in the unit imports the server, the frontier or forget, and of `opyt_core` only
    `paths`;
  • the library half of the X timeline (`ingest_x_footprint`) and the worker import nothing from
    the unit, and the tool reaches it only through its flows;
  • imports sit at the top, every `__init__.py` stays empty, and what the restructure deleted
    stays deleted.

Relative imports are resolved, as in tests/kb/test_retrieval_import_contract.py, which this
mirrors. Changing a tier or an exception is changing the design: do it with the plan, not to make
a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")
UNIT = "pipeline.oracles"

FOUNDATION = {"roster", "registry", "windows", "asks", "found", "review_items", "probe_store"}
JOBS = {"pairs", "topics", "coverage", "freshness", "milestones", "told", "checks", "ranking",
        "probe_search", "mentions", "reference", "identify", "preview"}
FLOWS = {"screen", "candidates", "admit", "progress", "review", "people_map"}
RANK = {"foundation": 0, "jobs": 1, "discovery": 1, "sources": 2, "flows": 3}

# The foundation's own arrows (§2, plus the two departures the Step 6 receipt recorded:
# roster → review_items, found → asks). Anything else between foundation modules is new design.
FOUNDATION_EDGES = {
    ("roster", "windows"), ("registry", "windows"), ("found", "windows"),
    ("review_items", "windows"), ("probe_store", "windows"),
    ("asks", "roster"), ("asks", "registry"), ("asks", "windows"),
    ("roster", "review_items"), ("found", "asks"),
}
# A job reaching up into the scrapers, each for a stated reason: a check, and the milestones that
# report a pick, read the very pair definitions and quota state the walks use, so a check and the
# walk that answers it cannot disagree about which pairs exist (`sources/pair_rows.py`).
JOB_READS_OF_SOURCES = {
    "checks": {"sources.pair_rows", "sources.timeline.walks"},
    "milestones": {"sources.pair_rows", "sources.threads.state", "sources.timeline.walks",
                   "sources.web.archives", "sources.web.platform"},
}
# `discovery/` stands on the foundation and these two jobs only.
DISCOVERY_JOBS = {"reference", "pairs"}

BANNED_OUTSIDE = ("mcp_server", "service", "pipeline.frontier", "pipeline.forget")
OPYT_CORE_ALLOWED = {"opyt_core.paths"}

TOOL = "mcp_server/oracle_tools.py"
# The flows, plus `windows`: the tool parses the lookback presets itself (the S4 item filed in
# Step 6 moves every parse there).
TOOL_REACHES = {f"{UNIT}.{m}" for m in FLOWS | {"windows"}}
LIBRARY_HALF = "pipeline/kb/ingest_x_footprint.py"
WORKER = "pipeline/worker"

# Deleted by the restructure; their code lives in pipeline/oracles/ (or went, S7).
RETIRED = ["pipeline/kb/background_scrapers", "pipeline/kb/oracles.py",
           "pipeline/kb/oracle_refresh.py", "pipeline/kb/oracle_refresh_state.py",
           "pipeline/kb/oracle_reviews.py", "pipeline/kb/pull_runs.py", "pipeline/kb/screen.py",
           "pipeline/kb/eligibility.py", "pipeline/kb/candidate_probe.py",
           "pipeline/kb/candidate_search.py", "pipeline/kb/probe_store.py",
           "pipeline/kb/probe_search.py", "pipeline/kb/scholar_probe.py",
           "pipeline/kb/ingest_scholar_footprint.py", "pipeline/kb/onboard_footprint.py",
           "pipeline/kb/source_adapters.py", "pipeline/ingestion/discover_profile.py",
           "pipeline/ingestion/handle_match.py", "pipeline/ingestion/trust_edges.py",
           "pipeline/ingestion/trust_graph.py", "pipeline/ingestion/trust_types.py",
           "pipeline/oracles/sources/after_forget.py", "pipeline/oracles/sources/web/outcomes.py"]


def _module_of(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def _is_module(dotted: str) -> bool:
    path = ROOT / dotted.replace(".", "/")
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


def _imported(rel: str, tree: ast.AST) -> list[tuple[int, str]]:
    """Every repository module `rel` imports, as `(line, dotted name)`, relative ones resolved."""
    me = _module_of(rel)
    package = me if rel.endswith("__init__.py") else me.rsplit(".", 1)[0]
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
                out.append((node.lineno, full if _is_module(full) else base))
    return [(ln, m) for ln, m in out if m.split(".")[0] in INTERNAL]


def _parse(rel: str) -> ast.AST:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)


def _unit_files() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pipeline/oracles").rglob("*.py"))


def _inner(dotted: str) -> str:
    """`pipeline.oracles.sources.web.archives` → `sources.web.archives`."""
    return dotted[len(UNIT) + 1:]


def _tier(inner: str) -> str:
    top = inner.split(".")[0]
    if top in ("discovery", "sources"):
        return top
    if top in FOUNDATION:
        return "foundation"
    if top in JOBS:
        return "jobs"
    if top in FLOWS:
        return "flows"
    raise AssertionError(f"{UNIT}.{inner} has no tier: add it to the plan's table and here")


def _edge_allowed(src: str, dst: str) -> bool:
    ts, td = _tier(src), _tier(dst)
    s_top, d_top = src.split(".")[0], dst.split(".")[0]
    if ts == "foundation" and td == "foundation":
        return (s_top, d_top) in FOUNDATION_EDGES
    if ts == "discovery":
        return td in ("foundation", "discovery") or d_top in DISCOVERY_JOBS
    if ts == "jobs" and td == "sources":
        return dst in JOB_READS_OF_SOURCES.get(s_top, set())
    if td == "discovery":
        return ts in ("discovery", "sources", "flows")
    return RANK[td] < RANK[ts] or ts == td


def unit_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for rel in _unit_files():
        me = _module_of(rel)
        if me == UNIT:
            continue
        graph[_inner(me)] = {_inner(m) for _ln, m in _imported(rel, _parse(rel))
                             if m.startswith(UNIT + ".")}
    return graph


def arrows_against_the_tiers() -> list[str]:
    return [f"{UNIT}.{src} imports {UNIT}.{dst}" for src, dsts in sorted(unit_graph().items())
            for dst in sorted(dsts) if src != dst and not _edge_allowed(src, dst)]


def cycles() -> list[list[str]]:
    graph = unit_graph()
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    found: list[list[str]] = []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(sorted(comp))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return found


def banned_outside_imports() -> list[str]:
    bad = []
    for rel in _unit_files():
        for line, mod in _imported(rel, _parse(rel)):
            banned = any(mod == b or mod.startswith(b + ".") for b in BANNED_OUTSIDE)
            core = mod.startswith("opyt_core") and mod not in OPYT_CORE_ALLOWED
            if banned or core:
                bad.append(f"{rel}:{line} imports {mod}")
    return bad


def reaches_into_the_unit(rels: list[str], allowed: set[str]) -> list[str]:
    return [f"{rel}:{line} imports {mod}" for rel in rels
            for line, mod in _imported(rel, _parse(rel))
            if (mod == UNIT or mod.startswith(UNIT + ".")) and mod not in allowed]


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in _unit_files():
        for fn in ast.walk(_parse(rel)):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bad.append(f"{rel}:{node.lineno} imports inside {fn.name}()")
    return bad


def test_every_arrow_inside_the_unit_points_down_a_tier():
    assert arrows_against_the_tiers() == []


def test_the_unit_has_no_cycle():
    assert cycles() == []


def test_the_unit_imports_nothing_it_sits_below():
    assert banned_outside_imports() == []


def test_the_library_half_and_the_worker_import_nothing_from_the_unit():
    worker = sorted(str(p.relative_to(ROOT)) for p in (ROOT / WORKER).rglob("*.py"))
    assert reaches_into_the_unit([LIBRARY_HALF, *worker], set()) == []


def test_the_tool_reaches_the_unit_only_through_its_flows():
    assert reaches_into_the_unit([TOOL], TOOL_REACHES) == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_every_package_init_stays_empty():
    inits = [rel for rel in _unit_files() if rel.endswith("__init__.py")
             and (ROOT / rel).read_text(encoding="utf-8").strip()]
    assert inits == []


def test_the_modules_the_restructure_deleted_stay_deleted():
    assert [rel for rel in RETIRED if (ROOT / rel).exists()] == []


def test_the_generic_names_it_retired_stay_retired():
    """Too generic for a guard on call names (`is_oracle` is curation's own; a check's status is a
    word every loop uses), so held here on the modules that had them."""
    from pipeline.oracles import checks, roster

    assert not hasattr(roster, "is_oracle"), "roster.on_roster answers that, inside the read's tx"
    assert not hasattr(checks, "check_status"), "a check's state is progress's to report (Q-O8)"
