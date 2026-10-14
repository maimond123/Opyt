"""The server unit's import-direction contract — the "Allowed imports" table below,
checked against the code.

Three things a reader of the unit relies on, each of which an edit can break without failing any
behaviour test:
  • every module imports only what its row allows, so every arrow points down: server → build →
    hook → riders → (rules, offers, each owner's producer);
  • nothing below reaches back up — no owner or tool module imports the server's own modules, and
    retrieval (`pipeline/retrieval/`) imports no Oracle freshness or what-next;
  • imports sit at the top of the unit's files, so the import line names the file to open.

Standard-library and third-party imports are not policed here; only this repository's packages
are. `build.register_optional` loads each tool module by name (`importlib`), an optional
exception, so the tool modules are not rows of `build.py`. `line.py` imports `build` inside one
function, a heavy exception (`NESTED`).

Changing the table is changing the design: do it deliberately, not to make a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")

M = "mcp_server."
ALLOWED: dict[str, set[str]] = {
    "mcp_server/server.py": {M + "build", "opyt_core.bootstrap"},
    # The shell entry (`opyt`), for an agent with no MCP client (such as a VM).
    "mcp_server/cli.py": {M + "build", "opyt_core.bootstrap"},
    # The worker's `line` job, which runs the tool calls the web and agents ask for
    # The worker names it by a string
    # (`registry.RAILS`), so nothing imports it.
    "mcp_server/line.py": {M + "build", "pipeline.sharing.account", "pipeline.sharing.client",
                           "pipeline.sharing.push_gate", "pipeline.worker.job_queue",
                           "pipeline.worker.registry", "pipeline.ingestion.utils"},
    "mcp_server/build.py": {M + "instructions", M + "rider_hook"},
    "mcp_server/instructions.py": set(),
    "mcp_server/rider_hook.py": {M + "riders", M + "rider_rules"},
    "mcp_server/riders.py": {M + "rider_rules", M + "offers", "pipeline.arrivals.notice",
                             "pipeline.model_routing",
                             "pipeline.curation.still_scoring", "pipeline.curation.imports",
                             "pipeline.retrieval.new_material", "pipeline.oracles.freshness",
                             "pipeline.kb.schema", "pipeline.onboarding.optional_keys",
                             "pipeline.oracles.milestones", "pipeline.oracles.told",
                             "pipeline.sharing.share_back", "pipeline.sharing.page_words",
                             "pipeline.timeparse"},
    "mcp_server/rider_rules.py": set(),
    "mcp_server/offers.py": {M + "rider_rules", "opyt_core.paths",
                             "pipeline.kb.derive", "pipeline.kb.schema",
                             "pipeline.oracles.mentions", "pipeline.oracles.people_map",
                             "pipeline.oracles.probe_store",
                             "pipeline.oracles.ranking", "pipeline.oracles.roster",
                             "pipeline.retrieval.corpus_census", "pipeline.retrieval.rank",
                             "pipeline.topics.store"},
}

# The server's own modules: only the rows above import them (server → build → hook → riders).
SERVER_OWN = {M + name for name in ("server", "build", "instructions",
                                    "rider_hook", "riders", "rider_rules", "offers", "line")}

# The functions that may import inside themselves, as `(file, function)`. `line._server` builds
# the server on the job's first call: an idle line that loaded FastMCP and every tool module
# would hold 112 MB instead of 47 (measured).
NESTED = {("mcp_server/line.py", "_server")}

# Retrieval depends only on the library: it never asks for Oracle freshness or what to do next.
RETRIEVAL = {f"pipeline/retrieval/{name}.py" for name in (
    "reads", "local", "served", "engine", "filters", "who", "notices", "cards", "rank", "peers",
    "corpus_census", "new_material")}
NOT_FOR_RETRIEVAL = {"pipeline.oracles.freshness", M + "offers"}


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


def _production_files() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for top in INTERNAL for p in (ROOT / top).rglob("*.py"))


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def reaches_back_up() -> list[str]:
    """Files below the server importing its own modules, and retrieval importing above itself."""
    bad = []
    for rel in _production_files():
        for line, mod in _imported(rel, _parse(rel)):
            if mod in SERVER_OWN and rel not in ALLOWED:
                bad.append(f"{rel}:{line} imports {mod}: the server's modules sit above it")
            if rel in RETRIEVAL and mod in NOT_FOR_RETRIEVAL:
                bad.append(f"{rel}:{line} imports {mod}: retrieval depends only on the library")
    return bad


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in ALLOWED:
        for fn in ast.walk(_parse(rel)):
            if (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and (rel, fn.name) not in NESTED):
                bad += [f"{rel}:{node.lineno} imports inside {fn.name}()" for node in ast.walk(fn)
                        if isinstance(node, (ast.Import, ast.ImportFrom))]
    return bad


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_nothing_below_reaches_back_up():
    assert reaches_back_up() == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_mcp_servers_init_stays_empty():
    assert (ROOT / "mcp_server/__init__.py").read_text(encoding="utf-8").strip() == ""
