"""The frontier unit's import-direction contract (S3, S5, S7) — the "Allowed imports" table,
checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down and the unit has no
    cycle;
  • the paper and GitHub transport the unit handed to platforms imports nothing of frontier, so
    platforms stays below it;
  • outside the unit, only the files named in IMPORTERS import `pipeline.frontier`: the
    foundations below (the store, the writes, the worker) never reach up into it;
  • the modules retired from the unit stay deleted;
  • imports sit at the top of the file, and `pipeline/frontier/__init__.py` stays empty.

Relative imports are resolved, as in tests/test_server_import_contract.py.
Changing the table is changing the design, not a way to make a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")

F = "pipeline.frontier."
_LOG, _TIME, _SCHEMA, _WRITES = ("pipeline.ingestion.utils", "pipeline.timeparse",
                                 "pipeline.kb.schema", "pipeline.kb.writes")
_REGISTRY = "pipeline.platforms.papers.registry"
_LOOP = "pipeline.worker.quota_loop."
ALLOWED: dict[str, set[str]] = {
    # ── the leaves: decisions and stores that import nothing of the unit ──
    "pipeline/frontier/intake.py": set(),
    "pipeline/frontier/cadence.py": {_TIME},
    "pipeline/frontier/candidates.py": {"pipeline.kb.id_batches", _WRITES},
    "pipeline/frontier/pulls.py": {_WRITES, _TIME},
    "pipeline/frontier/queries.py": set(),
    "pipeline/frontier/chain.py": {"pipeline.worker.job_queue", "pipeline.worker.registry"},
    "pipeline/frontier/allowance.py": set(),
    # ── jobs ──
    "pipeline/frontier/admit.py": {F + "candidates", _LOG, "pipeline.kb.ingest_common",
                                   "pipeline.kb.ingest_github", "pipeline.kb.ingest_papers",
                                   "pipeline.kb.ingest_x", "pipeline.kb.ingest_substack",
                                   "pipeline.platforms.papers.lookup", _SCHEMA, _WRITES},
    # ── the source contract, its five implementations, and the registry ──
    "pipeline/frontier/source_search.py": {F + "candidates"},
    "pipeline/frontier/arxiv_candidates.py": {F + "candidates", "pipeline.platforms.papers.arxiv"},
    "pipeline/frontier/github_candidates.py": {"pipeline.platforms.github.search",
                                               F + "candidates"},
    # OpenAlex asks for a landscape's works in a different order from a pull's, which is the
    # unit's one window-policy leaf deciding (`cadence.landscape`).
    "pipeline/frontier/openalex_candidates.py": {F + "cadence", F + "candidates",
                                                 "pipeline.platforms.papers.openalex", _REGISTRY},
    # X and Substack X searches on the user's session
    # (`sessions`) and reads X's dates with the renderer's parse; Substack finds a post's mention
    # of the name with the name check's own reading (`intake.mention`).
    "pipeline/frontier/x_candidates.py": {F + "candidates", "pipeline.ingestion.x_render",
                                          "pipeline.platforms.sessions",
                                          "pipeline.platforms.x.search"},
    "pipeline/frontier/substack_candidates.py": {F + "candidates", F + "intake", _TIME,
                                                 "pipeline.platforms.substack.search"},
    "pipeline/frontier/adapters.py": {F + "arxiv_candidates", F + "github_candidates",
                                      F + "openalex_candidates", F + "x_candidates",
                                      F + "substack_candidates", F + "source_search"},
    # ── searching now: the side-by-side search a topic's preview and `look` share, and `look`'s
    #    flow. A find's card carries the `cite` link
    #    every search hit carries. ──
    "pipeline/frontier/lanes.py": {F + "candidates", F + "intake", F + "source_search", _LOG,
                                   "pipeline.retrieval.cards"},
    # `look` shows a landscape in the order the sources ranked it and news newest first, the same
    # line `cadence` draws for OpenAlex's own order, so the two cannot disagree.
    "pipeline/frontier/look.py": {F + "adapters", F + "cadence", F + "lanes", F + "queries"},
    # ── the two background stages: sources on the scrapers process ──
    "pipeline/frontier/searching.py": {F + "adapters", F + "allowance", F + "cadence",
                                       F + "candidates", F + "intake", F + "pulls", F + "queries",
                                       F + "source_search", "pipeline.platforms.errors", _LOG,
                                       _WRITES, _TIME,
                                       _LOOP + "retries", _LOOP + "source"},
    "pipeline/frontier/admitting.py": {F + "admit", F + "allowance", F + "candidates", _LOG,
                                       "pipeline.kb.embed", _SCHEMA, _WRITES, _TIME,
                                       _LOOP + "retries", _LOOP + "source"},
}
INIT = "pipeline/frontier/__init__.py"
# The transport the unit hands to platforms. It returns records; frontier maps them.
PLATFORMS = ("pipeline/platforms/papers/registry.py", "pipeline/platforms/papers/arxiv.py",
             "pipeline/platforms/papers/openalex.py", "pipeline/platforms/github/search.py",
             "pipeline/platforms/github/api.py", "pipeline/platforms/errors.py",
             "pipeline/platforms/papers/lookup.py", "pipeline/platforms/papers/pdf.py",
             "pipeline/platforms/x/search.py", "pipeline/platforms/substack/search.py")
# Outside the package, the only files that may import `pipeline.frontier`: the scrapers process's
# composition (which runs the two background stages); topics,
# whose watches become standing queries that wake the searches, whose preview runs the adapters
# and the name check the pulls use, and whose check records verdicts on finds; What's New
# (`latest` and its notice), which hands the waiting finds to the host; and `look`'s tool, which
# calls its flow.
IMPORTERS = {"pipeline/background.py",
             *(f"pipeline/topics/{m}.py" for m in ("store", "lists", "create", "preview",
                                                    "checked")),
             "pipeline/arrivals/show.py", "pipeline/arrivals/notice.py",
             "mcp_server/look_tools.py"}
# Retired modules; their code lives in pipeline/frontier/ and platforms'
# pipeline/platforms/papers/{registry,arxiv,openalex}.py. The two jobs' work is `searching.py` and
# `admitting.py`. The queue, its ranking and its tool are replaced by `latest`
# (pipeline/arrivals/). A topic owns its watches (`pipeline/topics/store.py`).
RETIRED = [*(f"pipeline/kb/frontier_{m}.py"
             for m in ("surface", "queries", "watchlist", "sources", "execute", "admit")),
           *(f"pipeline/frontier/{m}.py"
             for m in ("surface", "ranking", "events", "duplicates", "watchlist")),
           "mcp_server/frontier_tools.py"]


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


def _repository_files(*tops: str) -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for top in tops for p in (ROOT / top).rglob("*.py"))


def _is_frontier(mod: str) -> bool:
    return mod == F.rstrip(".") or mod.startswith(F)


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def platforms_reaching_up() -> list[str]:
    bad = []
    for rel in PLATFORMS:
        for line, mod in _imported(rel, _parse(rel)):
            if _is_frontier(mod):
                bad.append(f"{rel}:{line} imports {mod}: platforms' transport returns records "
                           f"and imports nothing of frontier")
    return bad


def reaches_into_the_unit() -> list[str]:
    bad = []
    for rel in _repository_files(*INTERNAL):
        if rel.startswith("pipeline/frontier/"):
            continue
        for line, mod in _imported(rel, _parse(rel)):
            if _is_frontier(mod) and rel not in IMPORTERS:
                bad.append(f"{rel}:{line} imports {mod}: only the files in IMPORTERS import "
                           f"pipeline.frontier")
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
    unit = set(_repository_files("pipeline/frontier")) - {INIT}
    assert sorted(unit - set(ALLOWED)) == []


def test_platforms_transport_imports_nothing_of_frontier():
    assert platforms_reaching_up() == []


def test_nothing_outside_the_unit_reaches_into_it_but_its_named_importers():
    assert reaches_into_the_unit() == []


def test_the_modules_the_restructure_deleted_stay_deleted():
    assert [rel for rel in RETIRED if (ROOT / rel).exists()] == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_the_package_init_stays_empty():
    assert (ROOT / INIT).read_text(encoding="utf-8").strip() == "", f"{INIT} must stay empty"
