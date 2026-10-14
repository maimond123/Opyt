"""The platforms unit's import-direction contract: the "Allowed imports" table below,
checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down, the unit has no
    cycle, and platforms imports no domain (library, curation, frontier, onboarding, worker,
    kb): it sits below every unit and imports only the foundations below and its
    own modules. One edge leaves the unit by name: x/originals → the split renderer x_render;
  • an import inside a function is only an exception (an optional third-party library) and
    carries its reason on the line above;
  • the retired names are defined nowhere in the code that ships; calls to them
    are also banned repo-wide by `.guards.py` (`retired-platform-names`);
  • every `__init__.py` of the unit stays empty.

Relative imports are resolved, as in tests/kb/test_worker_import_contract.py, which this
mirrors. Changing the table is changing the design: do it with the plan, not to make a red test
green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")
SHIPPING = ("pipeline", "mcp_server", "opyt_core", "service", "config")

# The foundations every row may import: the ones the unit actually uses (Step 7 enumerates them).
FOUNDATIONS = {"opyt_core.paths", "pipeline.ingestion.utils", "pipeline.host_access",
               "pipeline.credentials", "pipeline.circuit_breaker"}

P = "pipeline.platforms."
B, X, S, PA = P + "browser.", P + "x.", P + "substack.", P + "papers."
_ENDPOINT = {X + "request", X + "records", X + "rate"}
ALLOWED: dict[str, set[str]] = {
    "pipeline/platforms/collection_page.py": set(),
    "pipeline/platforms/url_canon.py": set(),
    "pipeline/platforms/identity_tokens.py": {P + "url_canon"},
    "pipeline/platforms/source_classify.py": {P + "url_canon"},
    "pipeline/platforms/errors.py": set(),
    # ── browser ──
    "pipeline/platforms/browser/cdp.py": set(),
    "pipeline/platforms/browser/cookies.py": {B + "cdp"},
    "pipeline/platforms/browser/login.py": {B + "cookies", B + "cdp"},
    "pipeline/platforms/browser/profile.py": set(),
    "pipeline/platforms/browser/render.py": {B + "cdp", B + "cookies"},
    # ── X ──
    "pipeline/platforms/x/contracts.py": set(),
    "pipeline/platforms/x/query_ids.py": set(),
    "pipeline/platforms/x/rate.py": set(),
    "pipeline/platforms/x/records.py": {P + "collection_page"},
    "pipeline/platforms/x/local.py": {X + "query_ids", X + "contracts", X + "rate",
                                      B + "cookies"},
    "pipeline/platforms/x/request.py": {X + "contracts", X + "query_ids", X + "rate"},
    "pipeline/platforms/x/collections.py": {P + "sessions", X + "request", X + "records",
                                            X + "contracts", P + "collection_page"},
    "pipeline/platforms/x/originals.py": _ENDPOINT | {"pipeline.ingestion.x_render"},
    "pipeline/platforms/x/conversation.py": _ENDPOINT,
    "pipeline/platforms/x/profile.py": _ENDPOINT,
    # SearchTimeline: it asks the local transport whether it can run.
    "pipeline/platforms/x/search.py": _ENDPOINT | {X + "contracts", X + "local",
                                                   P + "collection_page"},
    # ── Substack ──
    "pipeline/platforms/substack/contracts.py": set(),
    "pipeline/platforms/substack/http.py": {S + "contracts", B + "cookies"},
    "pipeline/platforms/substack/publication.py": {S + "http", S + "contracts", B + "cookies"},
    "pipeline/platforms/substack/account.py": {S + "contracts", S + "publication",
                                               P + "collection_page"},
    "pipeline/platforms/substack/local.py": {S + "http", S + "contracts", S + "account",
                                             S + "publication", B + "cookies"},
    # Post search, signed out.
    "pipeline/platforms/substack/search.py": {S + "http", S + "contracts", S + "publication"},
    # ── the readers callers use ──
    "pipeline/platforms/sessions.py": {X + "request", X + "local", S + "local", S + "contracts"},
    "pipeline/platforms/connections.py": {X + "local", S + "local"},
    # ── web, GitHub, papers ──
    "pipeline/platforms/web/links.py": set(),
    "pipeline/platforms/web/discover.py": {P + "web.links", P + "url_canon"},
    "pipeline/platforms/web/article.py": {B + "render"},
    "pipeline/platforms/web/page_head.py": {B + "render"},     # card fill
    "pipeline/platforms/github/api.py": {P + "errors"},
    "pipeline/platforms/github/search.py": {P + "github.api", P + "errors"},
    "pipeline/platforms/papers/registry.py": {P + "errors"},
    "pipeline/platforms/papers/arxiv.py": {PA + "registry", P + "errors"},
    "pipeline/platforms/papers/openalex.py": {PA + "registry", P + "errors"},
    "pipeline/platforms/papers/lookup.py": {PA + "registry", PA + "arxiv", PA + "openalex"},
    "pipeline/platforms/papers/pdf.py": {PA + "arxiv"},
    "pipeline/platforms/papers/landing.py": set(),
}
INITS = tuple(f"pipeline/platforms/{d}__init__.py"
              for d in ("", "browser/", "x/", "substack/", "web/", "github/", "papers/"))

# Retired names: defined nowhere in shipping code. Each name is
# distinctive repo-wide, so the whole tree is checked, not only the unit.
RETIRED_NAMES = {
    "has_managed_x_session", "has_managed_substack_session",   # connections' probes
    "originals_cursor", "PageYield",                           # the Originals multi-page walk
    "openalex_progress", "WalkStep",                           # OpenAlex works()' cursor walk
    "seconds_until_reset",                                     # S7 amendment
    "record_rate_headers",                                     # Step 6 R2
    "_parse_twitter_date",                                     # Step 6 R3: parse_twitter_date
    "_HostedFollows", "_HostedSubscriptionList", "_HostedSaved",   # forwarding wrappers
    "select_backend_family", "BackendFamily",                  # the hosted selector (2026-10-14)
}
# The hosted transports went with the gateway (2026-10-14): onboarding is local.
RETIRED_MODULES = ("pipeline/ingestion/hosted_browser.py",
                   "pipeline/platforms/browser/hosted.py", "pipeline/platforms/browser/request.py",
                   "pipeline/platforms/x/hosted.py", "pipeline/platforms/substack/hosted.py")


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


def _unit_files() -> set[str]:
    return {str(p.relative_to(ROOT)) for p in (ROOT / "pipeline/platforms").rglob("*.py")}


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed | FOUNDATIONS:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def unexcused_imports_inside_functions() -> list[str]:
    """An import inside a function must be a third-party library with its reason in a comment
    on the line above (the optional-library exception)."""
    bad = []
    for rel in ALLOWED:
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        for fn in ast.walk(_parse(rel)):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""])
                if any(n.split(".")[0] in INTERNAL or not n for n in names):
                    bad.append(f"{rel}:{node.lineno} imports repository code inside {fn.name}()")
                elif not lines[node.lineno - 2].strip().startswith("#"):
                    bad.append(f"{rel}:{node.lineno} imports inside {fn.name}() with no reason "
                               f"on the line above")
    return bad


def retired_names_defined() -> list[str]:
    bad = []
    for top in SHIPPING:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            for node in ast.walk(_parse(rel)):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name in RETIRED_NAMES):
                    bad.append(f"{rel}:{node.lineno} defines {node.name}, which the "
                               f"platforms unit retired")
    return bad


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_every_module_of_the_unit_has_a_row():
    assert sorted(_unit_files() - set(ALLOWED) - set(INITS)) == []


def test_imports_inside_functions_are_only_optional_libraries_with_a_reason():
    assert unexcused_imports_inside_functions() == []


def test_the_names_the_restructure_retired_stay_retired():
    assert retired_names_defined() == []


def test_the_retired_modules_stay_deleted():
    assert [m for m in RETIRED_MODULES if (ROOT / m).exists()] == []


def test_the_package_inits_stay_empty():
    for init in INITS:
        assert (ROOT / init).read_text(encoding="utf-8").strip() == "", f"{init} must stay empty"
