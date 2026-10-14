"""The onboarding unit's import-direction contract (S3, S5, S7) — the "Allowed imports" table of
docs/plans/2026-10-05-restructure-onboarding.md, checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down (`flow` → the
    steps → `roots` / `worker_check` → other units) and the unit has no cycle;
  • outside the unit, only the entry (`mcp_server/onboard_tools.py`) imports
    `pipeline.onboarding`;
  • the module the restructure deleted stays deleted, and the unit defines none of the names it
    retired (.guards.py's `retired-onboarding-names` bans calling them anywhere);
  • imports sit at the top of the file, and `pipeline/onboarding/__init__.py` stays empty.

Relative imports are resolved, as in tests/sharing/test_import_contract.py, which this mirrors.
Changing the table is changing the design: do it with the plan, not to make a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "gateway", "config")

O = "pipeline.onboarding."
ENTRY = "mcp_server/onboard_tools.py"
ALLOWED: dict[str, set[str]] = {
    # ── the entry ──
    ENTRY: {O + "flow"},
    # ── the flow ──
    "pipeline/onboarding/flow.py": {O + "phase", O + "roots", O + "connect", O + "kickoff",
                                    O + "handoff", O + "worker_check", O + "optional_keys"},
    # ── the steps ──
    "pipeline/onboarding/phase.py": {O + "roots", "opyt_core.paths",
                                     "pipeline.curation.collectors", "pipeline.kb.schema",
                                     "pipeline.topics.store",
                                     # platforms' session checks (QP2)
                                     "pipeline.platforms.connections"},
    "pipeline/onboarding/connect.py": {O + "roots", O + "service_signin",
                                       # the marker a local sign-in leaves (2026-10-13)
                                       "opyt_core.paths",
                                       "pipeline.platforms.browser.login",
                                       "pipeline.ingestion.utils",
                                       "pipeline.platforms.connections"},
    "pipeline/onboarding/kickoff.py": {O + "roots", "pipeline.curation.imports",
                                       "pipeline.worker.job_queue", "pipeline.worker.registry"},
    "pipeline/onboarding/optional_keys.py": {"opyt_core.github_device", "opyt_core.key_page",
                                             "opyt_core.keys", "opyt_core.credentials_registry",
                                             "opyt_core.paths", "pipeline.credentials",
                                             "pipeline.kb.schema"},
    "pipeline/onboarding/handoff.py": {O + "roots", O + "worker_check", "pipeline.kb.schema",
                                       "pipeline.oracles.ranking",
                                       # the offer to read the library from claude.ai (2026-10-12)
                                       "pipeline.sharing.account"},
    # X through the sign-in Opyt's service runs, off macOS (2026-10-12)
    "pipeline/onboarding/service_signin.py": {"opyt_core.keys", "opyt_core.paths",
                                              "pipeline.sharing.account",
                                              "pipeline.sharing.client"},
    # ── the leaves of the unit ──
    "pipeline/onboarding/roots.py": {"pipeline.platforms.browser.cookies"},
    "pipeline/onboarding/worker_check.py": {"pipeline.worker.liveness"},
}
INIT = "pipeline/onboarding/__init__.py"
# Outside the unit, the files that may import it: the entry, where server registers the tool.
# Since 2026-10-12 also the rider flow, which puts the optional-keys question on the answer to the
# first Oracle confirmation (`optional_keys.question`).
IMPORTERS = {ENTRY, "mcp_server/riders.py"}
# Deleted by the restructure (2026-10-05); its code lives in pipeline/onboarding/phase.py.
RETIRED = ["pipeline/kb/onboard_state.py"]
# Names the restructure retired, which no file of the unit may define again.
RETIRED_NAMES = {"_onboard", "_handoff", "_phase_sources", "_connect_step", "_ensure_worker",
                 "_worker_now", "_local_unsupported", "_report_worker", "_connected_x",
                 "_connected_substack", "mark_sources_skipped", "live_platforms", "now", "named"}
# Imports inside functions the entry keeps. None since the 2026-10-11 free tier: model access's
# half imports at the top like the rest. A new one fails.
PARKED_INNER: set[tuple[str, str]] = set()


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


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def reaches_into_the_unit() -> list[str]:
    bad = []
    for rel in _repository_files(*INTERNAL):
        if rel.startswith("pipeline/onboarding/") or rel in IMPORTERS:
            continue
        for line, mod in _imported(rel, _parse(rel)):
            if mod == O.rstrip(".") or mod.startswith(O):
                bad.append(f"{rel}:{line} imports {mod}: only {ENTRY} imports pipeline.onboarding")
    return bad


def retired_names_defined() -> list[str]:
    return [f"{rel}:{node.lineno} defines {node.name}"
            for rel in ALLOWED for node in ast.walk(_parse(rel))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in RETIRED_NAMES]


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in ALLOWED:
        for fn in ast.walk(_parse(rel)):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (rel, fn.name) in PARKED_INNER:
                continue
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bad.append(f"{rel}:{node.lineno} imports inside {fn.name}()")
    return bad


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_every_module_of_the_unit_has_a_row():
    unit = set(_repository_files("pipeline/onboarding")) - {INIT}
    assert sorted(unit - set(ALLOWED)) == []


def test_nothing_outside_the_unit_reaches_into_it_but_its_entry():
    assert reaches_into_the_unit() == []


def test_the_module_the_restructure_deleted_stays_deleted():
    assert [rel for rel in RETIRED if (ROOT / rel).exists()] == []


def test_the_names_the_restructure_retired_stay_retired():
    assert retired_names_defined() == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_the_package_init_stays_empty():
    assert (ROOT / INIT).read_text(encoding="utf-8").strip() == "", f"{INIT} must stay empty"
