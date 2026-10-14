"""The worker unit's import-direction contract (S3, S5, S7) — the "Allowed imports" table of
docs/plans/2026-10-04-restructure-worker.md ("2. Target"), checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down, the unit has no
    cycle, and the worker imports no domain (oracles, curation, frontier, research, sharing,
    retrieval, library, onboarding): each domain hands its work in;
  • the names the restructure retired stay retired: they are defined nowhere in the unit, so the
    old shape cannot come back beside the new one;
  • imports sit at the top of the file, and both `__init__.py` files stay empty.

Relative imports are resolved, as in tests/kb/test_frontier_import_contract.py, which this
mirrors. The retired names that are distinctive repo-wide are also guarded by `.guards.py`
(`retired-worker-names`, `retired-worker-module-names`); the generic ones (`pick`, `execute`,
`_wait`, …) are checked here, as definitions in the unit, where a repo-wide ban would hit
unrelated code. Changing the table is changing the design: do it with the plan, not to make a
red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")

W, Q = "pipeline.worker.", "pipeline.worker.quota_loop."
_LOG, _PATHS, _WRITES = "pipeline.ingestion.utils", "opyt_core.paths", "pipeline.kb.writes"
_X_ORIGINALS, _X_RATE = "pipeline.platforms.x.originals", "pipeline.platforms.x.rate"
ALLOWED: dict[str, set[str]] = {
    # ── the leaves ──
    "pipeline/worker/exit_codes.py": set(),
    "pipeline/worker/failure_run.py": set(),
    "pipeline/worker/registry.py": set(),
    "pipeline/worker/home.py": {_PATHS},
    # ── the control database, and the processes around it ──
    "pipeline/worker/job_queue.py": {W + "home", _PATHS, _LOG},
    "pipeline/worker/child.py": {W + "exit_codes", _PATHS},
    "pipeline/worker/liveness.py": {W + "job_queue", W + "home", "opyt_core.install_worker",
                                    _PATHS},
    "pipeline/worker/supervisor.py": {W + "registry", W + "job_queue", W + "home",
                                      W + "failure_run", _PATHS, _LOG},
    # ── the quota loop ──
    "pipeline/worker/quota_loop/source.py": set(),
    "pipeline/worker/quota_loop/loop_state.py": {_WRITES, "pipeline.loops_db"},
    "pipeline/worker/quota_loop/retries.py": {Q + "source", _WRITES, "pipeline.timeparse",
                                              "pipeline.circuit_breaker", _LOG,
                                              "pipeline.platforms.browser.profile", _X_ORIGINALS,
                                              _X_RATE},
    "pipeline/worker/quota_loop/choose.py": {Q + "source"},
    "pipeline/worker/quota_loop/loop.py": {Q + "source", Q + "choose", Q + "retries",
                                           Q + "loop_state", "pipeline.sync_lock", _X_RATE,
                                           _LOG},
    "pipeline/worker/quota_loop/serial.py": {Q + "loop", Q + "retries", Q + "source", _LOG},
    "pipeline/worker/quota_loop/pool.py": {Q + "loop", Q + "choose", Q + "retries", Q + "source",
                                           "pipeline.host_access"},
    "pipeline/worker/quota_loop/process.py": {Q + "loop", Q + "loop_state", W + "job_queue",
                                              W + "registry", W + "failure_run", _LOG},
}
INITS = ("pipeline/worker/__init__.py", "pipeline/worker/quota_loop/__init__.py")

# Retired by the restructure and its review rounds (2026-10-04/05), as `(file, scope, name)`:
# no function, method, class or attribute of that name may be defined again in that file, inside
# that class (`scope`), or anywhere in the file (`scope` None).
_LOOP, _PROCESS = "pipeline/worker/quota_loop/loop.py", "pipeline/worker/quota_loop/process.py"
RETIRED_DEFINITIONS = [
    # third review B2: an `Executor` (`Serial`, `WebPool`) runs the units, so the loop lost its
    # own executor methods (the Protocol's `units` and `execute` are the executors'), and the
    # one-class `Loop` Protocol went
    (_LOOP, "QuotaLoop", "pick"),
    (_LOOP, "QuotaLoop", "units"),
    (_LOOP, "QuotaLoop", "execute"),
    (_PROCESS, None, "Loop"),
    # third review A1, A2: a wait is a value recorded by `_record_wait`; a unit settles in `settle`
    (_LOOP, "QuotaLoop", "_wait"),
    (_LOOP, "QuotaLoop", "finish_unit"),
    (_LOOP, "QuotaLoop", "_fetch_raised"),
    # third review B4: `Wait` lives in `retries`
    ("pipeline/worker/quota_loop/source.py", None, "Wait"),
    # third review D1: `stopping` is the one stop signal
    (_LOOP, "QuotaLoop", "stopped"),
    # third review A4, diff check A4: the row's read, re-arm and wake is `_rearm_and_wake`
    (_PROCESS, "LoopProcess", "_watch"),
    (_PROCESS, "LoopProcess", "_read_row"),
    # third review A6: `FailureRun` holds a run of failures
    ("pipeline/worker/supervisor.py", None, "failing_since"),
    (_PROCESS, None, "failing_since"),
    # diff check A4 and second diff check G3: names that say what the method does
    (_LOOP, "QuotaLoop", "wait_idle"),
    (_LOOP, "QuotaLoop", "back_off"),
    (_LOOP, "QuotaLoop", "_cut_short"),
    (_LOOP, "QuotaLoop", "_forgotten_or_lease_lost"),
    (_LOOP, "QuotaLoop", "_purge_forgotten_or_end_lost"),
    # the cleanup after lanes: a job's every exit earns its cadence, and `child.main` is
    # the one way a job starts
    ("pipeline/worker/registry.py", "RailSpec", "prompt_cadence"),
    ("pipeline/worker/registry.py", "RailSpec", "rate_paused_cadence"),
    ("pipeline/worker/registry.py", "RailSpec", "exit_code"),
    ("pipeline/worker/registry.py", None, "next_run_in"),
    ("pipeline/worker/registry.py", None, "SIX_HOURLY"),
    ("pipeline/worker/registry.py", None, "X_RATE_MARGIN"),
    ("pipeline/worker/registry.py", None, "X_RATE_MIN_WAIT"),
    ("pipeline/worker/exit_codes.py", None, "EXIT_MORE_WORK"),
    ("pipeline/worker/exit_codes.py", None, "EXIT_RATE_PAUSED"),
    ("pipeline/worker/supervisor.py", None, "_x_reset_in"),
    ("pipeline/worker/child.py", None, "run_once"),
    # the resident split (2026-10-06), deleted when `scrapers` was the only job. A job runs beside
    # the others again since the line (2026-10-13), as `RailSpec.beside` and
    # `supervisor.launchable`, which has no local/hosted split; these names stay retired
    ("pipeline/worker/registry.py", None, "RESIDENT"),
    ("pipeline/worker/supervisor.py", None, "claimable"),
    ("pipeline/worker/supervisor.py", "RailWorker", "stop_resident"),
]


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
    return {str(p.relative_to(ROOT)) for p in (ROOT / "pipeline/worker").rglob("*.py")}


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def _defined(tree: ast.AST, scope: str | None) -> set[str]:
    """Every name defined in `scope` (a class of the file; None: the whole file): functions,
    methods, classes, assignments, and attributes set on `self`."""
    if scope is not None:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == scope)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Attribute):
                    names.add(t.attr)
    return names


def retired_names_defined() -> list[str]:
    return [f"{rel} defines {scope + '.' if scope else ''}{name}, which the restructure retired"
            for rel, scope, name in RETIRED_DEFINITIONS if name in _defined(_parse(rel), scope)]


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
    assert sorted(_unit_files() - set(ALLOWED) - set(INITS)) == []


def test_the_names_the_restructure_retired_stay_retired():
    assert retired_names_defined() == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_the_package_inits_stay_empty():
    for init in INITS:
        assert (ROOT / init).read_text(encoding="utf-8").strip() == "", f"{init} must stay empty"
