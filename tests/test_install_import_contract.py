"""The install unit's import-direction contract: the "Allowed imports" table below,
checked against the code.

What a reader of the unit relies on, and what an edit can break without failing any behaviour
test:
  • every module imports only what its row allows, so every arrow points down and the unit has
    no cycle. install sits below every unit that calls it (worker, onboarding, server,
    model-access, sharing) and imports none of them: no `pipeline`, `mcp_server` or `service`,
    and none of model-access's `opyt_core` modules;
  • no install module imports inside a function (none of them needs an optional-library
    exception);
  • the retired names are defined nowhere in shipping code, and `uvx_command`
    and `running_distribution` are defined only in `opyt_core/launcher.py`. Calls and imports
    of the retired names are also banned repo-wide by `.guards.py` (`retired-install-names`,
    `retired-install-import-paths`);
  • the fail-safe owner-only chmod has one home, `opyt_core/paths.py::owner_only`.

Changing the table is changing the design: do it deliberately, not to make a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "config")
SHIPPING = ("pipeline", "mcp_server", "opyt_core", "service", "config")

ALLOWED: dict[str, set[str]] = {
    "opyt_core/launcher.py": set(),
    "opyt_core/install_worker.py": {"opyt_core.launcher", "opyt_core.paths"},
    "opyt_core/install_client.py": {"opyt_core.install_worker", "opyt_core.launcher"},
    "opyt_core/keys.py": {"opyt_core.credentials_registry", "opyt_core.paths"},
    "opyt_core/bootstrap.py": {"opyt_core.config", "opyt_core.install_worker", "opyt_core.paths"},
}

# Retired names. Each name is distinctive repo-wide.
RETIRED_NAMES = {"repair_worker", "validate_credential", "_validate_github", "_validate_s2",
                 "_validate_opyt_service_token"}
# Moved into launcher.py: a second definition would be the old two-file shape coming back.
LAUNCHER_ONLY = {"uvx_command", "running_distribution"}


def _module_of(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def _parse(rel: str) -> ast.AST:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)


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


def _shipping_files():
    for top in SHIPPING:
        for path in sorted((ROOT / top).rglob("*.py")):
            yield str(path.relative_to(ROOT))


def violations_of_the_table() -> list[str]:
    return [f"{rel}:{line} imports {mod}, which its row does not allow"
            for rel, allowed in ALLOWED.items()
            for line, mod in _imported(rel, _parse(rel)) if mod not in allowed]


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in ALLOWED:
        for fn in ast.walk(_parse(rel)):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bad += [f"{rel}:{node.lineno} imports inside {fn.name}()"
                        for node in ast.walk(fn) if isinstance(node, (ast.Import, ast.ImportFrom))]
    return bad


def misplaced_definitions() -> list[str]:
    bad = []
    for rel in _shipping_files():
        for node in ast.walk(_parse(rel)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name in RETIRED_NAMES:
                bad.append(f"{rel}:{node.lineno} defines {node.name}, which is retired")
            elif node.name in LAUNCHER_ONLY and rel != "opyt_core/launcher.py":
                bad.append(f"{rel}:{node.lineno} defines {node.name}; it lives in launcher.py")
    return bad


def chmod_outside_owner_only() -> list[str]:
    return [f"{rel}:{node.lineno} calls .chmod(); use opyt_core.paths.owner_only"
            for rel in _shipping_files() if rel != "opyt_core/paths.py"
            for node in ast.walk(_parse(rel))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "chmod"]


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_no_install_module_imports_inside_a_function():
    assert imports_inside_functions() == []


def test_the_names_the_restructure_retired_stay_retired():
    assert misplaced_definitions() == []


def test_owner_only_is_the_one_chmod():
    assert chmod_outside_owner_only() == []


def test_keys_re_exports_nothing():
    """`keys.KNOWN` re-exported the registry's list (S3); readers import the registry."""
    tree = _parse("opyt_core/keys.py")
    bound = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names}
    assert "KNOWN" not in bound
