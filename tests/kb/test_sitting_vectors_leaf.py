"""`sitting_vectors` is the leaf of the sitting subsystem — the real enforcement of that.

The module's own header states the invariant and names its consequence:

    Zero dependency on any other `sitting_*` module — this is the leaf every one of them reads
    chunk vectors through, so it has to stay that way or the import graph gets a cycle.

Until 2026-09-05 nothing enforced it. There was no `.guards.py` rule and no import-linter contract
anywhere in the repo, so the invariant's entire enforcement was that sentence — one refactor from
being false, in the file with the highest fan-in of the five (six modules read chunk vectors through
it: sitting_builder, sitting_render, sitting_store, sitting_zoom, sitting_scheduler, sitting_reader).
Found by the B7 review, docs/reviews/2026-09-05-b7-sittings.md, LEAD 1.

WHY THE CYCLE MATTERS, in one line: `scripts/codemap.py` partitions the codebase for review by
strongly-connected component, and a cycle is indivisible. One import in the wrong direction folds
all seven sitting modules into a single slice that has no first member, and the subsystem stops
being reviewable a module at a time.

The `.guards.py` rule this test is cross-referenced from (`sitting-vectors-stays-the-leaf`) is
line-based text and cannot see an import that is aliased or split across lines. This test parses the
real AST, matching the precedent set by `tests/kb/test_amendment3_guard.py` for
`sitting-membership-never-time-bounded`: the grep rule is the cheap tripwire on every commit, the
AST check is what the rule's message tells you to run before concluding an import-shaped edit is safe.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

TARGET = Path(__file__).resolve().parents[2] / "pipeline" / "kb" / "sitting_vectors.py"

# The module's ENTIRE import surface, measured 2026-09-05 — module path segments and imported names
# together, because `from pipeline.kb import sitting_store` hides the module in the NAME and no AST
# walk can tell that from `from .embed import stored_dtype`, which hides a function there. Pinning
# both is what makes the check work in the one direction that matters; the cost is that adding any
# import here is a deliberate edit to this line.
#
# `schema` and `embed` both sit BELOW this module: neither imports any sitting_* module, which is
# what makes them the legal place to put something `sitting_vectors` needs.
ALLOWED = {"__future__", "annotations", "numpy", "schema", "embed", "stored_dtype"}


def _import_surface(path: Path) -> set[str]:
    """Every module segment and imported name that appears in an import statement in `path`.

    Deliberately over-collects rather than resolving. `from pkg import x` cannot be told from
    `from mod import func` without importing the package, and this check has to work on a file it
    never executes — so it takes both and lets `ALLOWED` carry the precision.
    """
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out |= set(alias.name.split("."))     # `import a.b.c` — every segment is a module
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out |= set(node.module.split("."))
            # `.name`, never `.asname`: an alias renames the binding, not the thing imported.
            out |= {alias.name for alias in node.names}
    return out


def test_sitting_vectors_imports_no_sitting_sibling():
    """THE INVARIANT. Not "no cycle today" — no import in this direction at all, which is the
    property that makes a cycle impossible rather than merely absent."""
    siblings = sorted(m for m in _import_surface(TARGET) if m.startswith("sitting_"))
    assert siblings == [], (
        f"sitting_vectors imports {siblings} — that closes an import cycle and folds the whole "
        f"sitting subsystem into one indivisible review slice. Move the shared piece DOWN into "
        f"schema.py or embed.py instead, or move the caller UP."
    )


def test_the_import_surface_is_pinned_so_a_new_dependency_is_a_decision():
    """POSITIVE CONTROL for the check above. A test that only forbids `sitting_*` keeps passing if
    the file grows a dependency on some other layer that itself imports a sibling — the cycle
    arrives one hop away and this file still looks clean. Pinning the WHOLE surface means any new
    edge gets looked at, including an indirect one.
    """
    extra = sorted(_import_surface(TARGET) - ALLOWED)
    assert extra == [], (
        f"sitting_vectors gained {extra}. If that module is below it in the graph, add it to "
        f"ALLOWED and say why in the commit body; if it is beside or above it, the dependency is "
        f"pointing the wrong way."
    )


@pytest.mark.parametrize("spelling", [
    "from . import sitting_store",
    "from .sitting_store import get_sitting",
    "import pipeline.kb.sitting_store",
    "from pipeline.kb import sitting_store",
    "from pipeline.kb import sitting_store as sst",       # aliased — the grep rule still sees this
    "from .sitting_store import (\n    get_sitting,\n)",   # split across lines — grep does not
])
def test_the_check_catches_every_spelling_of_the_forbidden_import(spelling, tmp_path):
    """THE CHECK'S OWN NEGATIVE CONTROL. A guard that cannot fail is not a guard, and the last case
    is exactly the one the `.guards.py` substring rule is blind to — which is why that rule's
    message points here. This caught two real holes in `_import_surface` when it was written.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(spelling + "\n", encoding="utf-8")
    assert any(m.startswith("sitting_") for m in _import_surface(probe)), \
        f"the AST walk missed {spelling!r} — the enforcement has a hole"
