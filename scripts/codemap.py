#!/usr/bin/env python3
"""codemap — partition the deployed codebase into reviewable units.

Stdlib only, like guard.py. Computes the module import graph, finds the strongly
connected components (dependency CYCLES — mutually importing modules that cannot be
reviewed apart), and topologically orders the rest.

WHY a script and not a list: the cycles change as the code changes. A hardcoded slice
list in a plan document is stale the first time anyone breaks a cycle.

  python3 scripts/codemap.py                 the full map
  python3 scripts/codemap.py --slice 1       what slice 1 is, and why
  python3 scripts/codemap.py --json          machine-readable

SCOPE is the deployed product only (David, 2026-09-04); scripts/, website/,
tests/ and docs/ are excluded and are not slices. (mcpb/ was on that list until the
Claude Desktop bundle was deleted on 2026-09-14.)
"""
from __future__ import annotations
import ast, collections, json, pathlib, subprocess, sys

IN_SCOPE = ("pipeline", "mcp_server", "opyt_core", "service", "gateway")
BRANCH_NODES = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.match_case,
                ast.IfExp, ast.Assert, ast.comprehension)


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout
    return [f for f in out.split() if f.startswith(IN_SCOPE)]


def _modname(path: str) -> str:
    m = path[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def build():
    """-> (modules {name: path}, edges {name: {name}}, stats {name: (loc, complexity)})"""
    mods = {_modname(f): f for f in _tracked()}
    edges: dict[str, set[str]] = collections.defaultdict(set)
    stats: dict[str, tuple[int, int]] = {}
    for name, path in mods.items():
        src = pathlib.Path(path).read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        cx = 1
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                cx += len(node.values) - 1
            elif isinstance(node, BRANCH_NODES):
                cx += 1
            base = None
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:                       # relative: resolve against this module
                    anchor = ".".join(name.split(".")[: -node.level])
                    base = f"{anchor}.{base}" if base else anchor
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in mods and alias.name != name:
                        edges[name].add(alias.name)
                continue
            if not base:
                continue
            for cand in [base] + [f"{base}.{a.name}" for a in node.names]:
                if cand in mods and cand != name:
                    edges[name].add(cand)
        stats[name] = (len(src.splitlines()), cx)
    return mods, edges, stats


def sccs(mods, edges) -> list[list[str]]:
    """Tarjan, iterative — recursion overflows on a graph this size."""
    index, low, on_stack, stack, out, counter = {}, {}, set(), [], [], [0]
    for root in mods:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, child_i = work[-1]
            if child_i == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on_stack.add(v)
            descended = False
            for i, w in enumerate(sorted(edges[v])[child_i:], child_i):
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
    return out


def slices(mods, edges, stats):
    """Cycles first (largest first), then single modules in dependency order.

    Cycles lead because a cycle is indivisible AND because untangling one removes the
    function-level imports that exist only to break it. Singles follow their own
    dependencies so a module is reviewed after what it rests on.
    """
    comps = sccs(mods, edges)
    cycles = sorted((c for c in comps if len(c) > 1), key=len, reverse=True)
    in_cycle = {m for c in cycles for m in c}
    singles = [c[0] for c in comps if len(c) == 1]          # Tarjan emits reverse-topological
    singles.sort(key=lambda m: (-stats[m][1], m))            # worst complexity first
    return cycles, singles, in_cycle


def main(argv: list[str]) -> int:
    mods, edges, stats = build()
    cycles, singles, in_cycle = slices(mods, edges, stats)
    units = [("cycle", c) for c in cycles] + [("single", [s]) for s in singles]

    if "--json" in argv:
        print(json.dumps([{"n": i + 1, "kind": k, "modules": [mods[m] for m in ms],
                           "loc": sum(stats[m][0] for m in ms),
                           "complexity": sum(stats[m][1] for m in ms)}
                          for i, (k, ms) in enumerate(units)], indent=2))
        return 0

    if "--slice" in argv:
        n = int(argv[argv.index("--slice") + 1])
        if not 1 <= n <= len(units):
            print(f"slice must be 1..{len(units)}", file=sys.stderr)
            return 2
        kind, ms = units[n - 1]
        print(f"SLICE {n} of {len(units)}  ({kind})")
        for m in ms:
            print(f"  {mods[m]:56} {stats[m][0]:5} loc   complexity {stats[m][1]}")
        if kind == "cycle":
            print("\n  edges closing the cycle:")
            for a in ms:
                for b in sorted(edges[a] & set(ms)):
                    print(f"    {a}  ->  {b}")
            print("\n  These modules import each other. Review them TOGETHER — a cycle has no"
                  "\n  first member. Their function-level imports exist to break this cycle at"
                  "\n  load time; untangling it is what removes them.")
        return 0

    total_loc = sum(v[0] for v in stats.values())
    print(f"in scope: {len(mods)} modules, {total_loc} lines  ({', '.join(IN_SCOPE)})")
    print(f"import edges: {sum(len(v) for v in edges.values())}")
    print(f"\n{len(cycles)} CYCLES  ({len(in_cycle)} modules) — each is one indivisible slice")
    for i, c in enumerate(cycles, 1):
        loc = sum(stats[m][0] for m in c)
        print(f"  slice {i:2}  [{len(c)}]  {loc:5} loc   " + " · ".join(m.split(".")[-1] for m in c))
    print(f"\n{len(singles)} SINGLE modules — reviewable alone, worst complexity first")
    for i, m in enumerate(singles[:10], len(cycles) + 1):
        print(f"  slice {i:2}       {stats[m][0]:5} loc   cx {stats[m][1]:3}   {mods[m]}")
    print(f"  ... {len(singles) - 10} more" if len(singles) > 10 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
