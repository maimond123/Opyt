#!/usr/bin/env bash
# scripts/build_mcpb.sh — pack the Claude Desktop extension (.mcpb).
#
# WHY THIS IS A BUILD-TIME INSTALL
# Claude Desktop hard-codes a 60 000 ms MCP startup timeout in its bundled SDK and ignores the
# `timeout` config field, so whatever the extension does before it can answer `initialize` has to
# fit in a minute. The first version of this bundle shipped `uv` and let it fetch CPython plus 96
# packages on the user's machine at first launch; measured 2026-08-30 that took 179 s cold and
# Desktop killed it at 60 s. Moving the install here — onto a machine with no clock running —
# makes the budget irrelevant instead of something to squeeze under. First launch is then the
# same work as a warm relaunch, which measured 2.5 s.
#
# WHAT GOES IN, per architecture:
#   runtime/<arch>/python/          a relocatable standalone CPython (python-build-standalone)
#   runtime/<arch>/site-packages/   opyt + every dependency, already installed
# `uv` itself is NOT in the bundle any more. Its only job was to fetch and install; with the
# install already done there is nothing left for it to do.
#
# Python 3.12 is pinned deliberately. Left to resolve `requires-python >=3.10`, uv picks the
# newest — it chose 3.14.7 on 2026-08-30, a version this repo has never run its tests on. Pinning
# ships the interpreter the suite actually passes on.
#
#   bash scripts/build_mcpb.sh                     -> dist/opyt-<version>.mcpb
#   bash scripts/build_mcpb.sh --sandbox <dir>     -> dist/opyt-<version>-sandbox.mcpb
#
# --sandbox makes the bundle behave like a first install on a machine that is not one. It stamps
# OPYT_HOME into the manifest env, so the extension reads and writes a throw-away store instead
# of the author's real ~/.opyt with its corpus and its keys. (The uv cache and interpreter
# overrides the uv-based bundle needed are gone with uv.) Never ship a sandbox build.
set -euo pipefail

PY_VERSION="3.12.14"
PY_RELEASE="20260825"          # python-build-standalone release tag
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHES=("aarch64-apple-darwin" "x86_64-apple-darwin")

SANDBOX=""
if [ "${1:-}" = "--sandbox" ]; then
  SANDBOX="${2:?--sandbox needs a directory}"
  mkdir -p "$SANDBOX"; SANDBOX="$(cd "$SANDBOX" && pwd)"
fi

command -v uv >/dev/null || { echo "uv is required to build (not to run) the bundle" >&2; exit 1; }

# One home for the version: pyproject. The manifest carries a placeholder, never a second literal.
VERSION="$(python3 -c "
import re,pathlib
s=pathlib.Path('$REPO_ROOT/pyproject.toml').read_text()
print(re.search(r'^version = \"([^\"]+)\"', s, re.M).group(1))
")"
echo "── opyt $VERSION   ·   CPython $PY_VERSION+$PY_RELEASE"

# The wheel is built from THIS tree, not fetched from PyPI: the bundle is self-contained, so it
# never resolves anything at run time, and pinning it to a published version would mean the
# extension could only ever ship code that had already been uploaded.
# `uv build`, not `python3 -m build`: the ambient python3 is whatever is first on PATH (an
# Anaconda one here) and need not have `build` installed. uv is already this script's one
# requirement, and like `python -m build` it builds in an isolated temp dir — so a stale
# gitignored build/lib/ can never leak deleted modules into the artifact.
rm -rf "$REPO_ROOT/dist"/*.whl
uv build --quiet --wheel --out-dir "$REPO_ROOT/dist" "$REPO_ROOT"
WHEEL="$(ls "$REPO_ROOT/dist"/opyt-*.whl)"
echo "── wheel: $(basename "$WHEEL")"

STAGE="$REPO_ROOT/build/mcpb"
CACHE="$REPO_ROOT/build/runtime-cache"
rm -rf "$STAGE"; mkdir -p "$STAGE/bin" "$CACHE"

for arch in "${ARCHES[@]}"; do
  # ── the interpreter ───────────────────────────────────────────────────────────
  # Downloaded here and checksum-verified, never committed: 25 MB of someone else's compiled
  # binary per architecture does not belong in git history, and embedding it in an artifact
  # other people install is a supply-chain boundary.
  name="cpython-$PY_VERSION+$PY_RELEASE-$arch-install_only.tar.gz"
  tarball="$CACHE/$name"
  rel="https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE"
  # One SHA256SUMS for the whole release; this project publishes no per-asset .sha256.
  [ -f "$CACHE/SHA256SUMS" ] || curl -fsSL "$rel/SHA256SUMS" -o "$CACHE/SHA256SUMS"
  if [ ! -f "$tarball" ]; then
    echo "── downloading CPython $arch"
    curl -fsSL "$rel/$name" -o "$tarball"
  fi
  expected="$(awk -v n="$name" '$2 == n {print $1}' "$CACHE/SHA256SUMS")"
  [ -n "$expected" ] || { echo "no SHA256SUMS entry for $name" >&2; exit 1; }
  actual="$(shasum -a 256 "$tarball" | awk '{print $1}')"
  [ "$expected" = "$actual" ] || { echo "checksum mismatch for CPython $arch" >&2; exit 1; }

  mkdir -p "$STAGE/runtime/$arch"
  tar xzf "$tarball" -C "$STAGE/runtime/$arch"      # unpacks as ./python/

  # ── the packages ──────────────────────────────────────────────────────────────
  # --python-platform cross-installs: every dependency ships wheels for both macOS arches, so an
  # Intel machine can lay down a working arm64 tree. Verified 2026-08-30 by checking the Mach-O
  # arch of the resulting .so files.
  echo "── installing packages for $arch"
  uv pip install --quiet \
    --python-platform "$arch" --python-version "${PY_VERSION%.*}" \
    --target "$STAGE/runtime/$arch/site-packages" \
    "$WHEEL"
done

cp "$REPO_ROOT/mcpb/bin/launch.sh" "$STAGE/bin/launch.sh"; chmod +x "$STAGE/bin/launch.sh"
cp "$REPO_ROOT/mcpb/icon.png" "$STAGE/icon.png"

python3 - "$REPO_ROOT/mcpb/manifest.template.json" "$STAGE/manifest.json" "$VERSION" "$SANDBOX" <<'PY'
import json, sys
src, dst, version, sandbox = sys.argv[1:5]
m = json.loads(open(src).read().replace("__OPYT_VERSION__", version))
if sandbox:
    m["server"]["mcp_config"]["env"] = {"OPYT_HOME": sandbox + "/opyt-home"}
    m["display_name"] += " (sandbox)"
    m["name"] += "-sandbox"
open(dst, "w").write(json.dumps(m, indent=2) + "\n")
PY

# The official CLI, because it validates the manifest against the spec. A hand-rolled zip
# produces a file that only fails at double-click time, inside Claude Desktop, with no log.
mkdir -p "$REPO_ROOT/dist"
OUT="$REPO_ROOT/dist/opyt-$VERSION${SANDBOX:+-sandbox}.mcpb"
rm -f "$OUT"
npx --yes @anthropic-ai/mcpb pack "$STAGE" "$OUT"

echo
echo "── built $OUT"
ls -lh "$OUT"
[ -n "$SANDBOX" ] && echo "── SANDBOX build: OPYT_HOME=$SANDBOX/opyt-home — do not distribute this file."
exit 0
