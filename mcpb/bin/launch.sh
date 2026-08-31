#!/bin/sh
# mcpb/bin/launch.sh — the entry point Claude Desktop spawns for the OPYT extension.
#
# It execs an interpreter that is already inside this bundle, against packages that are already
# installed inside this bundle. No network, no resolver, no install step.
#
# WHY, precisely: Claude Desktop hard-codes a 60 000 ms MCP startup timeout in its bundled SDK
# and ignores the `timeout` config field. An earlier bundle shipped `uv` and let it fetch
# CPython plus 96 packages at first launch; measured 2026-08-30 on a cold machine that took 179 s
# and Desktop killed it at 60 s. The install still finished and the server still came up — three
# minutes after nobody was listening. So the install moved to BUILD time (scripts/build_mcpb.sh)
# and this script only starts things.
#
# Arch is resolved here because MCPB's `platform_overrides` keys on darwin/win32/linux with no
# key for CPU architecture, and Apple Silicon and Intel need different runtimes.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)

case "$(uname -m)" in
  arm64)  ARCH="aarch64-apple-darwin" ;;
  x86_64) ARCH="x86_64-apple-darwin" ;;
  *)      echo "opyt: unsupported CPU architecture $(uname -m)" >&2; exit 1 ;;
esac

RUNTIME="$ROOT/runtime/$ARCH"
# python3.12 and not the python3 symlink beside it: whether a zip preserves symlinks depends on
# the packer and the extractor, and this bundle is unpacked by code we do not own.
PYTHON="$RUNTIME/python/bin/python3.12"

if [ ! -f "$PYTHON" ]; then
  echo "opyt: no bundled runtime for $ARCH at $PYTHON" >&2
  exit 1
fi
# Same reasoning as the symlink: restore the execute bit if extraction dropped it. One
# idempotent syscall, and without it the failure reads as an unexplained "server disconnected".
[ -x "$PYTHON" ] || chmod +x "$PYTHON"

# -s keeps the user's ~/.local site-packages out of the path: this runtime is meant to be exactly
# what was built and audited, not that plus whatever the machine happens to have.
PYTHONPATH="$RUNTIME/site-packages"
export PYTHONPATH
exec "$PYTHON" -s -m mcp_server.server
