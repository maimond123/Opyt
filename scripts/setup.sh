#!/usr/bin/env bash
# scripts/setup.sh
# Bootstrap a contributor checkout of OPYT. Run once after cloning the repo.
# Usage: bash scripts/setup.sh
#
# Not the user install path (that is `uvx --from opyt@latest opyt-mcp`). Assumes, without
# checking, a `python3` >= 3.10; stock macOS ships 3.9. `editable_mode=compat` may force source
# builds (e.g. `cryptography` on an Intel Mac needs Rust and OpenSSL).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Setting up OPYT pipeline at: $REPO_ROOT"

# ── Python environment ────────────────────────────────────────────────────────
echo ""
echo "── Creating Python virtual environment..."
python3 -m venv "$REPO_ROOT/venv"
source "$REPO_ROOT/venv/bin/activate"

echo "── Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e "$REPO_ROOT[test]" --config-settings editable_mode=compat
echo "   ✓ Dependencies installed"

# ── Directory structure ───────────────────────────────────────────────────────
echo ""
echo "── Creating directories..."
mkdir -p "$REPO_ROOT"/state
echo "   ✓ Directories created"

# ── .env check ───────────────────────────────────────────────────────────────
echo ""
if [ ! -f "$REPO_ROOT/.env" ]; then
  echo "── Creating .env from template..."
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  chmod 600 "$REPO_ROOT/.env"
  echo "   ✓ .env created — fill in your credentials before running the pipeline"
else
  echo "── .env already exists, skipping"
fi

# ── Git hooks ──────────────────────────────────────────────────────────────────
echo ""
echo "── Installing git hooks..."
HOOKS_DIR="$(cd "$REPO_ROOT" && git rev-parse --git-common-dir)/hooks"
if [ -d "$REPO_ROOT/scripts/hooks" ]; then
  for hook in "$REPO_ROOT"/scripts/hooks/*; do
    name="$(basename "$hook")"
    cp "$hook" "$HOOKS_DIR/$name"
    chmod +x "$HOOKS_DIR/$name"
    echo "   ✓ $name installed (migration guard + any others)"
  done
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Setup complete."
echo ""
echo "Next steps:"
echo "  1. Fill in credentials in .env (OPENROUTER_API_KEY + TWITTERAPI_KEY required)"
echo "  2. Review ~/.opyt/settings.yaml (created from config/settings.example.yaml on first run)"
echo "  3. Log into x.com in Chrome — bookmarks sync from your own session (no key/OAuth)"
echo "  4. Test manually: python -m pipeline.worker.child pipeline.background:run"
