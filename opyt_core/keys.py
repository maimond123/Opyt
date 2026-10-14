"""
opyt_core/keys.py — manage API keys for a distributed install.

Which keys exist is one table: `opyt_core/credentials_registry.py`. Keys live in ~/.opyt/.env,
outside the repo, which opyt_core loads on import. Values are never printed back.

    opyt-keys --list                       # which keys are set, and which ones matter
    opyt-keys --set KEY=VALUE              # run --list first for the exact names
    python -m opyt_core.keys --list        # equivalent without the console script
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

from opyt_core.credentials_registry import REGISTRY
from opyt_core.paths import opyt_home, owner_only


def env_path() -> Path:
    """Where the keys file is, resolved at call time so `$OPYT_HOME` moves it."""
    return opyt_home() / ".env"


def _read() -> dict[str, str]:
    """`.env` as every other reader sees it (python-dotenv). A bare `KEY` line has no value and
    is not kept."""
    return {k: v for k, v in dotenv_values(env_path()).items() if v is not None}


def set_key(name: str, value: str) -> Path:
    """Upsert KEY=VALUE in ~/.opyt/.env (chmod 600 so secrets aren't world-readable)."""
    p = env_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = _read()
    cur[name] = value
    p.write_text("".join(f"{k}={v}\n" for k, v in cur.items()), encoding="utf-8")
    owner_only(p)
    return p


def status() -> dict[str, str]:
    """Per known key: 'set' or 'MISSING', plus its tier. No values are ever returned."""
    cur = _read()
    return {c.env: ("set" if (os.environ.get(c.env) or cur.get(c.env)) else "MISSING")
                   + f" ({c.tier})"
            for c in REGISTRY}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opyt-keys",
        description="Set/inspect OPYT API keys in ~/.opyt/.env (values never printed). "
                    "Run --list to see every key OPYT uses and whether it is required.")
    ap.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                    help=f"set a key (repeatable), e.g. {REGISTRY[0].env}=...")
    ap.add_argument("--list", action="store_true", help="show which keys are set (masked)")
    args = ap.parse_args(argv)

    for kv in args.set:
        if "=" not in kv:
            ap.error(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        if not k.strip():
            ap.error(f"--set expects a key name before '=', got {kv!r}")
        set_key(k.strip(), v.strip())
        print(f"[set] {k.strip()} -> {env_path()}")

    if args.list or not args.set:
        for k, st in status().items():
            print(f"  {k}: {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
