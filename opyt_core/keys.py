"""
opyt_core/keys.py — manage API keys for a distributed install.

Which keys exist, and what each one buys, is not written here — it is one table:
`opyt_core/credentials_registry.py`. Read the registry, not this docstring.

Search needs a key unless you ask for BM25. On the atom rail the DEFAULT search mode is
hybrid, and its semantic arm goes through a HOSTED embedder — so the LLM provider key is required
to build or query the index. Only `mode="bm25"` is keyless.

Keys live in ~/.opyt/.env — OUTSIDE the repo — which opyt_core loads on import, so a user with no
repo checkout still resolves them. Values are NEVER printed back; --list reports only set vs
MISSING, plus whether the key is required or optional.

    opyt-keys --list                       # which keys are set, and which ones matter
    opyt-keys --set KEY=VALUE              # run --list first for the exact names
    python -m opyt_core.keys --list        # equivalent without the console script

Key names are not listed here; `--list` prints them from the registry so it never shows a
retired key.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .credentials_registry import REGISTRY
from .credentials_registry import KNOWN as KNOWN          # re-export: the ONE list, derived
from .paths import opyt_home


def _env_path() -> Path:
    return opyt_home() / ".env"  # the one sandbox knob ($OPYT_HOME), via paths.py


def _read() -> dict[str, str]:
    p = _env_path()
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def set_key(name: str, value: str) -> Path:
    """Upsert KEY=VALUE in ~/.opyt/.env (chmod 600 so secrets aren't world-readable)."""
    p = _env_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = _read()
    cur[name] = value
    p.write_text("".join(f"{k}={v}\n" for k, v in cur.items()), encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass
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
                    # Example is derived from the registry so it can't go stale.
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
        print(f"[set] {k.strip()} -> {_env_path()}")

    if args.list or not args.set:
        for k, st in status().items():
            print(f"  {k}: {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
