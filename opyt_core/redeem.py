"""
opyt_core/redeem.py — `opyt-redeem`: a grant code in, a queryable peer out.

The reader's entire setup is this one command. It exchanges the one-time code for a reader
token at `POST /v1/redeem` and writes the peer row the `kb=` entry points read — which is why
Phase 2 deliberately shipped no `peers` MCP tool: registering somebody's knowledge base is a
thing a person does once with a code, not a tool a host model can call.

The peer name DEFAULTS TO THE OWNER'S, and that default is a contract, not a convenience:
search notices tell the host to pass `kb='<owner>'` back to `open()` (the envelope crosses the
service verbatim, so the name in it is the owner's), and the default is what makes that hint
resolve on this install. `--name` still overrides it for the reader who already has a peer by
that name.
"""
from __future__ import annotations

import argparse
import sys
import uuid

import requests
from requests import RequestException

from opyt_core.kb_remote import error_detail
from opyt_core.paths import opyt_path
from pipeline.kb import peers


def get_install_id() -> str:
    """A random id, minted once per install and sent with the redeem, so the service can count
    distinct installations (TELEMETRY.md: `tokens.install_id` — no account behind it, never
    linked to a person). Lives here because this is its one caller; move it when a second
    arrives."""
    p = opyt_path("install_id")
    if p.exists():
        return p.read_text().strip()
    iid = uuid.uuid4().hex
    p.parent.mkdir(parents=True, exist_ok=True)   # redeem may be the first opyt command ever run
    p.write_text(iid)
    return iid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opyt-redeem",
        description="Exchange a grant code for read access to somebody's knowledge base.")
    ap.add_argument("url", help="the service, e.g. https://api.useopyt.com")
    ap.add_argument("code", help="the one-time grant code the owner sent you")
    ap.add_argument("--name", help="what kb= will call this knowledge base "
                                   "(default: the owner's name — see the module docstring)")
    ap.add_argument("--label", help="a display label for search notices (default: none)")
    args = ap.parse_args(argv)

    url = args.url.rstrip("/")
    try:
        r = requests.post(f"{url}/v1/redeem",
                          json={"code": args.code, "install_id": get_install_id()},
                          timeout=30)
    except RequestException as e:
        print(f"could not reach {url}: {e}", file=sys.stderr)
        return 1
    if r.status_code != 200:
        print(error_detail(r), file=sys.stderr)
        return 1

    body = r.json()
    owner, token = body["owner"], body["token"]
    name = args.name or owner
    peers.add(name, f"{url}/v1/kb/{owner}", args.label, token=token)

    print(f"Registered '{name}' -> {url}/v1/kb/{owner}")
    print(f"Search it with kb='{name}' from any MCP client on this install.")
    if body.get("notice"):
        print(body["notice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
