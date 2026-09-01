"""
opyt_core/redeem.py — `opyt-redeem`: a grant code in, a queryable peer out.

The OPERATOR rail, and no longer the reader's only path. It exchanges the one-time code for a
reader token at `POST /v1/redeem` and writes the peer row the `kb=` entry points read.
`mcp_server/share_tools.accept` does the same thing from an assistant and is what a person
actually reaches — R2's frictionless constraint rules out a shell on the reader's side, and
`accept` is single-phase for the same reason a preview cannot exist here: a grant code buys one
reader token and checking it would spend it. This command stays because a terminal is sometimes
the only thing available.

THE NAME IS THE READER'S TO PICK, and after R4 nothing breaks if two readers pick differently.
It used to be a contract: the envelope crossed the service verbatim carrying the OWNER's name,
so search notices told the host to pass `kb='<owner>'` back to `open()` and only a peer under
exactly that string made the hint resolve. Every request now sends `as_kb`, so the service
labels the answer with whatever THIS install calls the peer, and the name in the URL is an
opaque routing key nobody types. `suggested_name` from the redeem response is a starting point —
what the owner called themselves — and `--name` overrides it.
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
                                   "(default: the name the owner registered under)")
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
    asked = args.name or body.get("suggested_name") or owner
    name = peers.add(asked, f"{url}/v1/kb/{owner}", args.label, token=token)

    print(f"Registered '{name}' -> {url}/v1/kb/{owner}")
    if name != asked:
        # `add` never overwrites a row it did not recognise, because the token in it is the only
        # copy in existence. So the reader gets a working name and is told which, rather than a
        # prompt or a silently destroyed credential.
        print(f"('{asked}' was already another knowledge base on this install.)")
    print(f"Search it with kb='{name}' from any MCP client on this install.")
    if body.get("notice"):
        print(body["notice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
