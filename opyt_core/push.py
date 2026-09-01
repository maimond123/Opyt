"""
opyt_core/push.py — `opyt-push`: build the export and replace what the service serves.

The owner's whole publish loop is `publish()`, and this module is the sequencing and the two
settings, nothing else. Everything it does is already built — `build_export` projects the store,
`POST /v1/upload/{owner}` swaps the served file atomically.

TWO CALLERS, ONE IMPLEMENTATION. `main()` is the CLI: it owns argv, the printing, and the exit
code. `publish()` is the function, and `pipeline/kb/push_catchup.py`'s rail imports it. That
split is the whole point of the refactor — a rail with its own upload sequence would be a second
implementation of the thing the fidelity of every reader's copy depends on.

FULL REPLACE, NEVER A DIFF. An export is a projection of a store, not a log of changes to one, so
"the newest upload wins" is the entire update model (`service/uploads.py`).

`push_watermark` is the one piece of state this rail keeps, and it belongs to the RAIL rather
than to this module — `publish()` is stateless, and `push_catchup` writes the watermark only
after `publish()` reports success. Nothing here mirrors the service: the served file's sha256,
returned on every run, is the only fact worth comparing and the service computes it from the
bytes it actually received.

The owner ROUTING KEY is derived, never configured: `GET /v1/tokens` answers it from the token
itself. A second config key would be a second place for one fact, and it would be the one place a
typo publishes to a key nobody reads. Asking first also validates the token and the URL before
the expensive build, so a wrong setting costs a round trip instead of a 115 MB projection.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from requests import RequestException

from opyt_core import config
from opyt_core.kb_remote import error_detail
from opyt_core.paths import opyt_path
from pipeline.credentials import get_credential
from pipeline.kb.export import build_export

_UPLOAD_TIMEOUT = 600   # seconds — a 115 MB body over a home connection, not a query


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_state(token: str, url: str) -> dict:
    """`GET /v1/tokens` — the routing key this token publishes under, plus R3's push gate.

    `{status: "ok", owner, last_upload_at, reads_since_last_upload, tokens}`, or a status naming
    why not. Separate from `publish` because the RAIL needs this answer to decide whether to
    publish at all, and paying for it twice would be a second round trip on every session open."""
    try:
        r = requests.get(f"{url}/v1/tokens", headers={"Authorization": f"Bearer {token}"},
                         timeout=30)
    except RequestException as e:
        return {"status": "unreachable", "message": f"could not reach {url}: {e}"}
    if r.status_code != 200:
        return {"status": "refused",
                "message": f"{url} refused the token: {error_detail(r)}"}
    return {"status": "ok", **r.json()}


def publish(token: str, url: str, *, owner: str | None = None) -> dict:
    """Build this store's export and make it what the service serves. Never raises.

    `{status: "ok", owner, atoms, bytes, sha256}`, or a status and a `message` saying why not.
    `owner` skips the `GET /v1/tokens` round trip for a caller that already made it — the rail
    did, to read the gate.

    The sha comparison is not ceremony. The service hashes what ARRIVED, and its `commit` runs
    before the response is written, so a mismatch means the bytes changed in flight AND the swap
    has already happened: the damaged copy is what readers get, and the honest report is that
    rather than a failure that changed nothing."""
    auth = {"Authorization": f"Bearer {token}"}
    if owner is None:
        state = fetch_state(token, url)
        if state["status"] != "ok":
            return state
        owner = state["owner"]

    out = opyt_path("tmp", "export-push.db")
    try:
        manifest = build_export(out)
        local_sha = _sha256(out)
        with open(out, "rb") as fh:
            # A file object streams — the export is never held in memory, on either end.
            r = requests.post(f"{url}/v1/upload/{owner}", data=fh, headers=auth,
                              timeout=_UPLOAD_TIMEOUT)
        if r.status_code != 200:
            return {"status": "upload_failed",
                    "message": f"upload failed ({r.status_code}): {error_detail(r)}"}
        served = r.json()
        if served["sha256"] != local_sha:
            return {"status": "corrupt",
                    "message": (f"the service received different bytes than were sent — sent "
                                f"{local_sha}, received {served['sha256']}. That damaged copy is "
                                f"what it serves now; run this again.")}
    except ValueError as e:
        # `build_export` on a store nobody has ingested into. It already says the useful sentence.
        return {"status": "empty_store", "message": str(e)}
    except RequestException as e:
        return {"status": "unreachable", "message": f"the upload to {url} failed: {e}"}
    finally:
        out.unlink(missing_ok=True)

    return {"status": "ok", "owner": owner, "atoms": manifest["tables"]["atoms"],
            "bytes": served["bytes"], "sha256": local_sha}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        prog="opyt-push",
        description="Build this store's export and replace what the service serves for you."
    ).parse_args(argv)

    token = get_credential("opyt_service")
    if not token:
        print("OPYT_SERVICE_TOKEN is not set, so there is nothing to publish with. "
              "Set it with `opyt-keys --set OPYT_SERVICE_TOKEN=<token>`, or ask your assistant "
              "to share your knowledge base, which registers one for you.", file=sys.stderr)
        return 1

    url = config.service_url().rstrip("/")
    res = publish(token, url)
    if res["status"] != "ok":
        print(res["message"], file=sys.stderr)
        return 1

    print(f"Published {res['atoms']} atoms as '{res['owner']}' — "
          f"{res['bytes']:,} bytes, sha256 {res['sha256'][:12]}.")
    print(f"Readers query it under whatever name they registered, once they redeem a grant "
          f"code for {url}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
