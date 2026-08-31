"""
opyt_core/push.py — `opyt-push`: build the export and replace what the service serves.

The owner's whole publish loop is this one command, and it is the mirror of `opyt-redeem`: the
reader runs one command with a code, the owner runs one command with a token. Everything it does
is already built — `build_export` projects the store, `POST /v1/upload/{owner}` swaps the served
file atomically — so this module is the sequencing and the two settings, and nothing else.

FULL REPLACE, NEVER A DIFF. An export is a projection of a store, not a log of changes to one, so
"the newest upload wins" is the entire update model (`service/uploads.py`). There is no state here
to keep in step with the service, which is why there is no manifest of what was last pushed: the
served file's sha256, printed on every run, is the only fact worth comparing and the service
computes it from the bytes it actually received.

The owner NAME is derived, never configured: `GET /v1/tokens` answers it from the token itself. A
second config key would be a second place for one fact, and it would be the one place a typo
publishes to a name nobody reads. Asking first also validates the token and the URL before the
expensive build, so a wrong setting costs a round trip instead of a 115 MB projection.
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


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        prog="opyt-push",
        description="Build this store's export and replace what the service serves for you."
    ).parse_args(argv)

    token = get_credential("opyt_service")
    if not token:
        print("OPYT_SERVICE_TOKEN is not set, so there is nothing to publish with. "
              "Set it with `opyt-keys --set OPYT_SERVICE_TOKEN=<token>`; the token comes from "
              "whoever runs the service.", file=sys.stderr)
        return 1
    url = config.service_url()
    if not url:
        print(f"no `service_url` in {config.config_path()} — that is the service to publish to, "
              f"e.g. `service_url: https://api.useopyt.com`.", file=sys.stderr)
        return 1
    url = url.rstrip("/")
    auth = {"Authorization": f"Bearer {token}"}

    try:
        r = requests.get(f"{url}/v1/tokens", headers=auth, timeout=30)
    except RequestException as e:
        print(f"could not reach {url}: {e}", file=sys.stderr)
        return 1
    if r.status_code != 200:
        print(f"{url} refused the token: {error_detail(r)}", file=sys.stderr)
        return 1
    owner = r.json()["owner"]

    out = opyt_path("tmp", "export-push.db")
    try:
        manifest = build_export(out)
        local_sha = _sha256(out)
        with open(out, "rb") as fh:
            # A file object streams — the export is never held in memory, on either end.
            r = requests.post(f"{url}/v1/upload/{owner}", data=fh, headers=auth,
                              timeout=_UPLOAD_TIMEOUT)
        if r.status_code != 200:
            print(f"upload failed ({r.status_code}): {error_detail(r)}", file=sys.stderr)
            return 1
        served = r.json()
        if served["sha256"] != local_sha:
            # The service hashed what ARRIVED, so a mismatch means the bytes changed in flight —
            # and the swap has ALREADY happened, because `commit` runs before the response is
            # written. So the honest sentence is that a damaged export is what readers now get,
            # and the fix is another push. Nonzero, loudly: a publish that half-worked must never
            # read as one that worked.
            print(f"the service received different bytes than were sent — sent {local_sha}, "
                  f"received {served['sha256']}. That damaged copy is what it serves now; "
                  f"run this again.", file=sys.stderr)
            return 1
    except ValueError as e:
        # `build_export` on a store nobody has ingested into. It already says the useful sentence.
        print(str(e), file=sys.stderr)
        return 1
    except RequestException as e:
        print(f"the upload to {url} failed: {e}", file=sys.stderr)
        return 1
    finally:
        out.unlink(missing_ok=True)

    print(f"Published {manifest['tables']['atoms']} atoms as '{owner}' — "
          f"{served['bytes']:,} bytes, sha256 {local_sha[:12]}.")
    print(f"Readers query it with kb='{owner}' once they redeem a grant code for {url}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
