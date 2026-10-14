"""
service/deletion.py — delete everything this service holds for one library.

One flow, two ways in: `POST /v1/unpublish`, with the library's owner token (the builder's
`unshare`), and this module's command, which the operator runs for a deletion requested by email
from a user who has lost their Mac:

    cd /srv/opyt-src && sudo -u opyt OPYT_HOME=/srv/opyt .venv/bin/python \\
        -m service.deletion --google-sub <sub>        # or: --owner <routing key>

What goes, in three steps, so a deletion that stops part-way is finished by running it again:

  1. every way to read or change it: the reader-view pages made from it, with their PDFs and
     the comments left on them (`page_comments`, which `pages.delete_all` empties with them); reader
     and agent tokens, invites (spent ones too, which keep a reader's label), the Google accounts
     linked to it, with any sign-in in flight, the accounts it was shared with
     (`account_peers`), and the tool calls waiting for its Mac (`line.py`);
  2. the served copy: its registry row, the file, and a partial upload a restart left behind;
  3. its records, and the owner's token last: the publication row, the read counts, the routing
     key's claim. Until this commits the builder can still call again.

Steps 1 and 3 are one transaction each. Deleting the owner token is also what stops the builder
re-publishing: without it the publication row's absence would read as "never published", which
the builder's push gate answers by uploading. The `/stats` totals are sums over these rows, so
they drop by what is deleted. `service.db`'s nightly backups (DEPLOY.md §6) keep a deleted row for
up to seven days.
"""
from __future__ import annotations

import argparse
import json
import sys

from service import accounts, db, tokens, uploads
from service.pages import store as pages

# Every table with rows keyed by a library's routing key (`owner`), by the step that empties it.
# `tests/service/test_deletion.py` fails when a table with an `owner` column is in none of these.
# `pages` is emptied by `pages.delete_all`, which also removes each page's folder.
PAGES = ("pages",)
ACCESS = ("grant_codes", "agent_grants", "accounts", "account_links", "account_peers",
          "line_calls", "line_seen")
RECORDS = ("owner_uploads", "usage_daily", "owner_claims", "tokens")


def delete_library(owner: str) -> dict:
    """Delete everything held for `owner`:
    `{owner, readers_revoked, export_deleted, deleted: True}`.

    Raises `uploads.BadOwner` for a name that cannot be a library, before anything is written.
    Idempotent."""
    uploads.valid_owner(owner)
    pages.delete_all(owner)
    conn = db.connect()
    try:
        with conn:
            # Every token but the owner's: readers, and agents, which also act.
            readers = conn.execute("DELETE FROM tokens WHERE owner = ? AND role != 'owner'",
                                   (owner,)).rowcount
            for table in ACCESS:
                conn.execute(f"DELETE FROM {table} WHERE owner = ?", (owner,))
        served = uploads.remove(owner)
        with conn:
            for table in RECORDS:
                conn.execute(f"DELETE FROM {table} WHERE owner = ?", (owner,))
    finally:
        conn.close()
    return {"owner": owner, "readers_revoked": readers, "export_deleted": served, "deleted": True}


def main(argv: list[str] | None = None) -> int:
    """The operator's command: one library, named by its Google account or its routing key."""
    parser = argparse.ArgumentParser(prog="python -m service.deletion",
                                     description="Delete everything held for one library.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--google-sub", help="the Google account id the library is linked to")
    which.add_argument("--owner", help="the library's routing key")
    args = parser.parse_args(argv)
    owner = args.owner or accounts.owner_of(args.google_sub)
    if owner is None:
        print("No library is linked to that Google account here.", file=sys.stderr)
        return 1
    if not tokens.list_tokens(owner):
        print(f"Nothing is held here for the library {owner!r}.", file=sys.stderr)
        return 1
    print(json.dumps(delete_library(owner)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
