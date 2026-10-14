"""
pipeline/sharing/client.py — the service's protocol routes, called from an install.

Register, grant, redeem, revoke, unpublish, link, the token list, the uploads, making or
deleting a page of sources, and the line's two routes (`mcp_server/line.py`): every route an
owner or a redeeming reader calls; and the two X sign-in routes an install with no browser of its
own calls (`pipeline/onboarding/service_signin.py`). Each answers the body of a 2xx, or raises
`Failed`, whose message is the sentence a tool result carries and whose `kind` (`unreachable`,
`refused` or `revoked`) is what a job status needs. The HTTP itself is
`opyt_core/service_http.py`, the one exchange every client of the service shares; a reader's
reads of a served knowledge base are `pipeline/retrieval/served.py`'s.

`revoked` is a 401: the service holds nothing for the owner token any more (`unshare`, or a
deletion done on the service). Only the owner's routes send a token, so the token is dropped here,
where all of them pass, and nothing publishes, shares or links with a dead one.
"""
from __future__ import annotations

from pathlib import Path

from opyt_core import service_http
from pipeline.sharing.account import Account, drop_token, hold_token, pending_token

_UPLOAD_TIMEOUT = 600   # seconds — a 115 MB body over a home connection, not a query
# The service fetches a page's papers' PDFs before it answers, up to 45 s each, side by side.
_PAGE_TIMEOUT = 120
# Muse's network holds a first request to a new host for 20-85 s, and the X
# sign-in is the one route an agent's computer calls first.
_SIGNIN_TIMEOUT = 120
# The service holds `line/next` open for the wait it is asked for, so the read outlasts the wait
# by this much.
_LINE_MARGIN = 15


_REVOKED = ("Opyt's server no longer holds this library, so Opyt here has stopped publishing it. "
            "`share` starts over, for a person or for the user's own claude.ai or ChatGPT.")


class Failed(Exception):
    """A route that did not answer with its body. `kind` is `unreachable`, `refused` or
    `revoked`; `status` is the HTTP status of a refusal, None otherwise."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


def _send(method: str, url: str, **kw) -> dict:
    try:
        return service_http.request(method, url, **kw)
    except service_http.Unreachable as e:
        raise Failed("unreachable", f"could not reach {url}: {e}") from e
    except service_http.Refused as e:
        if e.status == 401:
            drop_token()
            raise Failed("revoked", _REVOKED, e.status) from e
        raise Failed("refused", f"the service answered {e.status}: {e.detail}", e.status) from e
    except service_http.Unreadable as e:
        raise Failed("refused", str(e)) from e


def register(url: str, label: str | None) -> dict:
    """`POST /v1/register`: a new knowledge base, `{owner, token}`. Needs no token."""
    return _send("post", f"{url}/v1/register", json={"label": label})


def held_token(account: Account) -> str:
    """An owner token for a call that publishes nothing (the Google connection, a page): this
    install's own, else the one held aside, else a new one registered with no label and held aside
    (`account.hold_token`), so no publishing pass sees it."""
    token = account.token or pending_token()
    if token is None:
        token = register(account.url, None)["token"]
        hold_token(token)
    return token


def grant(account: Account, label: str | None, agent: bool = False) -> dict:
    """`POST /v1/grant`: a one-time invite code, `{owner, code}`. `agent` asks for an agent
    invite, whose code buys an agent token."""
    return _send("post", f"{account.url}/v1/grant", token=account.token,
                 json={"label": label, "agent": agent})


def redeem(url: str, code: str, install_id: str) -> dict:
    """`POST /v1/redeem`: a code spent for a reader token,
    `{owner, token, suggested_name, notice}`."""
    return _send("post", f"{url}/v1/redeem", json={"code": code, "install_id": install_id})


def revoke(account: Account, token_sha256: str) -> dict:
    """`POST /v1/revoke`: `{revoked}`, false when the token was already gone."""
    return _send("post", f"{account.url}/v1/revoke", token=account.token,
                 json={"token_sha256": token_sha256})


def unpublish(account: Account) -> dict:
    """`POST /v1/unpublish`: `{owner, readers_revoked, export_deleted}`."""
    return _send("post", f"{account.url}/v1/unpublish", token=account.token, json={})


def link(account: Account) -> dict:
    """`POST /v1/link`: one step of tying this knowledge base to the user's Google account —
    `{status: "waiting", user_code, verification_url}`, `{status: "linked"}`, or
    `{status: "failed", message}`."""
    return _send("post", f"{account.url}/v1/link", token=account.token, json={})


def unlink(account: Account) -> dict:
    """`POST /v1/unlink`: the Google account cut off from this knowledge base, `{unlinked}`,
    false when none was connected."""
    return _send("post", f"{account.url}/v1/unlink", token=account.token, json={})


def make_page(account: Account, title: str | None, note: str | None, answer: str | None,
              items: list[dict]) -> dict:
    """`POST /v1/pages`: a page of the answer and these items, `{id, path, days}`."""
    return _send("post", f"{account.url}/v1/pages", token=account.token,
                 json={"title": title, "note": note, "answer": answer, "items": items},
                 timeout=_PAGE_TIMEOUT)


def delete_page(account: Account, page_id: str) -> dict:
    """`POST /v1/pages/{id}/delete`: `{deleted}`, false when this token made no such page."""
    return _send("post", f"{account.url}/v1/pages/{page_id}/delete", token=account.token,
                 json={})


def line_next(account: Account, wait: int) -> dict:
    """`GET /v1/line/next`: the next tool call the web or an agent asked this library for, held
    open for up to `wait` seconds (the service takes at most 50) — `{call: {id, tool, args}}`,
    `{call: null}` when none arrived, or `{call: null, retry_after}` when the library has no web
    or agent writer and should ask again after that many seconds."""
    return _send("get", f"{account.url}/v1/line/next", token=account.token,
                 params={"wait": wait}, timeout=wait + _LINE_MARGIN)


def line_answer(account: Account, call_id: str, ok: bool, answer: dict) -> dict:
    """`POST /v1/line/{id}/answer`: what a call came to, `{recorded}`; false when no running call
    of this library has that id."""
    return _send("post", f"{account.url}/v1/line/{call_id}/answer", token=account.token,
                 json={"ok": ok, "answer": answer})


def x_signin(url: str) -> dict:
    """`POST /v1/x-signin`: a new X sign-in, `{signin, path, expires_in}`. Needs no token: a
    sign-in publishes nothing."""
    return _send("post", f"{url}/v1/x-signin", json={}, timeout=_SIGNIN_TIMEOUT)


def x_signin_collect(url: str, signin: str) -> dict:
    """`POST /v1/x-signin/collect`: `{status: "waiting" | "failed" | "expired"}`, or
    `{status: "connected", session}` once."""
    return _send("post", f"{url}/v1/x-signin/collect", json={"signin": signin},
                 timeout=_SIGNIN_TIMEOUT)


def tokens(account: Account) -> dict:
    """`GET /v1/tokens`: the routing key this token publishes under, every token issued for it,
    when the copy was last uploaded and when a Google account was connected — `{owner, tokens,
    last_upload_at, reads_since_last_upload, google_linked_at}`. The routing key comes from the
    service, never from config; asking
    first also validates the token and URL before the expensive build."""
    return _send("get", f"{account.url}/v1/tokens", token=account.token)


def upload(account: Account, owner: str, path: Path) -> dict:
    """`POST /v1/upload/{owner}`: replace what the service serves with the file at `path`,
    `{owner, bytes, sha256, path}`. A file object streams — the export is never held in memory,
    on either end."""
    with open(path, "rb") as fh:
        return _send("post", f"{account.url}/v1/upload/{owner}", token=account.token, data=fh,
                     timeout=_UPLOAD_TIMEOUT)


def upload_delta(account: Account, owner: str, path: Path) -> dict:
    """`POST /v1/upload/{owner}/delta`: patch what the service serves with the delta at `path`
    (`delta.py`), `{owner, bytes, digest}`. A 409 means the served copy cannot take it, and a
    service older than the route answers 404; either way the whole export fixes it."""
    with open(path, "rb") as fh:
        return _send("post", f"{account.url}/v1/upload/{owner}/delta", token=account.token,
                     data=fh, timeout=_UPLOAD_TIMEOUT)


def kb_location(url: str, owner: str) -> str:
    """Where a reader's peer row points for a knowledge base served at `url` under `owner`. The
    reads under it (`/meta`, `/search`, …) are `served`'s."""
    return f"{url}/v1/kb/{owner}"
