"""service/app.py — the HTTP surface for knowledge-base reads, publication, access, and stats,
the X sign-in pages for installs with no browser of their own (`x_signin.py`), pages of sources
(`pages/`), the line a Mac holds for the calls the web leaves it (`line.py`), and the connector
(`connector.py`) mounted beside them where Google sign-in is configured.

A thin adapter: each read handler authenticates and calls `reading`, which runs the same function
a local reader calls against a registered peer, capped and counted.

Keyword search needs the query text, so "reader queries are blind" is a retention commitment (no
query column, no request-body logging), not an architectural one. Read handlers are sync and GIL-bound, so throughput scales with uvicorn
`--workers`; `--log-config service/log_config.json` is required to keep IPs out of access logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.hosted_signin import display
from pipeline.sharing import delta

from . import (accounts, connector, deletion, google_device, grants, library_writes, line,
               owner_tools, publications, reading, serving, uploads, x_signin, x_signin_page)
from .access import owner_token, publisher
from .pages import comments, make, render, show, store
from .serving import acting, served
# Aliased: `stats`, `tokens` and `revoke` are also route handlers below.
from . import stats as public_stats
from . import tokens as service_tokens

# None where this service has no Google client configured: then there is no connector.
_connector = connector.asgi_app()

# The connector's lifespan runs its MCP session manager; FastAPI's own routes need none.
app = FastAPI(title="OPYT knowledge-base service",
              lifespan=_connector.lifespan if _connector is not None else None)


# ── request bodies ───────────────────────────────────────────────────────────────

class SearchBody(BaseModel):
    """`run_kb_search`'s arguments, minus `kb` (the path). An older client's `mode` and
    `query_vector` are ignored: pydantic drops fields a model does not declare."""
    query: str
    tags: list[str] | None = None
    what_kind: str | None = None
    source_type: str | None = None
    who_id: str | list[str] | None = None
    who: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    # Absent means the sectioned shape; forwarded, not defaulted, to match a local call.
    entry_mode: str | list[str] | None = None
    k: int = 8
    # Where this page of hits starts. Absent means the first page, answered without
    # `next_offset`, as it was before a search could page.
    offset: int | None = None
    # What the reader calls this knowledge base, echoed in every `kb` field; defaults to the
    # routing key.
    as_kb: str | None = None


class OpenBody(BaseModel):
    atom_id: str
    as_kb: str | None = None
    # Where this page of the item's text starts (`reading.page`).
    offset: int = 0


class AggregateBody(BaseModel):
    scope: dict | None = None
    as_kb: str | None = None


class GrantBody(BaseModel):
    label: str | None = None
    # True mints an agent invite: its code buys an agent token, which acts as the owner.
    agent: bool = False


class RedeemBody(BaseModel):
    code: str
    install_id: str | None = None


# Shown at redeem, before the first query.
_REDEEM_NOTICE = (
    "This service counts reads per day per reader so the owner can see usage and revoke access; "
    "it never records query text, IP addresses, or which atoms you read. "
    "Details: TELEMETRY.md in the Opyt repo."
)


class RevokeBody(BaseModel):
    token_sha256: str


class CollectBody(BaseModel):
    signin: str


class ClaimBody(BaseModel):
    session: str | None = None    # the tab's token from a claim before a reload


class RegisterBody(BaseModel):
    label: str | None = None


class PageBody(BaseModel):
    title: str | None = None
    note: str | None = None
    answer: str | None = None     # markdown, drawn above the sources
    # Each as `open()` returns an atom: `source_type`, `source_url`, `when_ts`, `description`,
    # `raw` and `payload` (`pages/make.py` keeps only those).
    items: list[dict]


class CommentBody(BaseModel):
    """A comment from a page's own script. `thread` makes it a reply; otherwise `item` says which
    source it is on, and `quote` (with the text around it) which passage."""
    name: str
    body: str
    thread: str | None = None
    item: int | None = None
    quote: str | None = None
    prefix: str | None = None
    suffix: str | None = None


class CommentKey(BaseModel):
    key: str


class AnswerBody(BaseModel):
    """The Mac's answer to a call it was handed: `ok` false when it could not run it."""
    ok: bool
    answer: dict


# The owner's agent's tools (`/v1/library/…`). Each body is its connector tool's arguments.
class ReaderViewBody(BaseModel):
    atom_ids: list[str] | None = None
    title: str | None = None
    note: str | None = None
    answer: str | None = None
    delete: str | None = None


class ShareBody(BaseModel):
    """An invite for a person. An agent cannot make an invite for another agent: only the owner
    can, from Opyt on their computer or the connector."""
    confirm: bool = False
    for_whom: str | None = None


class UnshareBody(BaseModel):
    reader: str
    confirm: bool = False


class HopperBody(BaseModel):
    reference: str
    kind_hint: str | None = None


class ForgetBody(BaseModel):
    atom_id: str | None = None
    atom_ids: list[str] | None = None
    oracle: str | None = None
    confirm: bool = False


class WatchBody(BaseModel):
    action: str
    name: str | None = None
    watches: list | None = None
    words: list[str] | None = None
    people: list[str] | None = None
    topic_id: str | None = None
    drop_watches: list[str] | None = None


class AddOracleBody(BaseModel):
    reference: str
    x_lookback: str | None = None
    web_lookback: str | None = None
    scholar_lookback: str | None = None
    scholar_topics: list[str] | None = None


# ── the query endpoints ──────────────────────────────────────────────────────────

@app.post("/v1/kb/{owner}/search")
def search(owner: str, body: SearchBody,
           auth: dict = Depends(served)) -> dict:
    return reading.search(
        owner, auth["token_sha256"], body.query, tags=body.tags, what_kind=body.what_kind,
        source_type=body.source_type, who_id=body.who_id, who=body.who,
        date_from=body.date_from, date_to=body.date_to, entry_mode=body.entry_mode, k=body.k,
        offset=body.offset, as_kb=body.as_kb,
    )


@app.post("/v1/kb/{owner}/open")
def open_atom(owner: str, body: OpenBody,
              auth: dict = Depends(served)) -> dict:
    return reading.open_atom(owner, auth["token_sha256"], body.atom_id, as_kb=body.as_kb,
                             offset=body.offset)


@app.post("/v1/kb/{owner}/aggregate")
def aggregate(owner: str, body: AggregateBody,
              auth: dict = Depends(served)) -> dict:
    return reading.aggregate(owner, auth["token_sha256"], body.scope, as_kb=body.as_kb)


@app.get("/v1/kb/{owner}/feed")
def feed(owner: str, since: str | None = None, since_id: str | None = None,
         limit: int = 50, entry_mode: str | None = None,
         date_from: str | None = None, date_to: str | None = None,
         as_kb: str | None = None,
         auth: dict = Depends(served)) -> dict:
    """The newest arrivals, cursor-paged. A GET, for standing polls from plain HTTP clients.
    `entry_mode` is comma-separated."""
    modes = ([m.strip() for m in entry_mode.split(",") if m.strip()] or None) \
        if entry_mode else None
    try:
        return reading.feed(owner, auth["token_sha256"], since=since, since_id=since_id,
                            limit=limit, entry_mode=modes, date_from=date_from,
                            date_to=date_to, as_kb=as_kb)
    except ValueError as e:
        # A malformed date is a 400, not a silently dropped filter.
        raise HTTPException(status_code=400, detail=str(e)) from e


# ── upload ───────────────────────────────────────────────────────────────────────

@app.post("/v1/upload/{owner}")
async def upload(owner: str, request: Request,
                 auth: dict = Depends(publisher)) -> dict:
    """Raw body = the export file. Replaces what is served, atomically.

    `async` so the body streams instead of being held in memory; the blocking disk writes are
    accepted."""
    try:
        return await serving.receive(owner, auth["label"], request.stream())
    except uploads.BadOwner as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except uploads.NoSpace as e:
        # 507: the storage is the problem, so the client should retry later.
        raise HTTPException(status_code=507, detail=str(e)) from e
    except uploads.TooLarge as e:
        # 413: the owner can shrink the export; the previous one is still served.
        raise HTTPException(status_code=413, detail=str(e)) from e


@app.post("/v1/upload/{owner}/delta")
async def upload_delta(owner: str, request: Request,
                       auth: dict = Depends(publisher)) -> dict:
    """Raw body = a delta (`pipeline/sharing/delta.py`): only the rows that changed. Patches what
    is served in place, in one transaction. 409 means this copy cannot take it and the owner
    should upload the whole export instead; nothing was changed."""
    try:
        return await serving.receive_delta(owner, request.stream())
    except uploads.BadOwner as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except uploads.NoSpace as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except (uploads.TooLarge, delta.TooLarge) as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except delta.Stale as e:
        raise HTTPException(status_code=409, detail=f"send the whole export: {e}") from e


# ── register / grant / redeem / revoke ───────────────────────────────────────────

@app.post("/v1/register")
def register(body: RegisterBody) -> dict:
    """A new knowledge base: `{owner, token}`, unauthenticated and unmetered, returned once.

    `label` is the owner's display name, returned to readers as `suggested_name`; it never routes.
    """
    owner, token = service_tokens.register_owner(body.label)
    return {"owner": owner, "token": token}


@app.post("/v1/grant")
def grant(body: GrantBody, auth: dict = Depends(owner_token)) -> dict:
    """Mint a one-time code, returned once; it buys exactly one reader token, or one agent
    token when `agent` is set."""
    return {"owner": auth["owner"],
            "code": grants.mint_grant(auth["owner"], label=body.label, agent=body.agent)}


@app.post("/v1/redeem")
def redeem(body: RedeemBody) -> dict:
    """Exchange a code for the token it buys. For a reader token the client
    (`pipeline/sharing/accept.py`) writes its peer row from the answer; an agent invite's answer
    also carries `role: "agent"`.

    `owner` is the opaque routing key; `suggested_name` (may be null) is the owner's label, which
    the client registers the peer under. No base URL is returned: the client knows it."""
    try:
        owner, token, role = grants.redeem_grant(body.code, body.install_id)
    except grants.GrantUnavailable as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    answer = {"owner": owner, "token": token,
              "suggested_name": service_tokens.owner_label(owner), "notice": _REDEEM_NOTICE}
    # A reader's answer is unchanged, so an install older than agent keys reads it as before.
    return {**answer, "role": role} if role == "agent" else answer


@app.post("/v1/revoke")
def revoke(body: RevokeBody, auth: dict = Depends(owner_token)) -> dict:
    """One row delete, effective on the next request."""
    return {"revoked": service_tokens.revoke(auth["owner"], body.token_sha256)}


@app.post("/v1/link")
def link(auth: dict = Depends(owner_token)) -> dict:
    """One step of tying this token's knowledge base to the Google account the connector signs
    in with: a code for the user to enter, `linked`, or `failed` (see `accounts.link`)."""
    try:
        return accounts.link(auth["owner"], time.time())
    except google_device.Unavailable as e:
        # 503: Google or this service's configuration, never the caller; nothing was written.
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/unlink")
def unlink(auth: dict = Depends(owner_token)) -> dict:
    """Cut the Google account off from this token's knowledge base: `{unlinked}`, false when none
    was linked. The connector can no longer read it; the copy and every reader stay."""
    return {"unlinked": accounts.unlink(auth["owner"])}


@app.post("/v1/unpublish")
def unpublish(auth: dict = Depends(owner_token)) -> dict:
    """Delete everything this service holds for this knowledge base (`deletion.py`): readers,
    invites, Google links, the served copy, its records, and last the calling token itself, which
    is refused from then on. The token names the knowledge base."""
    return deletion.delete_library(auth["owner"])


@app.get("/v1/tokens")
def tokens(auth: dict = Depends(owner_token)) -> dict:
    """Who currently holds a token for this knowledge base (hashes only), plus the push gate's
    demand terms, since the push job already calls this first, and when a Google account was
    linked (null when none is): the rest of who can reach it."""
    return {"owner": auth["owner"], "tokens": service_tokens.list_tokens(auth["owner"]),
            **publications.publish_demand(auth["owner"]),
            "google_linked_at": accounts.linked_at(auth["owner"])}


# ── the line: calls the web leaves for the owner's Mac ────────────────────────────────

@app.get("/v1/line/next")
async def line_next(wait: float = line.NEXT_WAIT_MAX_S,
                    auth: dict = Depends(owner_token)) -> dict:
    """The Mac asks for the next call waiting for it; the answer is held up to `wait` seconds (at
    most 50) for one to arrive: `{call: {id, tool, args}}`, `{call: null}` (ask again at once),
    or `{call: null, retry_after}` when nothing could ask this library for a call. `async`,
    because the wait sleeps between looks at the table."""
    return await line.next_call(auth["owner"], wait)


@app.post("/v1/line/{call_id}/answer")
def line_answer(call_id: str, body: AnswerBody, auth: dict = Depends(owner_token)) -> dict:
    """The Mac's answer to a call it was handed: `{recorded}`, false when this library has no
    running call by that id, and the Mac drops it."""
    return {"recorded": line.answer(auth["owner"], call_id, body.ok, body.answer)}


# ── the owner's agent: the web's tools over plain HTTPS ─────────────────────────────
#
# One route per tool the connector offers beyond reading, for an agent that calls web addresses
# and holds an agent key (`access.agent_token`): the same flows, the same arguments and the same
# answers as the connector's tools (`connector.py`). An agent reads through `/v1/kb/{owner}/…`.

def _told(auth: dict, answer: dict) -> dict:
    """An answer to the agent, with a line for each change that ended after its call stopped
    waiting: what the connector's `Earlier` adds for its callers."""
    return library_writes.with_earlier(auth["owner"], answer)


@app.get("/v1/library/status")
def library_status(auth: dict = Depends(acting)) -> dict:
    """What the library holds, when it was last published, and what was asked to change in it
    lately with how each ended."""
    return _told(auth, owner_tools.status(auth["owner"]))


@app.post("/v1/library/reader_view")
def library_reader_view(body: ReaderViewBody, auth: dict = Depends(acting)) -> dict:
    """A page of the items in `atom_ids`, for anyone with its link to read and comment on, or
    (`delete`) one taken down. Each item is counted as a read by this key."""
    return _told(auth, owner_tools.reader_view(auth["owner"], auth["token_sha256"], body.atom_ids,
                                               body.title, body.note, body.answer, body.delete))


@app.post("/v1/library/share")
def library_share(body: ShareBody, auth: dict = Depends(acting)) -> dict:
    """A preview of what an invite gives, or (`confirm`) a one-time invite for one person."""
    return _told(auth, owner_tools.share(auth["owner"], body.confirm, body.for_whom, agent=False))


@app.post("/v1/library/unshare")
def library_unshare(body: UnshareBody, auth: dict = Depends(acting)) -> dict:
    """A preview of cutting one reader or agent off, or (`confirm`) the cut."""
    return _told(auth, owner_tools.unshare(auth["owner"], body.reader, body.confirm))


@app.post("/v1/library/hopper")
async def library_hopper(body: HopperBody, auth: dict = Depends(acting)) -> dict:
    """Save one link into the library."""
    return _told(auth, await library_writes.hopper(auth["owner"], body.reference, body.kind_hint))


@app.post("/v1/library/forget")
async def library_forget(body: ForgetBody, auth: dict = Depends(acting)) -> dict:
    """A preview of removing items or of no longer following one Oracle, or (`confirm`) the
    removal. Nothing asks before a confirmed call but the caller."""
    return _told(auth, await library_writes.forget(
        auth["owner"], atom_id=body.atom_id, atom_ids=body.atom_ids, oracle=body.oracle,
        confirm=body.confirm))


@app.post("/v1/library/watch")
async def library_watch(body: WatchBody, auth: dict = Depends(acting)) -> dict:
    """One action on the topics the library follows: preview, create, edit, drop or list."""
    return _told(auth, await library_writes.watch(
        auth["owner"], body.action, name=body.name, watches=body.watches, words=body.words,
        people=body.people, topic_id=body.topic_id, drop_watches=body.drop_watches))


@app.post("/v1/library/add_oracle")
async def library_add_oracle(body: AddOracleBody, auth: dict = Depends(acting)) -> dict:
    """Add one person whose writing the library collects."""
    return _told(auth, await library_writes.add_oracle(
        auth["owner"], body.reference, x_lookback=body.x_lookback, web_lookback=body.web_lookback,
        scholar_lookback=body.scholar_lookback, scholar_topics=body.scholar_topics))


# ── the public stats page ─────────────────────────────────────────────────────────

@app.get("/v1/stats")
def stats() -> dict:
    """The roll-up as JSON, unauthenticated on purpose (the transparency page for `TELEMETRY.md`).
    `stored_bytes_by_kb` is keyed by routing key, with no label; nothing is per-reader."""
    return public_stats.rollup()


@app.get("/stats", response_class=HTMLResponse)
def stats_page() -> str:
    """The same numbers as `/v1/stats`, as a page."""
    return public_stats.page()


# ── pages of sources ──────────────────────────────────────────────────────────────

# A page's link is its only secret: no search engine, no cache, no referrer carries it on.
_PAGE_HEADERS = {"Content-Security-Policy": render.CSP, "X-Robots-Tag": "noindex, nofollow",
                 "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
                 "X-Content-Type-Options": "nosniff"}
_PDF_HEADERS = {"X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer",
                "Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"}


@app.post("/v1/pages")
def make_page(body: PageBody, auth: dict = Depends(owner_token)) -> dict:
    """A page of the sources an install sends, readable by anyone with its link until nobody
    has opened it for `days` days: `{id, path, days}`. Its papers' PDFs are fetched before it
    answers."""
    try:
        return make.create(auth["owner"], body.items, body.title, body.note, body.answer,
                           time.time())
    except make.Refused as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/pages/{page_id}/delete")
def delete_page(page_id: str, auth: dict = Depends(owner_token)) -> dict:
    """Take one of this token's pages down now: `{deleted}`, false when it has no such page."""
    return {"deleted": store.delete(auth["owner"], page_id)}


@app.get("/p/{page_id}", response_class=HTMLResponse, include_in_schema=False)
def page_view(page_id: str) -> HTMLResponse:
    """The page, drawn now from what it stores. An unknown or expired link gets the Opyt page
    that says so."""
    html = show.page(page_id, time.time())
    if html is None:
        return HTMLResponse(render.gone(store.DAYS), status_code=404, headers=_PAGE_HEADERS)
    return HTMLResponse(html, headers=_PAGE_HEADERS)


@app.get("/p/{page_id}/pdf/{n}", include_in_schema=False)
def page_pdf(page_id: str, n: int) -> FileResponse:
    """A paper's PDF, from the page's own address so the page can frame it."""
    path = show.pdf(page_id, n, time.time())
    if path is None:
        raise HTTPException(status_code=404, detail="no such PDF")
    return FileResponse(path, media_type="application/pdf", headers=_PDF_HEADERS)


# A page's comments, for the page's own script. Whoever holds the link reads them, adds one and
# deletes their own; nothing here takes a token.
_COMMENT_HEADERS = {"X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer",
                    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


@app.get("/p/{page_id}/comments", include_in_schema=False)
def page_comments(page_id: str) -> JSONResponse:
    """`{comments, limits}`: every comment on the page, oldest first."""
    got = comments.listed(page_id, time.time())
    if got is None:
        raise HTTPException(status_code=404, detail="This page has expired.")
    return JSONResponse(got, headers=_COMMENT_HEADERS)


@app.post("/p/{page_id}/comments", include_in_schema=False)
def post_comment(page_id: str, body: CommentBody) -> JSONResponse:
    """Keep a comment: `{comment, key}`. The key, returned this once, is what deletes it."""
    try:
        got = comments.post(page_id, body.model_dump(), time.time())
    except comments.Refused as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if got is None:
        raise HTTPException(status_code=404, detail="This page has expired.")
    return JSONResponse(got, headers=_COMMENT_HEADERS)


@app.post("/p/{page_id}/comments/{comment_id}/delete", include_in_schema=False)
def delete_comment(page_id: str, comment_id: str, body: CommentKey) -> JSONResponse:
    """Delete a comment for the holder of its key: `{deleted}`."""
    return JSONResponse({"deleted": comments.delete(page_id, comment_id, body.key)},
                        headers=_COMMENT_HEADERS)


# ── X sign-in for an install with no browser of its own ─────────────────────────────

# noVNC and the sign-in page's desktop script, vendored in `service/static/`.
_STATIC = Path(__file__).resolve().parent / "static"
# The site's Opyt mark, for the browser tab and for a chat app's card of a sign-in link (without it,
# Muse drew "AP" in its place). Served from the site's own files, so there is one copy.
_SITE = Path(__file__).resolve().parent.parent / "website"


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return FileResponse(_SITE / "favicon.ico")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return FileResponse(_SITE / "favicon.svg")


@app.get("/icon.png", include_in_schema=False)
def icon_png() -> FileResponse:
    return FileResponse(_SITE / "icon.png")


@app.post("/v1/x-signin")
def x_signin_mint() -> dict:
    """A new sign-in: a collect token for the install and a page path for its user. No account:
    a sign-in publishes nothing."""
    return x_signin.mint(time.monotonic())


@app.post("/v1/x-signin/collect")
def x_signin_collect(body: CollectBody) -> dict:
    """`waiting`, `connected` with the session (once), `failed` or `expired`. The token rides the
    body, never the URL, so no access log holds it."""
    return x_signin.collect(body.signin, time.monotonic())


@app.get("/x-signin/{nonce}", response_class=HTMLResponse)
def x_signin_open(nonce: str) -> HTMLResponse:
    """Serve the page, spending nothing: a link scanner's fetch must not use up the link. The
    page's script claims the sign-in."""
    if not x_signin.link_open(nonce, time.monotonic()):
        return x_signin_page.refusal("This sign-in link has expired.", 404)
    return HTMLResponse(x_signin_page.page_html(nonce), headers={"Cache-Control": "no-store"})


@app.post("/x-signin/{nonce}/claim")
def x_signin_claim(nonce: str, body: ClaimBody) -> dict:
    return x_signin.claim(nonce, body.session, time.monotonic())


@app.post("/x-signin/s/{session}/start")
def x_signin_start(session: str) -> dict:
    return x_signin.start(session, time.monotonic())


@app.post("/x-signin/s/{session}/restart")
def x_signin_restart(session: str) -> dict:
    if not x_signin.restart(session):
        raise HTTPException(status_code=404, detail="that sign-in is not running")
    return {"status": "restarted"}


@app.get("/x-signin/s/{session}/status")
def x_signin_status(session: str) -> dict:
    return {"status": x_signin.status(session, time.monotonic())}


@app.websocket("/x-signin/s/{session}/stream")
async def x_signin_stream(websocket: WebSocket, session: str) -> None:
    """The page's view of its desktop: raw RFB bytes to and from its loopback VNC server."""
    port = x_signin.vnc_port(session)
    if port is None:
        await websocket.close(code=4404)
        return
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        await websocket.close(code=1011)
        return
    await websocket.accept()
    try:
        await display.relay_vnc(websocket, reader, writer)
    except WebSocketDisconnect:
        pass
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# ── the connector ─────────────────────────────────────────────────────────────────

# Last, so every route above matches first: the connector's own paths are `/mcp` and its OAuth
# routes (`/.well-known/…`, `/authorize`, `/token`, `/register`, `/auth/callback`).
if _connector is not None:
    app.mount("/", _connector)
