"""service/app.py — the HTTP surface: three read endpoints, an upload, and three credential ones.

A THIN ADAPTER, NOT A SECOND IMPLEMENTATION. Each read handler authenticates, clamps, and then
calls the exact function a local reader calls against a registered peer —
`opyt_core.kb.run_kb_search(..., kb=owner)` and its two siblings. Provenance, the `foreign_kb`
notice, the foreign-aware notice suppression and the `kb` key on every hit card all come from
Phase 2 unchanged, because this layer never reimplements any of it.

**The fidelity contract, and the one assertion that pins the whole hop:** for a request within
the caps, the response body equals `run_kb_search(..., kb=owner)` computed in-process against the
same export. If that ever needs special-casing, the Phase-2 seam was drawn in the wrong place.

WHAT CROSSES THE WIRE, AND WHAT "BLIND" MEANS. The semantic arm travels as an opaque vector: the
reader embeds their own query with the model the export's `kb_meta` names and sends the floats
(`query_vector` → `embed.PrecomputedEmbedder`), so this process holds no embedding key and pays
nothing per query (I10). **The keyword arm cannot work that way.** BM25 tokenizes the query
STRING, so on `mode="bm25"` and on the keyword half of `mode="hybrid"` the text reaches this
process. "Reader queries are blind" is therefore a RETENTION commitment — held by the `usage_daily`
schema having no query column, and by nothing here logging a request body — and not a property
the architecture enforces. Restricting foreign reads to `mode="semantic"` was rejected: the
keyword arm is exactly what still works when a reader's embedder cannot reach the owner's model,
and rare literal tokens are what it is for.

SYNC HANDLERS, AND WHY THE DEPLOY COMMAND NEEDS `--workers`. `run_kb_search` is blocking SQLite
plus NumPy, so the read routes are declared `def` and FastAPI runs them in a worker thread rather
than stalling the event loop. Threads do NOT buy throughput here: the ranking loop decodes and
max-pools in Python between NumPy calls, so it holds the GIL for most of its runtime and
concurrent queries in ONE process serialize. Measured on the real 2,805-atom export, 16 at once:
7,699 ms in a single process, 2,133 ms across four worker processes.

So throughput scales with PROCESSES, and the deploy command is where that is set. The shape of
it, with `<n>` = the box's CORE COUNT and never more, since processes time-slicing one core are
still one core (`service/DEPLOY.md` §3 is the real command, and today's box is one core):

    uvicorn service.app:app --workers <n> --limit-concurrency 32 \
            --log-config service/log_config.json

`--log-config` is not optional. Uvicorn's default access formatter writes `%(client_addr)s` —
every reader's IP against the knowledge base they read — which the collection policy refuses
(docs/plans/2026-08-27-what-opyt-collects.md). That file is the default config with the address
dropped from the access format, and nothing else changed.

Sizing, measured on the same export rather than estimated: ~59 MB baseline per worker, ~22 MB per
in-flight query, and ~66 MB of page cache per SERVED EXPORT (its vector column, which every
search scans in full) shared across all workers. The earlier estimate of ~100 MB per query was
about 4.5x too high. The limits live in the deploy command and not in this file, because a second
limit in the code would be a second source of truth for the same number.

Design record: docs/plans/2026-08-26-foreign-kb-service-phase3.md §3.1–§3.6.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from opyt_core import kb as kb_entry
from pipeline.kb import peers
from pipeline.kb.embed import PrecomputedEmbedder, SubspaceError, read_kb_meta

from . import store, uploads

# ── caps (§3.5) ──────────────────────────────────────────────────────────────────
#
# Throttling REQUESTS does not bound extraction — one call can return everything — so the
# response is what gets capped. Both values are deliberately generous: `k` past 100 is a
# distributional question that `aggregate` answers better, and the largest snapshot measured in
# the real corpus is 146 KB against a 3.4 KB mean, so the byte cap binds on almost nothing today.
# That is the point. The cap ships now so the bound is a number an operator can lower the day a
# stranger has access, rather than a change they have to design under pressure.
K_MAX = 100
OPEN_BYTES_MAX = 200_000

app = FastAPI(title="OPYT knowledge-base service")


# ── auth (§3.2) ──────────────────────────────────────────────────────────────────

def _bearer(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return authorization[7:].strip()


def _token(token: str = Depends(_bearer)) -> dict:
    """The row a bearer token names. 401 for anything else — including a REVOKED token, which is
    a row that is simply gone, so revocation takes effect on the very next request with no
    refresh cycle and no window.

    `store` opens and closes its own connection, so nothing here holds one across the dependency
    boundary. FastAPI resolves dependencies and runs sync handlers on DIFFERENT worker threads,
    and a `sqlite3` connection may only be used on the thread that created it — measured as a
    500 on the first genuinely concurrent request when the connection was passed in."""
    row = store.resolve_token(token)
    if row is None:
        raise HTTPException(status_code=401, detail="unknown or revoked token")
    return row


def _owner_token(auth: dict = Depends(_token)) -> dict:
    """An owner token — upload, grant, revoke. The knowledge base it acts on is the token's own
    `owner`, never a path segment, so there is no scope to compare and nothing to get wrong."""
    if auth["role"] != "owner":
        raise HTTPException(status_code=403, detail="this action needs an owner token")
    return auth


def _reader_of(owner: str, auth: dict = Depends(_token)) -> dict:
    """A reader token for the knowledge base named in the path.

    The `owner` column IS the scope, so a reader token for A reaching for B is one string
    comparison. Strict about the role in the other direction too: an owner token does not read
    here. The roles name what a token is FOR, an owner already holds the store the export was
    built from, and an owner who wants to exercise their own service redeems a grant like anyone
    else — which is the only way the reader path gets tested honestly."""
    if auth["role"] != "reader" or auth["owner"] != owner:
        raise HTTPException(status_code=403, detail="this token cannot read that knowledge base")
    return auth


def _served(owner: str, auth: dict = Depends(_reader_of)) -> dict:
    """A reader token AND an export to serve. 404 when there is nothing here yet.

    ⚠️THIS IS A BOUNDARY CHECK AND IT IS NOT OPTIONAL, because the read handlers call the LOCAL
    entry points, whose answer to an unreadable `kb=` is written for a person at their own
    install: it names every knowledge base registered here — on this box, that is every published
    routing key — and advises omitting `kb` to search their own store, which is not a thing a
    reader of a served export can do. Measured 2026-09-01, before this existed: `search`, `open`
    and `aggregate` all answered 200 with the service's whole registry inside the message, and
    only `meta` 404'd. The fix belongs HERE rather than in `opyt_core/kb.py`, because naming the
    registered peers is genuinely useful to the local caller it was written for; what is wrong is
    that this service was passing a local answer across a trust boundary without deciding
    anything first.

    ONE sentence for two states the service cannot tell apart and need not: an owner who has
    shared but whose export has not landed yet, and one whose first push failed. Both are ordinary
    now — `share` returns the invite immediately and pushes detached, so a link is live for the
    minute or two the upload takes — and the reader does the same thing in either case."""
    if not uploads.export_path(owner).exists():
        raise HTTPException(
            status_code=404,
            detail="this knowledge base has not arrived on the service yet. Its owner shared it, "
                   "and the copy usually lands within a minute or two of them doing so — nothing "
                   "is wrong with your install. Try again shortly.")
    return auth


# ── request bodies ───────────────────────────────────────────────────────────────

class SearchBody(BaseModel):
    """`run_kb_search`'s arguments, minus `kb` (the path says which) and `embedder` (that is
    `query_vector`, wrapped by the handler)."""
    query: str
    tags: list[str] | None = None
    what_kind: str | None = None
    source_type: str | None = None
    who_id: str | list[str] | None = None
    who: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    # Not just a filter — absent means the SECTIONED shape, with `frontier_atoms` beside `hits`.
    # It has to be forwarded rather than defaulted here, or a foreign search would answer a
    # different question than the same call against a local store (the fidelity contract).
    entry_mode: str | list[str] | None = None
    k: int = 8
    mode: str = "hybrid"
    # The reader's own query embedding. Present → this process never calls an embedder and never
    # pays; absent → the vector arm falls back to whatever `run_kb_search` can build locally,
    # which on a served export means the owner's model must also be this install's.
    query_vector: list[float] | None = None
    # R4: what the READER calls this knowledge base. The `{owner}` path segment is the routing
    # key and picks the export; this picks the string every `kb` field in the answer carries
    # back. Absent falls back to the routing key, which is what a pre-R4 client sends.
    as_kb: str | None = None


class OpenBody(BaseModel):
    atom_id: str
    as_kb: str | None = None


class AggregateBody(BaseModel):
    scope: dict | None = None
    as_kb: str | None = None


class GrantBody(BaseModel):
    label: str | None = None


class RedeemBody(BaseModel):
    code: str
    install_id: str | None = None


# Said BEFORE the first query, which is the only moment it is worth anything. Aggregation
# protects nobody at one reader — "this knowledge base served 40 reads" IS that reader's activity
# when they are the only one — so the honest mechanism at this scale is telling them rather than
# a scheme. Weak as disclosures go: an MCP client consumes this, not a person, which is why the
# same text is in TELEMETRY.md and on the public stats page and not only here.
_REDEEM_NOTICE = (
    "This service counts reads per day per reader so the owner can see usage and revoke access; "
    "it never records query text, IP addresses, or which atoms you read. "
    "Details: TELEMETRY.md in the Opyt repo."
)


class RevokeBody(BaseModel):
    token_sha256: str


class RegisterBody(BaseModel):
    label: str | None = None


# ── the query endpoints (§3.1) ───────────────────────────────────────────────────

def _query_embedder(owner: str, vector: list[float] | None):
    """A `PrecomputedEmbedder` for this export, or None to let `run_kb_search` decide.

    Opens the peer store a second time — once here for `kb_meta`, once inside `run_kb_search` for
    the query — which is a few microseconds against a ~230 ms read, and buys the validation
    happening at the boundary the vector arrived through rather than several frames deeper.

    An unopenable peer returns None rather than raising: `run_kb_search` is about to open the
    same store, fail the same way, and return the empty envelope with the sentence that explains
    it. Two error paths for one cause is how a caller ends up reading two shapes."""
    if vector is None:
        return None
    try:
        conn, _label = peers.open_peer(owner)
    except peers.PeerUnavailable:
        return None
    try:
        return PrecomputedEmbedder(conn, vector)
    except SubspaceError as e:
        # The reader sent a vector this store cannot use. A 400, not a degrade: they know which
        # model they embedded with and can fix it, and silently dropping to the keyword arm would
        # hide a subspace mismatch behind plausible results.
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


@app.get("/v1/kb/{owner}/meta")
def meta(owner: str, auth: dict = Depends(_served)) -> dict:
    """The embedding identity of this export — what a remote reader must embed their query with.

    Read from the served file on demand and never stored. `kb_meta` changes the day the owner
    re-embeds and re-uploads, so a copy kept anywhere else — in `service.db`, or handed out once
    in the `redeem` response — would be a second home for a fact the export file owns, and it
    would go stale silently: a reader embedding at the old width gets a 400 from `search` with no
    way to learn why."""
    try:
        conn, label = peers.open_peer(owner)
    except peers.PeerUnavailable as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    try:
        m = read_kb_meta(conn)
    finally:
        conn.close()
    if m is None:
        raise HTTPException(status_code=404,
                            detail="this knowledge base records no embedding model")
    return {"owner": owner, "label": label, "model": m["model"], "dim": m["dim"],
            "provider": m["provider"], "query_instruction": m["query_instruction"]}


@app.post("/v1/kb/{owner}/search")
def search(owner: str, body: SearchBody,
           auth: dict = Depends(_served)) -> dict:
    envelope = kb_entry.run_kb_search(
        body.query, tags=body.tags, what_kind=body.what_kind, source_type=body.source_type,
        who_id=body.who_id, who=body.who, date_from=body.date_from, date_to=body.date_to,
        entry_mode=body.entry_mode, k=min(body.k, K_MAX), mode=body.mode, kb=owner,
        as_kb=body.as_kb, embedder=_query_embedder(owner, body.query_vector),
    )
    store.record_usage(owner, auth["token_sha256"], "search",
                       zero_results=not envelope["hits"])
    return envelope


@app.post("/v1/kb/{owner}/open")
def open_atom(owner: str, body: OpenBody,
              auth: dict = Depends(_served)) -> dict:
    envelope = kb_entry.kb_open(body.atom_id, kb=owner, as_kb=body.as_kb)
    raw = envelope.get("raw")
    if isinstance(raw, str):
        encoded = raw.encode("utf-8")
        if len(encoded) > OPEN_BYTES_MAX:
            # `body_state` needs no new signalling: it exists precisely so a truncated snapshot
            # is never quoted as a whole article, every consumer already reads it, and "partial"
            # already means exactly this. Decoded with errors="ignore" because the cut lands mid
            # codepoint often enough — losing a character beats returning invalid UTF-8.
            envelope["raw"] = encoded[:OPEN_BYTES_MAX].decode("utf-8", "ignore")
            envelope["body_state"] = "partial"
    store.record_usage(owner, auth["token_sha256"], "open")
    return envelope


@app.post("/v1/kb/{owner}/aggregate")
def aggregate(owner: str, body: AggregateBody,
              auth: dict = Depends(_served)) -> dict:
    # No cap applied, and that is a finding rather than an omission: every list `kb_aggregate`
    # returns is already `LIMIT`ed in its own SQL (15/15/12) and its two dicts are keyed on
    # closed enums, so there is no unbounded surface here to clamp. Adding a truncation branch
    # that can never fire would be a cap in name only. The bound is asserted at this boundary
    # instead — see tests/service/test_caps.py — so a future edit that raises those LIMITs has to
    # decide about this endpoint rather than silently widening it.
    envelope = kb_entry.kb_aggregate(scope=body.scope, kb=owner, as_kb=body.as_kb)
    store.record_usage(owner, auth["token_sha256"], "aggregate")
    return envelope


# ── upload (§3.3) ────────────────────────────────────────────────────────────────

@app.post("/v1/upload/{owner}")
async def upload(owner: str, request: Request,
                 auth: dict = Depends(_owner_token)) -> dict:
    """Raw body = the export file. Replaces what is served, atomically.

    `async` — unlike the read routes — because the body arrives as an async stream and the whole
    point is never to hold a 115 MB file in memory. The writes are blocking and do stall the
    event loop for the length of a local disk write; that is accepted rather than threaded away,
    because an upload happens once per sync run by one person while a query happens constantly.

    Scope comes from the TOKEN, and the path segment must agree — an owner token cannot replace
    somebody else's export by changing the URL. The token must also HOLD the name's claim
    (`store.claim_holder`): agreeing strings are not enough, because nothing in `tokens` ever
    made `owner` unique and two tokens sharing a name must not share the served file."""
    if auth["owner"] != owner:
        raise HTTPException(status_code=403, detail="this token cannot upload that knowledge base")
    if store.claim_holder(owner) != auth["token_sha256"]:
        # A second owner token for the same name — a population `mint_token` refuses to create,
        # but a pre-claims database can already hold. This comparison, not the mint refusal, is
        # what stops that token replacing the claim holder's knowledge base.
        raise HTTPException(status_code=403,
                            detail="this name is published under a different owner token")
    try:
        rx = uploads.Receiver(owner)
    except uploads.BadOwner as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except uploads.NoSpace as e:
        # 507, not 500: the request is well-formed and the owner did nothing wrong. It says the
        # STORAGE is the problem, so a client knows retrying later is the right move and
        # shrinking the export is not.
        raise HTTPException(status_code=507, detail=str(e)) from e
    try:
        async for chunk in request.stream():
            rx.write(chunk)
    except uploads.TooLarge as e:
        # 413, and the previously served export is still answering — `abort` touches only the
        # temp file. The owner CAN act on this one, which is why it is not the same code as a
        # full disk.
        rx.abort()
        raise HTTPException(status_code=413, detail=str(e)) from e
    except BaseException:
        # Including a disconnect mid-body. The previously served export is untouched, because
        # `commit` is the only statement that would have replaced it.
        rx.abort()
        raise
    result = rx.commit(label=auth["label"])
    # AFTER the commit, so the recorded size always describes a file that is really being served.
    store.record_upload(owner, result["bytes"])
    return result


# ── register / grant / redeem / revoke (§3.4) ────────────────────────────────────

@app.post("/v1/register")
def register(body: RegisterBody) -> dict:
    """A new knowledge base: `{owner, token}`, unauthenticated, returned once.

    R5: anyone with Opyt installed can publish. `DEPLOY.md` §7 used to say there was deliberately
    no such endpoint, because one that hands out owner tokens hands out the right to publish. Its
    rationale was largely the permanent NAME claim, which R4 deleted by making the key an
    assigned address rather than a name anybody would type. What remains is resource abuse, and
    R5a rules that a quota question rather than a gate question — handled after the fact, by the
    operator, reading `/v1/stats`.

    UNMETERED on purpose (R5a). No rate limit, no identity, no verification: every candidate is a
    weaker imitation of the human gate R5 removed deliberately, and the damage they prevent is
    bounded, cheap and reversible. The transition rule for the paid tier is that the free tier
    must be a LIMIT and not a GATE — going paid raises a cap, so nobody loses a capability and
    there is no grandfather cohort to negotiate.

    `label` is the owner's display name, and it is what every reader gets back from `redeem` as
    `suggested_name`. It is not a claim, not unique, and never routes.

    The on-box `mint_token` stays as the operator fallback and as the rotation path for a leaked
    token; this endpoint is the one a person reaches."""
    owner, token = store.register_owner(body.label)
    return {"owner": owner, "token": token}


@app.post("/v1/grant")
def grant(body: GrantBody, auth: dict = Depends(_owner_token)) -> dict:
    """Mint a one-time code, returned once. The owner sends it however they like; because it
    buys exactly one reader token and then dies, what sits in that chat window afterwards is not
    a standing credential."""
    return {"owner": auth["owner"], "code": store.mint_grant(auth["owner"], label=body.label)}


@app.post("/v1/redeem")
def redeem(body: RedeemBody) -> dict:
    """Exchange a code for a reader token. THE REAL CLIENT ENTRY POINT: on redemption the client
    writes its own peer row from what comes back here — `share_tools.accept` in an assistant, or
    `opyt-redeem` from a terminal.

    `owner` is the ROUTING key — opaque, in the URL path, and not something anybody would type.
    `suggested_name` is what the owner called themselves, and it is what the client registers the
    peer under. A SUGGESTION, deliberately: `peers.add` may suffix it if this reader already
    knows another `alex`, and nothing about serving depends on it being unique, because R4 moved
    uniqueness onto the routing key. It can be null — an owner registered without a label — and
    the client then falls back to the routing key.

    NOT returned: a base URL. The client knows the URL it POSTed to, and synthesizing one here
    would mean reading proxy headers to find out what the world calls this service."""
    try:
        owner, token = store.redeem_grant(body.code, body.install_id)
    except store.GrantUnavailable as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"owner": owner, "token": token, "suggested_name": store.owner_label(owner),
            "notice": _REDEEM_NOTICE}


@app.post("/v1/revoke")
def revoke(body: RevokeBody, auth: dict = Depends(_owner_token)) -> dict:
    """One row delete, effective on the next request."""
    return {"revoked": store.revoke(auth["owner"], body.token_sha256)}


@app.post("/v1/unpublish")
def unpublish(auth: dict = Depends(_owner_token)) -> dict:
    """Stop sharing this knowledge base: every reader cut off AND the served copy deleted.

    ONE act, not two, because a person who says "stop sharing my KB" means both halves and will
    not think to say it twice. Each half alone is a wrong end state: revoking every token leaves
    the export on this disk with no reader and no removal path, and deleting the file alone
    leaves live tokens meeting 404s that read like an outage rather than a decision.

    READERS FIRST, then the file. A failed unlink then leaves readers already cut off (safe) and
    an orphaned file that a retry clears; the reverse order would open a window where live tokens
    reach for a file that is gone. This is not one transaction and cannot be — a SQLite delete
    and a filesystem unlink have no shared commit — so the ordering is what makes the crash-in-
    the-middle state the harmless one.

    Takes no body: the token names the knowledge base, the same way `grant` and `revoke` do.
    """
    readers = store.revoke_all_readers(auth["owner"])
    served = uploads.remove(auth["owner"])
    # The bytes are gone, so the accounting must say so — a stored-bytes total that only ever
    # climbs is not a disk-usage number. The row stays: `first_published_at` is the fact that
    # cannot be recovered.
    store.clear_upload(auth["owner"])
    return {"owner": auth["owner"], "readers_revoked": readers, "export_deleted": served}


@app.get("/v1/tokens")
def tokens(auth: dict = Depends(_owner_token)) -> dict:
    """Who currently holds a token for this knowledge base — the list `revoke` takes its handle
    from. Hashes only, because that is all this service has.

    Also carries R3's push gate: `last_upload_at` and `reads_since_last_upload`. They ride HERE
    rather than on an endpoint of their own because `push` already calls this first — it is how
    the routing key is derived from the token — so the gate costs the rail no extra round trip.
    The rail pushes when someone has READ since the last push AND the local store has CHANGED
    since it; this answers the first, and the second is a question about the owner's own store
    that this service has no view of."""
    return {"owner": auth["owner"], "tokens": store.list_tokens(auth["owner"]),
            **store.publish_demand(auth["owner"])}


# ── the public stats page (the collection ruling's transparency obligation) ───────

def _rollup() -> dict:
    """Every published number, composed from the two files that hold them: `service.db` (counts,
    tokens, codes) and the peers registry (how many knowledge bases are served). `store.py` owns
    the first and deliberately does not reach into the second, so the join happens here."""
    return {**store.stats_rollup(),
            "kbs_published": len(peers.list_peers()),
            "stored_bytes_by_kb": store.stored_bytes(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/v1/stats")
def stats() -> dict:
    """The roll-up as JSON. UNAUTHENTICATED on purpose: this is the transparency mechanism for
    what `TELEMETRY.md` says is collected, and a transparency page behind a credential is not
    one.

    `stored_bytes_by_kb` is the one per-knowledge-base list, and it rides the JSON rather than
    the page: it exists so an operator can see who is eating the disk and remove them by hand
    (R5a puts abuse handling after the fact, not at admission), and a notifier reads JSON. It
    carries the routing key and NO label — after R4 that key is an assigned address, not a name
    anybody chose. Nothing here is per-reader."""
    return _rollup()


# A page, not a dashboard. Numbers plus one sentence each, no charts and no series: a retention
# curve needs weeks of rows before it says anything, and shipping an empty one would claim a
# measurement nobody has. Self-contained by requirement — a stats page that fetched a font or a
# script would hand a third party the visitor list this whole design refuses to keep.
_STATS_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>OPYT sharing service — what it has served</title>
<style>
 body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }}
 dt {{ font-weight: 600; margin-top: 1.2rem; }}
 dd {{ margin: 0.1rem 0 0; color: #444; }}
 .n {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
 footer {{ margin-top: 3rem; color: #666; font-size: 0.9rem; }}
</style>
<h1>What this service has served</h1>
<p>Every number on this page is a total. No reader and no query appears here or in the database
behind it. Knowledge bases do appear, by their routing key — the address of a file, which
<code>register</code> now assigns at random, though a key claimed before it did so may still
read like a name; <code>/v1/stats</code> lists how much disk each one uses. See TELEMETRY.md in
the Opyt repository for the full schema.</p>
<dl>
 <dt class="n">{kbs_published}</dt><dd>knowledge bases published to this service.</dd>
 <dt class="n">{stored_mb}</dt><dd>of exports stored, across all of them.</dd>
 <dt class="n">{readers_total}</dt><dd>reader tokens currently held. Revoked ones leave this count
   and stay in the read totals below.</dd>
 <dt class="n">{codes_redeemed} / {codes_minted}</dt>
   <dd>grant codes redeemed, of those minted. The gap is invitations nobody used.</dd>
 <dt class="n">{reads_total}</dt><dd>reads served in total.</dd>
 <dt class="n">{reads_30d}</dt><dd>reads served in the last 30 days.</dd>
 <dt class="n">{active_readers_30d}</dt><dd>readers who read something in the last 30 days.</dd>
 <dt class="n">{by_tool}</dt><dd>reads by tool. Opens against searches is how often a result
   was worth reading in full.</dd>
 <dt class="n">{zero_rate}</dt><dd>of searches returned nothing.</dd>
</dl>
<footer>Generated {generated_at}.</footer>
"""


@app.get("/stats", response_class=HTMLResponse)
def stats_page() -> str:
    """The same numbers as `/v1/stats`, as a page a person can read. One template, filled from
    the one roll-up — there is no second query here, so the page and the JSON cannot disagree."""
    d = _rollup()
    rate = d["zero_result_rate"]
    return _STATS_PAGE.format(
        by_tool=", ".join(f"{v} {k}" for k, v in sorted(d["reads_by_tool"].items())) or "none yet",
        zero_rate="not yet known" if rate is None else f"{rate:.0%}",
        stored_mb=f"{d['stored_bytes_total'] / 1_000_000:,.0f} MB",
        **d)
