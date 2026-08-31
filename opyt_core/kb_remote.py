"""
opyt_core/kb_remote.py — the HTTP branch of the `kb=` entry points.

A registered peer whose location is an https:// base URL is somebody's knowledge base served by
`service/app.py`. The three functions here do for that peer what `_open_kb` does for a file
peer: return the same envelope. The envelope crosses the wire VERBATIM — the service is a thin
adapter over the same entry points (the fidelity contract) — so nothing downstream of a
search/open/aggregate call can tell which transport answered.

The reader embeds their own query (I10): `search` fetches the owner's embedding identity from
`GET /meta`, embeds with the reader's own key via `embedder_from_meta`, and sends the floats.
The meta is cached per process — the MCP server is plain in-process stdio, so process ==
session — and a stale entry (the owner re-embedded at a new width and re-uploaded) surfaces as
a 400 naming dimensions, which drops the cache entry and retries once with fresh meta.

Every transport failure becomes `peers.PeerUnavailable` with the cause in the message, so the
tool layer answers a dead service the way it answers a missing file: an empty envelope and a
sentence (P3).
"""
from __future__ import annotations

import requests
from requests import RequestException

from pipeline.kb import peers
from pipeline.kb.embed import EmbedError, SubspaceError, embedder_from_meta

_TIMEOUT = 30   # seconds, on the network call — the one place a timeout belongs

_META_CACHE: dict[str, dict] = {}   # location → the owner's kb_meta, per process


class _BadRequest(peers.PeerUnavailable):
    """A 400 from the service, kept distinguishable so `search` can retry a stale-meta width
    mismatch. A subclass of `PeerUnavailable` on purpose: anywhere it is NOT caught specially,
    it is still the one failure type the tool layer absorbs."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def error_detail(r) -> str:
    """The service's own error sentence when there is one. A reverse proxy answering for a dead
    upstream sends HTML, not the service's JSON — hence the fallback to raw text. Public because
    every client of the service parses failures the same way — this module and the console
    scripts (`opyt-redeem`, `opyt-push`)."""
    try:
        return str(r.json().get("detail"))
    except Exception:
        return (r.text or "")[:200]


def _call(row: dict, path: str, payload: dict | None = None, *, method: str = "post") -> dict:
    """One HTTP exchange with the service, from a peer registry row.

    401 gets its own sentence: the token was revoked (a server-side row delete, effective
    immediately), and the fix — a new grant code from the owner — is named rather than implied.
    Everything else that is not a 2xx becomes `PeerUnavailable` with the cause in the message,
    except a 400, which stays inspectable for `search`'s stale-meta retry."""
    url = f"{row['location']}/{path}"
    headers = {"Authorization": f"Bearer {row['token']}"} if row.get("token") else {}
    try:
        if method == "get":
            r = requests.get(url, headers=headers, timeout=_TIMEOUT)
        else:
            r = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    except RequestException as e:
        raise peers.PeerUnavailable(
            f"the service at {row['location']} could not be reached ({e}).") from e
    if r.status_code == 401:
        raise peers.PeerUnavailable(
            "the service refused this install's token — it was revoked, or was never granted. "
            "Ask the owner for a new grant code and redeem it to restore access.")
    if r.status_code == 400:
        raise _BadRequest(error_detail(r))
    if not 200 <= r.status_code < 300:
        raise peers.PeerUnavailable(f"the service answered {r.status_code}: {error_detail(r)}")
    return r.json()


def _query_vector(row: dict, query: str) -> tuple[list[float] | None, dict | None]:
    """`(the reader's own query embedding, degrade notice)` — at most one is non-None.

    The owner's model comes from `GET /meta` (cached) and the embedding runs on the reader's
    own key, so the service holds no embedding key and pays nothing per query. A provider this
    install cannot reach (`SubspaceError`) or an embed call that fails (`EmbedError`) degrades
    to the keyword arm, with the SAME notice code the file-peer path emits — hosts treat the
    two transports identically."""
    meta = _META_CACHE.get(row["location"])
    if meta is None:
        meta = _call(row, "meta", method="get")
        _META_CACHE[row["location"]] = meta
    try:
        vec = embedder_from_meta(meta).embed([query], role="query")[0]
    except (SubspaceError, EmbedError) as e:
        from opyt_core.kb import _vector_arm_notice   # lazy: kb imports this module at top
        return None, _vector_arm_notice(e)
    return [float(x) for x in vec], None


def search(row: dict, query: str, *, k: int, mode: str, **filters) -> dict:
    """The served counterpart of the local search body in `run_kb_search`: embed, POST, return
    the envelope verbatim (plus the degrade notice, if the vector arm could not run — appended
    exactly where the local path appends its own).

    `filters` are the `SearchBody` field names; the server normalizes them the same way the
    local path does, because it calls the same function."""
    vector, notice = (None, None) if mode not in ("hybrid", "semantic") \
        else _query_vector(row, query)
    if notice is not None:
        mode = "bm25"   # the arm that needs no vectors; `notice` says why the other is gone
    payload = {"query": query, "k": k, "mode": mode, "query_vector": vector, **filters}
    try:
        envelope = _call(row, "search", payload)
    except _BadRequest as e:
        if vector is None or "dimensions" not in e.detail:
            raise
        # The cached meta went stale: the owner re-embedded at a new width and re-uploaded.
        # Fresh meta, re-embed, retry ONCE — a second 400 propagates as PeerUnavailable.
        _META_CACHE.pop(row["location"], None)
        vector, notice = _query_vector(row, query)
        if notice is not None:
            mode = "bm25"
        payload = {"query": query, "k": k, "mode": mode, "query_vector": vector, **filters}
        envelope = _call(row, "search", payload)
    if notice is not None:
        envelope.setdefault("notices", []).append(notice)
    return envelope


def open_atom(row: dict, atom_id: str) -> dict:
    return _call(row, "open", {"atom_id": atom_id})


def aggregate(row: dict, scope: dict | None) -> dict:
    return _call(row, "aggregate", {"scope": scope})
