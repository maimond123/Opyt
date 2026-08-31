"""pipeline/kb/embed.py — the hosted-embedding SEAM for the atom-KB.

ONE function, `get_kb_embedder()`, hands back an `Embedder` used for BOTH corpus
ingest and query. Routing both sides through the same object is what guarantees the
subspace invariant by construction: a query vector and a chunk vector are only
comparable (cosine means anything) when the identical model produced both.

Hosted OpenAI-compatible `/embeddings` client; default model is Qwen3-Embedding-8B
on OpenRouter. See docs/plans/2026-07-14-hosted-embedding-seam-handoff.md and

Three things that are easy to get wrong and are handled here:

1. **Asymmetric query prefix (the Qwen wrinkle).** Qwen embedding models were trained
   so the QUERY side is wrapped in an `Instruct: ...\nQuery: ...` tag and the DOCUMENT
   side is fed raw. Same model, same subspace — the tag is a role marker, not a second
   model. Omit it (or glue it to a chunk) and recall silently drops, no error. So
   `.embed()` takes a `role` and the seam applies the prefix ONLY to queries, ONLY for
   models that want it. Prefix-free models (OpenAI, the local fallback) ignore `role`.

2. **Model identity guard.** Vectors carry no self-describing model tag, so mixing two
   models in one store yields silent garbage. `kb_meta` records (model, dim, provider)
   on first ingest; `assert_model()` RAISES on any later mismatch. Switching model = a
   full re-embed, loudly, never a silent mix.

3. **Fail-safe contract (CLAUDE.md invariant).** A failed external call must SKIP — no
   partial write, no mark-processed. So a batch either returns a COMPLETE list of
   normalized vectors or RAISES; it never returns a half-filled list, and it records
   NO cost on failure. The caller catches and skips the unit of work.

Config: an optional `embeddings:` block in settings.yaml (parallel to `llm_backends`)
overrides the code defaults below. The seam is fully functional WITHOUT that block, so
it runs on a machine whose settings.yaml predates the atom-KB (distributable).

    embeddings:
      provider: openrouter
      model: <slug>          # default: model_routing.EMBED_MODEL
      endpoint: https://openrouter.ai/api/v1/embeddings
      dim: null            # null = discover from the API, don't hardcode
      batch_size: 64
      price_per_million: 0.01
      query_instruction: null   # null = derive from the model family
"""
from __future__ import annotations

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from opyt_core.config import merge_provider_routing
from pipeline.concurrency import AdaptiveSemaphore
from pipeline.model_routing import EMBED_MODEL

_OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
# Same browser UA as pipeline/llm_client.py — OpenRouter's Cloudflare edge 403s a bare python-urllib UA.
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Code DEFAULTS. A settings.yaml `embeddings:` block overrides any of these.
_DEFAULTS = {
    "provider": "openrouter",
    "model": EMBED_MODEL,
    "endpoint": _OPENROUTER_EMBED_URL,
    "dim": None,               # None = discover from the first response, never hardcode
    "batch_size": 64,
    "price_per_million": 0.01,  # qwen3-embedding-8b: $0.01 / 1M input tokens
    "query_instruction": None,  # None = derive from model family (see _resolve_config)
    # 30s: past that the backend is stalled, not slow.
    "timeout": 30.0,
    "max_attempts": 3,          # bounded retry on TRANSPORT failures only (see _embed_batch)
    # OpenRouter is a ROUTER: the embed model is served by multiple backends that are NOT
    # interchangeable in speed/precision/reliability. `order` ranks Nebius (full precision) ahead
    # of SiliconFlow (fp8); the hard exclusion is the `ignore` list applied by
    # `merge_provider_routing` (opyt_core.config), shared with the chat surface. See
    "provider_routing": {"order": ["Nebius", "SiliconFlow"], "allow_fallbacks": True},
}

# Qwen's query-side instruction. Documents are embedded RAW (no prefix). The task
# string is generic-retrieval phrasing per Qwen's model card, tuned to KB routing.
_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant knowledge-base passages "
    "that answer it\nQuery: "
)


class EmbedError(RuntimeError):
    """A hosted embedding call failed or returned a malformed/partial result.

    `retryable` marks a TRANSPORT-class failure (timeout, reset, 429/5xx) — the kind a fresh
    attempt can plausibly fix because it usually lands on a different upstream backend. A 4xx
    (bad key, bad model, malformed body) is NOT retryable: retrying just re-bills the same error."""

    def __init__(self, *args, retryable: bool = False, status: int | None = None):
        super().__init__(*args)
        self.retryable = retryable
        self.status = status          # HTTP status when known — lets the AIMD gate act only on 429s


class SubspaceError(RuntimeError):
    """A model-identity mismatch that would silently mix two vector subspaces."""


def _resolve_config() -> dict:
    """Merge code defaults with the optional settings.yaml `embeddings:` block.

    The query instruction is DERIVED from the model family when not set explicitly,
    so swapping the model in settings.yaml can't leave a stale Qwen prefix glued onto
    a prefix-free model (the silent-recall-loss footgun) — the prefix travels with the
    model, not with an independent config line."""
    cfg = dict(_DEFAULTS)
    try:
        from pipeline.ingestion.utils import load_yaml_config
        block = (load_yaml_config() or {}).get("embeddings") or {}
        # Only override with keys the block actually sets (None means "unset, use default"),
        # EXCEPT query_instruction/dim where an explicit null is a meaningful "derive it".
        for k, v in block.items():
            if k in cfg and v is not None:
                cfg[k] = v
    except Exception:
        pass  # no settings.yaml / no block -> pure defaults (fail-safe)
    if cfg.get("query_instruction") is None:
        cfg["query_instruction"] = (
            _QWEN_QUERY_INSTRUCTION if "qwen" in str(cfg["model"]).lower() else ""
        )
    return cfg


def _l2(raw) -> np.ndarray:
    """One raw embedding list -> an L2-normalized float32 vector (unit length, so a
    later dot product IS cosine similarity). A zero vector passes through unchanged."""
    v = np.asarray(raw, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ── HTTP (browser UA, Bearer; POOLED requests.Session for keep-alive) ─────────────

# One process-wide session so repeated embed calls reuse ONE pooled TCP+TLS connection instead
# of a fresh handshake each. Lazily built; a test that never embeds never allocates it.
_SESSION: "requests.Session | None" = None
_SESSION_LOCK = threading.Lock()   # guards lazy init — many embed threads may first-touch at once

# ── ARC-1 Phase 3: concurrent embed dispatch ──────────────────────────────────────
# `embed()` fans its 64-chunk sub-calls across a short-lived pool; this AIMD gate caps how many
# actually hit the provider at once and tunes the cap from feedback (climb on clean calls, halve
# on a 429). Process-wide singleton so every embed caller (ingest + query) shares ONE budget.
_EMBED_CONCURRENCY = 8
_EMBED_GATE = AdaptiveSemaphore(4, min_permits=2, max_permits=_EMBED_CONCURRENCY, increase_after=4)

# Last-seen OpenRouter rate-limit headers — Phase-2 PREP only. Sequential Phase 1 never contends,
# so this is inert now; it exists so the concurrent design can size its worker count against the
# real budget instead of guessing. Best-effort: reading a header must never break an embed.
_LAST_RATE_LIMIT: dict[str, str] = {}


def _get_session() -> "requests.Session":
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:            # double-checked: first-touch may be concurrent under fan-out
            if _SESSION is None:
                _SESSION = requests.Session()
    return _SESSION


def _note_rate_limit(resp) -> None:
    """Stash `X-RateLimit-*` and warn when the remaining budget runs low. Best-effort (Fail-safe:
    a header hiccup never fails the embed). Phase-2 informational — sequential Phase 1 won't trip it."""
    try:
        hdr = resp.headers
        remaining = hdr.get("X-RateLimit-Remaining")
        if remaining is None:
            return
        _LAST_RATE_LIMIT.update({
            "limit": hdr.get("X-RateLimit-Limit", ""),
            "remaining": remaining,
            "reset": hdr.get("X-RateLimit-Reset", ""),
        })
        rem, limit = int(remaining), int(hdr.get("X-RateLimit-Limit") or 0)
        if rem <= max(5, limit // 10):   # under ~10% of the window (or <5 absolute) → say so
            from pipeline.ingestion.utils import log
            log(f"[embed] rate-limit budget low: {rem}/{limit or '?'} remaining "
                f"(reset {hdr.get('X-RateLimit-Reset', '?')}).")
    except Exception:
        pass


def _http_post_json(req: urllib.request.Request, timeout: float) -> dict:
    """POST and parse JSON over the pooled session. Raises EmbedError on any HTTP/transport
    failure — the single seam tests monkeypatch, so every failure path is one type.

    `req` stays a `urllib.request.Request` only to carry url/data/headers and keep the test
    double's `(req, timeout)` shape stable; the transport is keep-alive `requests`. 408/429/5xx
    and transport/JSON errors are retryable; other 4xx are not."""
    session = _get_session()
    try:
        resp = session.post(req.full_url, data=req.data,
                            headers=dict(req.header_items()), timeout=timeout)
    except requests.RequestException as e:   # timeout, DNS, connection reset — a re-route may fix it
        raise EmbedError(f"{type(e).__name__}: {e}", retryable=True) from None
    _note_rate_limit(resp)
    if resp.status_code >= 400:
        body = (resp.text or "")[:500]
        raise EmbedError(f"HTTP {resp.status_code}: {body}",
                         retryable=resp.status_code in (408, 429, 500, 502, 503, 504),
                         status=resp.status_code) from None
    try:
        return resp.json()
    except ValueError as e:                  # malformed body — treat like a transport blip (retry)
        raise EmbedError(f"bad JSON: {e}", retryable=True) from None


def _maybe_breaker():
    """The OpenRouter-embed circuit breaker, or None if unavailable.

    A sustained embed outage trips it so a broad ingest can't retry-storm the bill.
    Best-effort: the breaker keeps state in a SQLite table that may not exist yet on a
    fresh rebuild, and a breaker hiccup must never break the tool (Fail-safe) — so any
    construction failure degrades to a direct, unwrapped call."""
    try:
        from pipeline.circuit_breaker import CircuitBreaker
        return CircuitBreaker("openrouter-embed")
    except Exception:
        return None


# ── Embedders ─────────────────────────────────────────────────────────────────


class HostedEmbedder:
    """OpenAI-compatible `/embeddings` client (OpenRouter by default).

    `dim` is DISCOVERED from the first response and locked thereafter, never hardcoded —
    robust to MRL truncation and model swaps. If the config pins `dim`, a discovered
    value that disagrees RAISES (catches a misconfigured model)."""

    def __init__(self, cfg: dict, *, use_breaker: bool = True):
        self.model = cfg["model"]
        self.provider = cfg["provider"]
        self.endpoint = cfg["endpoint"]
        self.batch_size = int(cfg["batch_size"])
        self.price_per_million = float(cfg["price_per_million"])
        self.query_instruction = cfg["query_instruction"] or ""
        self.timeout = float(cfg["timeout"])
        self.max_attempts = max(1, int(cfg.get("max_attempts") or 1))
        self.provider_routing = cfg.get("provider_routing")
        self._dim = cfg["dim"]           # None until discovered (or a pinned expectation)
        self._use_breaker = use_breaker

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed(self, texts: list[str], *, role: str = "document") -> list[np.ndarray]:
        """Embed `texts` -> list of L2-normalized float32 vectors, aligned to input order.

        `role="query"` applies the model's query instruction; `role="document"` (default)
        feeds raw text. All-or-nothing: returns a COMPLETE list or raises (never partial)."""
        if role not in ("query", "document"):
            raise ValueError(f"role must be 'query' or 'document', got {role!r}")
        if not texts:
            return []
        prefix = self.query_instruction if role == "query" else ""
        payload = [prefix + t for t in texts] if prefix else list(texts)

        bs = self.batch_size
        out: list[np.ndarray] = []
        if len(payload) <= bs:
            # One HTTP call — query, store_atom, or a per-atom re-embed. No pool, no threads.
            out = list(self._embed_batch(payload))
        else:
            # A batched flush: fire the 64-chunk slices concurrently, bounded by the AIMD gate
            # inside _embed_batch. Results are gathered in submission order so `out` stays aligned
            # to `payload` even though slices finish out of order. A slice that raises re-raises at
            # .result(), failing the whole call all-or-nothing.
            _get_session()                          # warm the pooled session once, before fan-out
            slices = [payload[s:s + bs] for s in range(0, len(payload), bs)]
            with ThreadPoolExecutor(max_workers=min(len(slices), _EMBED_CONCURRENCY),
                                    thread_name_prefix="embed") as ex:
                futures = [ex.submit(self._embed_batch, sl) for sl in slices]
                for f in futures:                   # in submission order → alignment preserved
                    out.extend(f.result())
        # Defensive: the per-batch check below already guarantees this, but assert the
        # whole-request contract too — a caller relies on 1 vector per input text.
        if len(out) != len(texts):
            raise EmbedError(f"expected {len(texts)} vectors, assembled {len(out)}")
        return out

    def _embed_batch(self, batch: list[str]) -> list[np.ndarray]:
        from pipeline.credentials import SERVICES, get_credential
        api_key = get_credential("openrouter")
        if not api_key:
            # Fail LOUD on a terminal, unrecoverable condition (no key) — not a silent skip.
            # Variable name comes from the credential registry, not a literal, so it can't go stale.
            raise EmbedError(f"{SERVICES['openrouter']} not set (no key in env or ~/.opyt/.env)")

        body = {"model": self.model, "input": batch, "encoding_format": "float"}
        # Deny-list merged in at request time (not baked into _DEFAULTS) so a settings.yaml
        # `embeddings.provider_routing` override can't drop the guard by replacing the whole block.
        prefs = merge_provider_routing(self.provider_routing or {})
        if prefs:
            body["provider"] = prefs
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _BROWSER_UA,
                "HTTP-Referer": "https://github.com/maimond123/Opyt",
                "X-Title": "opyt",
            },
        )

        def _do():
            # Bounded retry on TRANSPORT failures only. The final attempt still raises, so the
            # caller skips the unit of work — a batch never returns partial vectors.
            last: EmbedError | None = None
            for attempt in range(self.max_attempts):
                try:
                    return _http_post_json(req, self.timeout)
                except EmbedError as e:
                    last = e
                    if not e.retryable or attempt == self.max_attempts - 1:
                        raise
            raise last  # unreachable; keeps the type checker honest

        breaker = _maybe_breaker() if self._use_breaker else None
        # AIMD transport seam: caps concurrent embed calls and tunes the cap from feedback.
        # Decreases only on a 429 that survives the inner transport retry.
        with _EMBED_GATE:
            try:
                data = breaker.call(_do) if breaker is not None else _do()
            except EmbedError as e:
                if e.status == 429:
                    _EMBED_GATE.decrease()
                raise
        _EMBED_GATE.record_success()

        # OpenAI-compatible shape: {"data":[{"embedding":[...], "index":i}], "usage":{...}}.
        # Sort by "index" so vectors align to inputs even if the server reorders.
        rows = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        if len(rows) != len(batch):
            # Partial/short response -> RAISE, never return a misaligned subset.
            raise EmbedError(f"batch of {len(batch)} returned {len(rows)} embeddings")
        vecs = [_l2(r.get("embedding")) for r in rows]

        # Discover + lock the dimension. A drift mid-run means the server silently
        # changed models/config — that must not slip into the store.
        d = int(vecs[0].shape[0])
        if self._dim is None:
            self._dim = d
        elif int(self._dim) != d:
            raise EmbedError(f"embedding dim drift: config/first={self._dim}, got {d}")

        # Attribute spend into the shared api_stats.json (only on SUCCESS — a failed call
        # above raised before reaching here, so we never bill for a skipped write).
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        cost = tokens / 1_000_000 * self.price_per_million
        if cost > 0:
            try:
                from pipeline.llm_client import record_external_cost
                record_external_cost("openrouter-embed", cost, requests=1)
            except Exception:
                pass  # a stats hiccup must never fail the embed (Fail-safe)
        return vecs


class PrecomputedEmbedder:
    """A query vector the CALLER already computed, wearing the embedder interface.

    The third way `search` gets a query vector, and the one that exists for a knowledge base
    served over HTTP: the reader embeds their own query — with the model this store's `kb_meta`
    names, via `embedder_for_store` on their own install — and only the 4,096 floats cross the
    wire. So the machine hosting the export never holds an embedding key and never pays per
    query, and the query TEXT never has to reach it for the vector arm to run.

    `retrieve.atom_semantic_search` asks its embedder for a vector; this one already has it.
    `role` is accepted and ignored — the instruction prefix belongs to whoever did the embedding,
    and applying one here would move the vector out of the subspace it was built in.

    VALIDATED AT CONSTRUCTION, once, because the vector arrived over a network. The width is the
    checkable half: a wrong-width vector does not fail here, it fails in `mat @ qn` several frames
    deeper, where the message names neither the store nor the caller. The other half is NOT
    checkable — a right-width vector from the wrong model produces confident garbage and no
    exception — so a caller that cannot say which model it used has no business supplying one.
    That is why the store's model and dim are the reader's to read (`kb_meta`), not the server's
    to guess."""

    def __init__(self, conn, vector):
        meta = read_kb_meta(conn)
        if meta is None:
            raise SubspaceError(
                "this knowledge base records no embedding model, so it holds no vectors for a "
                "query vector to be compared against.")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != meta["dim"]:
            raise SubspaceError(
                f"this knowledge base was embedded with {meta['model']!r} at {meta['dim']} "
                f"dimensions; the query vector supplied has {vec.shape[0]}.")
        self.model = meta["model"]
        self.provider = meta["provider"]
        self.query_instruction = meta["query_instruction"]
        self.dim = meta["dim"]
        self._vector = vec / (np.linalg.norm(vec) + 1e-9)

    def embed(self, texts: list[str], *, role: str = "document") -> list[np.ndarray]:
        return [self._vector for _ in texts]


def get_kb_embedder(*, use_breaker: bool = True) -> HostedEmbedder:
    """The ONE embedder for corpus + query. Always the hosted client — there is no local/offline
    fallback, so a machine with no API key fails loudly (EmbedError) rather than writing empty or
    garbage vectors."""
    return HostedEmbedder(_resolve_config(), use_breaker=use_breaker)


def embedder_for_store(conn, *, use_breaker: bool = True) -> HostedEmbedder:
    """An embedder that lands a query in THIS store's subspace, not this install's.

    The foreign-read counterpart of `get_kb_embedder`. Reading somebody else's knowledge base
    (`kb=` — pipeline/kb/peers.py) inverts the model question: locally, a store whose recorded
    model differs from the configured one is a MISCONFIGURATION and `assert_model` rightly raises;
    on a peer's store it is the routine case, because the model is a fact the OWNER recorded when
    they built their corpus. So the store's `kb_meta` wins here, and the reader embeds their own
    query, in the owner's subspace, with their own key — no key custody by anyone.

    This CONSTRUCTS into the subspace rather than checking afterwards, which is why the foreign
    read path calls no `assert_model`: model, dim and query_instruction come from the row it would
    be compared against, so the comparison could only ever fail on a bug in these four lines.
    `storage_dtype` is deliberately not consulted — it is a WRITE guard against mixing blob widths
    in one store, and the vector arm decodes with `stored_dtype(conn)`, so a peer written by an
    older build reads perfectly well.

    A store with no `kb_meta` row has no vectors either, so there is nothing to be comparable with
    and the local embedder is returned unchanged.

    This is the FILE-shaped caller of `embedder_from_meta`: it knows how to get the meta out of a
    connection, and that function knows what to do with one — including the `provider` axis, which
    is the one thing that cannot follow the store. Its rule is stated there, once."""
    meta = read_kb_meta(conn)
    if meta is None:
        return get_kb_embedder(use_breaker=use_breaker)
    return embedder_from_meta(meta, use_breaker=use_breaker)


def embedder_from_meta(meta: dict, *, use_breaker: bool = True) -> HostedEmbedder:
    """The same rule as `embedder_for_store`, given the meta rather than a store to read it from.

    Split out 2026-08-27 for the reader of a REMOTE knowledge base, who has no file to open: their
    peer is an HTTPS prefix, and the owner's model reaches them as a dict from
    `GET /v1/kb/{owner}/meta`. Both callers land a query in the owner's subspace with the reader's
    own key, so nobody holds anybody else's credentials.

    `provider` is the one axis that cannot follow the meta: it is paired with `endpoint` in the
    config, and this install has credentials and a URL for its own only. A mismatch RAISES naming
    BOTH providers rather than guessing an endpoint, because a guessed endpoint returns vectors
    from some other model — confident garbage, not an error. The caller degrades to the keyword
    arm, which needs no vectors at all."""
    cfg = _resolve_config()
    if meta["provider"] != cfg["provider"]:
        raise SubspaceError(
            f"this knowledge base was embedded on {meta['provider']!r} with "
            f"{meta['model']!r}; your install speaks {cfg['provider']!r}, which has no endpoint "
            f"that produces vectors in that space."
        )
    cfg["model"] = meta["model"]
    cfg["dim"] = meta["dim"]
    cfg["query_instruction"] = meta["query_instruction"]
    return HostedEmbedder(cfg, use_breaker=use_breaker)


# ── chunk vector STORAGE width ───────────────────────────────────────────────────
#
# Half precision, measured lossless on the real corpus.
# This is purely what gets WRITTEN: the API response arrives full-precision (`encoding_format:
# "float"`), `_l2` computes in float32, and cosine is always computed in float32
# (`retrieve.atom_semantic_search`) — the width is a storage choice, never an arithmetic one.
CHUNK_STORAGE_DTYPE = "float16"


# ── kb_meta: the model-identity guard ────────────────────────────────────────────

_KB_META_DDL = """
CREATE TABLE IF NOT EXISTS kb_meta (
    id                INTEGER PRIMARY KEY CHECK (id = 1),   -- single-row table
    embed_model       TEXT    NOT NULL,
    embed_dim         INTEGER NOT NULL,
    provider          TEXT    NOT NULL,
    query_instruction TEXT    NOT NULL DEFAULT '',
    storage_dtype     TEXT    NOT NULL DEFAULT 'float32',   -- chunks.vector blob width
    strip_version     TEXT    NOT NULL DEFAULT '',          -- embed_surface.STRIP_VERSION
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def read_kb_meta(conn) -> dict | None:
    """This store's embedding identity, or None if it has never been embedded.

    Public since 2026-08-27, when `service/app.py` gained `GET /v1/kb/{owner}/meta` — a remote
    reader embeds their own query, so the model this store was built with is a fact they have to
    be able to ask for. Nothing outside this file may WRITE the row; `assert_model` and its
    siblings below own that."""
    conn.execute(_KB_META_DDL)
    # Additive, zero-migrations (schema._ensure_column): a store written before storage_dtype
    # existed gains the column defaulted to 'float32' — which is exactly what its blobs are.
    from .schema import _ensure_column
    _ensure_column(conn, "kb_meta", "storage_dtype", "TEXT NOT NULL DEFAULT 'float32'")
    # Defaults to '' — "these vectors were embedded from chunks.text verbatim", which is the truth
    # about every store written before embed_surface existed.
    _ensure_column(conn, "kb_meta", "strip_version", "TEXT NOT NULL DEFAULT ''")
    row = conn.execute(
        "SELECT embed_model, embed_dim, provider, query_instruction, storage_dtype, strip_version "
        "FROM kb_meta WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    return {"model": row[0], "dim": int(row[1]), "provider": row[2], "query_instruction": row[3],
            "storage_dtype": row[4], "strip_version": row[5]}


def stored_dtype(conn) -> str:
    """The width `chunks.vector` blobs were WRITTEN at in this store — what a reader must
    `np.frombuffer` with. Fail-safe to 'float32' for a store with no kb_meta row yet (nothing
    is stored either, so the reader has nothing to get wrong)."""
    meta = read_kb_meta(conn)
    return (meta or {}).get("storage_dtype") or "float32"


def ensure_kb_meta(conn, model: str, dim: int, provider: str,
                   query_instruction: str = "",
                   storage_dtype: str | None = None,
                   strip_version: str | None = None) -> dict:
    """Record the store's embedding identity on FIRST ingest; enforce it thereafter.

    Absent -> write it. Present & matching -> no-op. Present & DIFFERENT -> SubspaceError.

    Identity = (model, dim, provider, query_instruction, storage_dtype, strip_version). Each axis
    guards a different failure: model/dim/provider define the vector space (mismatch needs a full
    re-embed); query_instruction only shifts how queries project (no re-embed needed);
    storage_dtype guards mixed blob widths, which is cheap to fix locally via
    `convert_chunk_storage_dtype`; strip_version guards the text surface a chunk was embedded
    from and is checked here on the write path only, not in `assert_model` (see its docstring).
"""
    # Resolved at CALL time (not as a def default, which would freeze the width at import and
    # leave the writer and the guard able to disagree).
    storage_dtype = storage_dtype or CHUNK_STORAGE_DTYPE
    if strip_version is None:
        from .embed_surface import DEFAULT_PROFILE
        from .embed_surface import strip_version as _sv
        strip_version = _sv(DEFAULT_PROFILE)
    existing = read_kb_meta(conn)
    incoming = {"model": model, "dim": int(dim), "provider": provider,
                "query_instruction": query_instruction or "",
                "storage_dtype": storage_dtype, "strip_version": strip_version}
    if existing is None:
        conn.execute(
            "INSERT INTO kb_meta (id, embed_model, embed_dim, provider, query_instruction, "
            "storage_dtype, strip_version) VALUES (1, ?, ?, ?, ?, ?, ?)",
            (model, int(dim), provider, query_instruction or "", storage_dtype, strip_version),
        )
        conn.commit()
        return incoming
    if existing != incoming:
        raise SubspaceError(
            f"kb_meta says {existing} but embedder is {incoming} — identity drift. "
            f"A model/dim/provider change needs a full corpus re-embed; a "
            f"query_instruction change needs a deliberate re-confirm; a storage_dtype change "
            f"needs `pipeline.kb.embed.convert_chunk_storage_dtype(conn)` (local, free); a "
            f"strip_version change needs `python3 scripts/restrip_embed_surface.py --apply` "
            f"(re-embeds the corpus, ~$0.02)."
        )
    return existing


def assert_model(conn, embedder, *, storage_dtype: str | None = None) -> None:
    """RAISE if `embedder` (or this build's storage width) disagrees with the store's identity.

    Called on every ingest AND query. A fresh store (no kb_meta yet) is fine — the first ingest
    writes it via ensure_kb_meta. Model+provider+query_instruction are always checked; dim only
    once the embedder has discovered it. `storage_dtype` is also checked here, before the adapter
    pays to embed a run it could only corrupt.

    `strip_version` is deliberately NOT checked here (unlike `ensure_kb_meta`): this function runs
    on every query and a strip mismatch doesn't make results wrong, only stale-quality, so it must
    not take search down."""
    storage_dtype = storage_dtype or CHUNK_STORAGE_DTYPE
    existing = read_kb_meta(conn)
    if existing is None:
        return  # fresh store; nothing recorded to violate yet
    qi = getattr(embedder, "query_instruction", "") or ""
    if (existing["model"] != embedder.model
            or existing["provider"] != embedder.provider
            or existing["query_instruction"] != qi):
        raise SubspaceError(
            f"query/ingest embedder {{'model': {embedder.model!r}, "
            f"'provider': {embedder.provider!r}, 'query_instruction': {qi!r}}} "
            f"!= store {existing}"
        )
    if embedder.dim is not None and existing["dim"] != int(embedder.dim):
        raise SubspaceError(
            f"embedder dim {int(embedder.dim)} != store dim {existing['dim']}"
        )
    if existing["storage_dtype"] != storage_dtype:
        raise SubspaceError(
            f"this build stores chunk vectors as {storage_dtype}, but the store holds "
            f"{existing['storage_dtype']} blobs — writing into it would mix two blob widths and "
            f"the vector arm would reshape them into garbage. Convert first (local, free, no "
            f"re-embed): `from pipeline.kb.embed import convert_chunk_storage_dtype; "
            f"convert_chunk_storage_dtype(conn)`."
        )


def assert_strip_version(conn, strip_version: str | None = None) -> None:
    """RAISE if this build's embed-surface strip disagrees with the one the store was built from.

    The before-spend half of the strip guard: `ensure_kb_meta` already refuses the write, but not
    until the END of a flush, after the batch is paid for. Called once, where a batched ingest
    begins (`AtomSink.__init__`), so an un-migrated store costs one exception, not one corpus of
    embeddings."""
    if strip_version is None:
        from .embed_surface import DEFAULT_PROFILE
        from .embed_surface import strip_version as _sv
        strip_version = _sv(DEFAULT_PROFILE)
    existing = read_kb_meta(conn)
    if existing is None:
        return  # fresh store; the first write stamps it
    if existing["strip_version"] != strip_version:
        from .embed_surface import DEFAULT_PROFILE
        # `--profile` is always named explicitly, never left implicit, so a re-embed under the
        # wrong profile can't silently stamp an identity the guard would still refuse.
        raise SubspaceError(
            f"this build embeds from a strip_version {strip_version!r} surface, but the store's "
            f"vectors were built from {existing['strip_version'] or '(unstripped)'!r} — writing "
            f"into it would leave the corpus holding two generations of document vectors, "
            f"comparable to each other only by accident. Re-embed first: "
            f"`python3 scripts/restrip_embed_surface.py --profile {DEFAULT_PROFILE} --apply` "
            f"(re-embeds only the chunks whose surface actually changed)."
        )


def convert_chunk_storage_dtype(conn, target: str | None = None) -> int:
    """Rewrite every `chunks.vector` blob to `target` width and re-stamp kb_meta. Returns the
    number of rows converted.

    Local and free — a re-cast of vectors already held, not a re-embed. Idempotent: a store
    already at `target` is a no-op. One transaction, so a crash leaves the store at its original
    width; kb_meta and the blobs never disagree.

    Converts every vector-bearing table, not just `chunks` — also `probe_chunks`, via
    `probe_store` (per the `.guards.py` trust boundary) inside the same transaction, so kb_meta
    never claims a width one of the tables doesn't actually hold."""
    from . import probe_store          # lazy: probe_store imports this module's ensure_kb_meta

    target = target or CHUNK_STORAGE_DTYPE
    current = stored_dtype(conn)
    if current == target:
        return 0
    src, dst = np.dtype(current), np.dtype(target)
    rows = conn.execute(
        "SELECT chunk_id, vector FROM chunks WHERE vector IS NOT NULL").fetchall()
    n = 0
    for r in rows:
        cid, blob = (r["chunk_id"], r["vector"]) if hasattr(r, "keys") else (r[0], r[1])
        v = np.frombuffer(blob, dtype=src).astype(dst)
        conn.execute("UPDATE chunks SET vector = ? WHERE chunk_id = ?", (v.tobytes(), cid))
        n += 1
    n += probe_store.convert_probe_chunk_dtype(conn, src, dst)   # same transaction, by design
    conn.execute("UPDATE kb_meta SET storage_dtype = ? WHERE id = 1", (target,))
    conn.commit()
    return n


# ── standalone proof (live; spends a few cents) ──────────────────────────────────


def _smoke() -> int:
    """Prove the seam end-to-end against the LIVE API: real vectors, correct dim,
    unit-normalized, query prefix applied, cost recorded, kb_meta guard fires.

    Run:  python -m pipeline.kb.embed   (needs OPENROUTER_API_KEY in env or ~/.opyt/.env)
    """
    import sqlite3

    emb = get_kb_embedder(use_breaker=False)
    print(f"[smoke] embedder: model={emb.model} provider={emb.provider}")

    docs = ["Transformers use self-attention.", "The Fed sets interest rates."]
    q = ["how do neural networks weigh their inputs?"]
    dvecs = emb.embed(docs, role="document")
    qvecs = emb.embed(q, role="query")

    dim = dvecs[0].shape[0]
    dnorm = float(np.linalg.norm(dvecs[0]))
    print(f"[smoke] dim={dim}  ||doc||={dnorm:.4f}  n_docs={len(dvecs)}  n_q={len(qvecs)}")
    assert len(dvecs) == 2 and len(qvecs) == 1, "vector count mismatch"
    assert abs(dnorm - 1.0) < 1e-3, "vectors are not L2-normalized"

    # Retrieval sanity: the ML query should sit closer to the ML doc than the Fed doc.
    sims = [float(qvecs[0] @ d) for d in dvecs]
    print(f"[smoke] cos(query, ML doc)={sims[0]:.4f}  cos(query, Fed doc)={sims[1]:.4f}")
    assert sims[0] > sims[1], "query is not closer to the on-topic document"

    # kb_meta guard: write identity, then a mismatched embedder must RAISE.
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, emb.model, dim, emb.provider, emb.query_instruction)
    assert_model(conn, emb)  # matches -> ok
    class _Other:
        model, provider, dim = "openai/text-embedding-3-large", "openrouter", 3072
    try:
        assert_model(conn, _Other())
        raise SystemExit("[smoke] FAIL: mismatch did not raise")
    except SubspaceError:
        print("[smoke] kb_meta guard raised on model mismatch — correct")

    from pipeline.llm_client import spend_total
    print(f"[smoke] lifetime spend now: ${spend_total():.6f}")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
