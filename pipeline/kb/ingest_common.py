"""
pipeline/kb/ingest_common.py — the shared atom write-path every source ingester funnels through.

One place owns snapshot -> chunks -> embed -> store, so a source ingester only differs in how it
renders to markdown and derives fields, never in how an atom lands in the DB.

Two entry points onto the same store:
  - `store_atom` — synchronous submit+flush of ONE atom, durable on return.
  - `AtomSink` — batches the embed across MANY atoms (used by ingest_x): pools chunks from many
    atoms into fewer, fuller embed HTTP calls; same per-atom commit either way.

Fail-safe: embedding is paid and all-or-nothing — a failed batch raises rather than returning a
partial vector list. `store_atom`/`embed_chunks` propagate that so the caller skips the atom
entirely. `AtomSink` re-embeds per atom on batch failure so only the bad atom skips. An atom is
never written with missing vectors.
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager

import numpy as np

from . import schema
from .chunk import split_text, strip_frontmatter
from . import embed as _embed           # module, not the name: the width is read at CALL time, so
from .embed import EmbedError, assert_strip_version, ensure_kb_meta  # writer+guard never drift
from .embed_surface import DEFAULT_PROFILE, strip_for_embedding
from .raw_store import write_snapshot, snapshot_hash

# Image/media by file extension (end of path, before an optional query/fragment).
_IMG_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|avif|bmp|ico|tiff?)(?:[?#]|$)", re.I)
# Known image-CDN / proxy hosts + path markers (Substack's fetch/upload proxies, etc.).
_IMG_CDN_RE = re.compile(r"substackcdn\.com|/image/fetch/|/image/upload/", re.I)


def looks_like_image_url(url: str) -> bool:
    """Is this URL plausibly an IMAGE asset? Extension match OR a known image-CDN/proxy shape —
    both are needed since Substack's CDN URLs carry no extension. The VLM enricher includes only
    these. Extended rationale:
"""
    u = url or ""
    return bool(_IMG_EXT_RE.search(u) or _IMG_CDN_RE.search(u))


# fetch-outcome classification (shared by the scrape-backed adapters): a fetch has THREE
# outcomes but `str | None` only expresses two, so "host challenged us" was indistinguishable
# from "genuinely no body". Splitting them lets blog/Substack classify the same way and lets a
# future per-source concurrency gate react to a real block signal. See ingest_common.md.
FETCH_OK = "ok"                      # real content — proceed
FETCH_ABSENT = "absent"              # confirmed empty (podcast / link post / stub) — skip, count
FETCH_UNDETERMINED = "undetermined"  # we could not tell (challenge / transport failure) — skip, ALARM

MIN_BODY_CHARS = 200        # below this a "body" is a stub/interstitial, never a real post
CHALLENGE_MAX_CHARS = 600   # a marker only means "challenge" on a SHORT body — a real essay that
                            # merely MENTIONS captchas/JS is long, so gating avoids false skips
CHALLENGE_MARKERS = (
    "just a moment", "checking your browser", "enable javascript",
    "captcha", "attention required", "verify you are human",
)
# Cloudflare sets this response header when it issues a challenge. It is the one MACHINE-READABLE
# signal in the whole path — the body test below is a heuristic that a reworded challenge page can
# defeat, this is not. Absent on non-Cloudflare WAFs, so the body test stays as the fallback.
CF_MITIGATED_HEADER = "cf-mitigated"


def challenge_in_headers(headers) -> bool:
    """True iff the response headers carry Cloudflare's explicit challenge marker. Tolerates
    None / any mapping with case-insensitive lookup (requests and curl_cffi both give a
    case-insensitive mapping, but a plain dict from a test double does not)."""
    if not headers:
        return False
    try:
        for k, v in dict(headers).items():
            if str(k).lower() == CF_MITIGATED_HEADER:
                return bool(str(v).strip())
    except Exception:      # a header mapping we can't read is not evidence of a challenge
        return False
    return False


def classify_fetch(body: str | None, *, headers=None, title: str = "",
                   min_chars: int = MIN_BODY_CHARS,
                   challenge_max_chars: int = CHALLENGE_MAX_CHARS) -> str:
    """`body` → one of FETCH_OK / FETCH_ABSENT / FETCH_UNDETERMINED.

    Header wins over body inspection (a long challenge page would otherwise pass the marker gate).
    Below that, a short body carrying a challenge marker is UNDETERMINED; a short body without one
    is ABSENT. The length gate keeps a genuine essay about captchas from reading as a block."""
    if challenge_in_headers(headers):
        return FETCH_UNDETERMINED
    text = (body or "").strip()
    if not text or len(text) < min_chars:
        # A too-short body can still be a challenge shell, so sniff before calling it absent.
        if text and len(text) < challenge_max_chars:
            hay = f"{title} {text}".lower()
            if any(m in hay for m in CHALLENGE_MARKERS):
                return FETCH_UNDETERMINED
        return FETCH_ABSENT
    if len(text) < challenge_max_chars:
        hay = f"{title} {text}".lower()
        if any(m in hay for m in CHALLENGE_MARKERS):
            return FETCH_UNDETERMINED
    return FETCH_OK


# the ATOM-level completeness contract: `classify_fetch` answers "did the fetch succeed?"; this
# answers "is the body on disk the WHOLE thing?" — a truncated body looks identical to a short
# one, so it must be stored as data, not inferred. Two separate keys (state + basis), not one
# enum, mirroring `when_ts`/`when_precision`. See ingest_common.md.
BODY_COMPLETE = "complete"   # the stored body is the whole thing
BODY_PARTIAL = "partial"     # real content, knowingly short of the whole — preview, abstract, teaser
BODY_ABSENT = "absent"       # stored WITHOUT a body — podcast, link post, README-less repo
BODY_PENDING = "pending"     # stored without a body because we were STOPPED — retryable

# How the state above is KNOWN. `assumed` is the weakest claim: no evidence either way, distinct
# from a source-stated or an observed fact.
BASIS_STATED = "stated"      # the SOURCE declared it (audience, has_fulltext, post type)
BASIS_OBSERVED = "observed"  # WE determined it from what came back (fetch failed, README missing)
BASIS_ASSUMED = "assumed"    # no evidence either way — a default, and the weakest claim here

_BODY_STATES = frozenset({BODY_COMPLETE, BODY_PARTIAL, BODY_ABSENT, BODY_PENDING})
_BODY_BASES = frozenset({BASIS_STATED, BASIS_OBSERVED, BASIS_ASSUMED})


def body_fields(state: str, basis: str) -> dict:
    """`{"body_state": …, "body_basis": …}` — splat into an atom's payload.

    Every adapter writes both keys on every atom, even when the value is a constant, so the
    invariant stays testable instead of tribal knowledge. Raises on an unknown value so a typo
    never becomes a silent extra category."""
    if state not in _BODY_STATES:
        raise ValueError(f"unknown body_state {state!r} (expected one of {sorted(_BODY_STATES)})")
    if basis not in _BODY_BASES:
        raise ValueError(f"unknown body_basis {basis!r} (expected one of {sorted(_BODY_BASES)})")
    return {"body_state": state, "body_basis": basis}


# the RUN-level counterpart of classify_fetch: same question ("did we succeed or get blocked?")
# one level up, for a WHOLE adapter run, since footprint callers consumed a summary as
# unconditional success. See ingest_common.md.
RUN_INGESTED = "ingested"
RUN_BLOCKED = "blocked"
RUN_ERROR = "error"

# Counters worth surfacing to a caller. `dispatched` vs `added` is the honest read on `limit`
# (caps posts dispatched, not atoms added); `producer_failed` is the only place a post that
# vanished mid-run is counted. `fetched`/`stale` are adapter-specific (X's paid unit; GitHub's
# pushed_at gate skip) — the list is the union worth surfacing, not every adapter's intersection.
RUN_STAT_KEYS = ("added", "dispatched", "producer_failed", "undetermined",
                 "gate_rejected", "skipped", "failed", "paywalled",
                 "fetched", "stale")

# Diagnostics (dict-valued), kept separate from the int counters above: RUN_STAT_KEYS is what a
# user is told, this is what an engineer reads (wall-clock breakdown, serving upstream).
RUN_DIAG_KEYS = ("stage_seconds", "stage_latency", "llm_call_latency", "llm_upstreams")


def classify_run(summary) -> str:
    """An adapter run summary → RUN_INGESTED / RUN_BLOCKED / RUN_ERROR.

    Adapters signal a hard stop by returning a summary carrying `error` rather than raising, so a
    caller must check both. BLOCKED means the host stopped us mid-run (transient, retry); ERROR
    means a caller/input fault (needs a human). `undetermined` alone is not a block — only `error`
    demotes a run, since some adapters increment it per-post on an otherwise healthy run."""
    if not isinstance(summary, dict):
        return RUN_INGESTED                      # fail-safe: an unreadable summary is not evidence
    if not summary.get("error"):
        return RUN_INGESTED
    return RUN_BLOCKED if summary.get("undetermined") else RUN_ERROR


def run_stats(summary) -> dict:
    """The subset of `RUN_STAT_KEYS` + `RUN_DIAG_KEYS` this summary actually carries. Keys are
    copied only when present, not defaulted to 0 — a missing counter and a zero counter differ.
    Counters must be `int`, diagnostics must be `dict`; a wrong-shaped key is dropped, not passed
    on."""
    if not isinstance(summary, dict):
        return {}
    out = {k: summary[k] for k in RUN_STAT_KEYS if isinstance(summary.get(k), int)}
    out.update({k: summary[k] for k in RUN_DIAG_KEYS if isinstance(summary.get(k), dict)})
    return out


def _percentile(sorted_xs: list[float], p: float) -> float:
    """The p-th percentile (0–100) of an ALREADY-SORTED list, nearest-rank. Empty → 0.0."""
    if not sorted_xs:
        return 0.0
    k = int(round((p / 100.0) * (len(sorted_xs) - 1)))
    return sorted_xs[max(0, min(len(sorted_xs) - 1, k))]


class StageTimer:
    """Accumulating per-stage wall-clock AND per-entry latency samples. Totals answer where time
    went; samples answer each stage's call-shape (p50/p95/max), used to size worker pools. `with
    t.stage("x"): ...` adds elapsed to the total and records that entry's duration. Grain differs
    per stage (per-call, per-bookmark, or per-flush); an unused timer just collects unread
    samples."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}
        # Many producer threads can time into the SAME timer concurrently, so totals/samples
        # updates must be serialized or a lost `+=` silently undercounts a stage. Held for µs.
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self.totals[name] = self.totals.get(name, 0.0) + dt
                self.samples.setdefault(name, []).append(dt)

    def distribution(self) -> dict[str, dict]:
        """Per-stage {count, mean, p50, p95, max} in seconds — the Phase-2 worker-sizing input."""
        out: dict[str, dict] = {}
        for name, xs in self.samples.items():
            if not xs:
                continue
            s = sorted(xs)
            out[name] = {
                "count": len(s),
                "mean": round(sum(s) / len(s), 3),
                "p50": round(_percentile(s, 50), 3),
                "p95": round(_percentile(s, 95), 3),
                "max": round(s[-1], 3),
            }
        return out


# the atom write-path: chunk (cheap, local) is separated from embed (paid, batchable). Splitting
# chunk-at-submit from embed-at-flush is what lets `AtomSink` pool the embed call across atoms.
# See ingest_common.md for the measured HTTP-call reduction.


def _chunk_snapshot(text: str, source_type: str | None = None) -> list[dict]:
    """Snapshot text → chunk dicts with vectors LEFT `None` (embedding happens later, at flush).

    Chunks the body only — frontmatter is provenance, not content. Spans stay snapshot-absolute
    by adding the stripped prefix length back. Each chunk also carries `embed_text`: the same
    window with OPYT's own renderer output stripped (`embed_surface`) — that's what gets
    embedded, while `text` is what gets stored/rendered/span-indexed. `source_type` keys the
    stripping rules; `None` gets the conservative path. Extended rationale:
"""
    body, offset = strip_frontmatter(text)
    return [
        {"seq": s, "text": t,
         "embed_text": strip_for_embedding(t, source_type, DEFAULT_PROFILE),
         "char_start": a + offset, "char_end": b + offset, "vector": None}
        for (s, t, a, b) in split_text(body)
    ]


def _attach(chunks: list[dict], vecs) -> list[dict]:
    """Fill a chunk list's `None` vectors from an aligned vector slice (L2-normalized, narrowed to
    `CHUNK_STORAGE_DTYPE` → bytes). Caller guarantees `len(vecs) == len(chunks)`.

    The single writer of `chunks.vector` — every reader takes the width from `kb_meta`, so a
    second writer at a different width would make those disagree. Extended rationale:
"""
    dt = np.dtype(_embed.CHUNK_STORAGE_DTYPE)
    return [
        {**c, "vector": np.asarray(v, dtype=dt).tobytes()}
        for c, v in zip(chunks, vecs)
    ]


def embed_chunks(embedder, text: str, source_type: str | None = None) -> list[dict]:
    """`text` → chunk dicts with L2-normalized vectors as `CHUNK_STORAGE_DTYPE` bytes, ready for
    `schema.replace_chunks`. Single-atom chunk+embed (used by `rechunk.py`); the batched ingest
    path drives `AtomSink` instead. Documents embed RAW (`role="document"`, no query prefix), and
    embed `embed_text` — not `text` — since the two differ by renderer scaffolding."""
    chunks = _chunk_snapshot(text, source_type)
    vecs = embedder.embed([c["embed_text"] for c in chunks], role="document")
    return _attach(chunks, vecs)


def _write_atom(conn, embedder, atom: dict, chunks: list[dict]) -> None:
    """Durably write ONE atom + its (already-embedded) chunks. Both `upsert_atom` and
    `replace_chunks` commit, so batching only widens the pre-write window, never the write
    itself."""
    ensure_kb_meta(conn, embedder.model, int(embedder.dim), embedder.provider,
                   getattr(embedder, "query_instruction", "") or "")
    schema.upsert_atom(conn, atom)
    schema.replace_chunks(conn, atom["atom_id"], chunks)


class AtomSink:
    """Batches atoms across an ingest loop so embedding pools MANY atoms' chunks per HTTP call
    instead of one call per atom. `submit()` chunks locally and buffers; crossing `flush_chunks`
    triggers `flush()` — embed every buffered chunk, then write each atom with its own vector
    slice. `close()` flushes the remainder.

    Two invariants: positional alignment between the flat embed response and each atom's chunk
    span is assert-guarded (a misalignment crashes loudly, never stores a wrong vector), and a
    poisoned batch re-embeds per atom so only the bad atom skips. A written atom commits per-atom;
    a crash mid-buffer loses at most one buffer of producer work, redone next run. Extended
    rationale:"""

    def __init__(self, conn, embedder, *, flush_chunks: int | None = None,
                 timer: "StageTimer | None" = None, writer=None):
        self.conn = conn
        self.embedder = embedder
        # Before-spend strip guard, checked here (the seam every batched ingest passes through)
        # rather than per-ingester, and FIRST as a precondition so a failed construction leaves
        # nothing half-built.
        assert_strip_version(conn)
        # Target store as a FUNCTION rather than a table name, so the probe-store trust boundary
        # (untrusted content must never reach the main store) can't be broken by a wrong string
        # argument.
        self._write = writer or _write_atom
        # Buffer to 4×batch_size, not batch_size, so each flush's tail HTTP call is amortized
        # rather than paid every flush. batch_size stays a separate retry/latency knob.
        bs = int(getattr(embedder, "batch_size", 64) or 64)
        self._flush_chunks = int(flush_chunks) if flush_chunks else 4 * bs
        self._timer = timer or StageTimer()          # never-null: unused totals just go unread
        self._buf: list[tuple] = []                  # (atom, chunk_dicts, on_written)
        self._pending = 0

    def submit(self, atom: dict, snapshot_text: str, *, on_written=None) -> None:
        """Chunk `snapshot_text` locally and buffer the atom. Auto-flushes once the buffered chunk
        count reaches the threshold, bounding buffer RAM. `on_written` fires after this atom is
        durably written. `atom["source_type"]` keys the embed-surface strip."""
        chunks = _chunk_snapshot(snapshot_text, atom.get("source_type"))
        self._buf.append((atom, chunks, on_written))
        self._pending += len(chunks)
        if self._pending >= self._flush_chunks:
            self.flush()

    def flush(self) -> None:
        """Embed every buffered chunk in one call, then write each atom with its own vector slice."""
        if not self._buf:
            return
        buf, self._buf, self._pending = self._buf, [], 0   # detach first: flush() is re-entrant-safe
        flat: list[str] = []
        spans = []
        for atom, chunks, cb in buf:
            lo = len(flat)
            flat.extend(c["embed_text"] for c in chunks)   # the STRIPPED surface, not `text`
            spans.append((atom, chunks, cb, lo, len(flat)))
        try:
            with self._timer.stage("embed"):
                vecs = self.embedder.embed(flat, role="document")   # ONE call (auto-splits >64)
        except EmbedError:
            self._flush_isolated(spans)      # batch poisoned → re-embed per atom so good atoms write
            return
        with self._timer.stage("write"):
            for atom, chunks, cb, lo, hi in spans:
                assert hi - lo == len(chunks), (       # positional-alignment guard (loud, never silent)
                    f"embed span {hi - lo} != {len(chunks)} chunks for {atom['atom_id']}")
                self._write(self.conn, self.embedder, atom, _attach(chunks, vecs[lo:hi]))
                if cb:
                    cb()                                # bookkeep: seen/added/threads — durable-only

    def _flush_isolated(self, spans) -> None:
        """One flush's batch embed failed. Re-embed each atom ALONE so a single poison chunk skips
        only its own atom (no write, no `on_written` → not marked seen → retried next run), while
        every good atom still writes. Preserves the Fail-safe invariant (a failure skips its unit)."""
        for atom, chunks, cb, _lo, _hi in spans:
            try:
                with self._timer.stage("embed"):
                    vecs = self.embedder.embed([c["embed_text"] for c in chunks], role="document")
                with self._timer.stage("write"):
                    self._write(self.conn, self.embedder, atom, _attach(chunks, vecs))
                if cb:
                    cb()
            except EmbedError as e:
                try:
                    from pipeline.ingestion.utils import log
                    log(f"[kb] embed failed for {atom['atom_id']} — skipped (no write): {e}")
                except Exception:
                    pass

    def close(self) -> None:
        """Flush whatever remains. Call once at end of run (or use the sink as the store_atom body)."""
        self.flush()


# The one machine lane an atom can be promoted OUT of. Named explicitly rather than derived as
# "anything not human-attested", because the deny-list direction would promote whatever mode gets
# added next, silently — the same allow-list discipline `schema.HUMAN_ATTESTED` states for the
# other direction.
_PROMOTABLE_FROM = "frontier"


def promote_atom(conn, atom_id: str, entry_mode: str) -> None:
    """Flip a machine-found atom to the human-attested mode of the ingest that just touched it.

    RULED 2026-08-25 (docs/plans/2026-08-24-era-reads-claims-carry.md): any human-initiated
    ingest — a bookmark sweep, a hopper deposit — that presence-hits a frontier atom promotes it.
    Attestation is evidence a human cares, and depositing the URL IS that evidence.

    ONE-WAY, enforced in the WHERE clause rather than by callers remembering: the update only ever
    fires on a `frontier` row, so no frontier refresh can demote a human-attested atom (the
    repo-refresh overwrite hazard `frontier_admit` already guards, arriving from the other side).

    `entry_mode` is the CALLER's mode, never a hardcoded 'user-saved' — the column records how the
    atom was FOUND, so a footprint crawl colliding with a frontier atom promotes to
    `oracle-footprint`, and a hand deposit to `user-saved`. A NON-human mode is a no-op rather
    than an error, and that is what lets every presence-hit site call this unconditionally: the
    same mint helper serves both lanes (`atomize_paper` runs under `entry_mode='frontier'` for
    Frontier stage 3 and under 'user-saved' for a hopper deposit), so "was this ingest
    human-initiated" is already carried by the argument. Deciding it here rather than re-writing
    the same guard at every call site keeps one definition of the question.

    `promoted_at` is stamped in the same statement because promotion is what opens the wallet —
    the re-read trigger keys on `COALESCE(promoted_at, first_seen)`, and a frontier atom's
    `first_seen` is the crawl date, not the engagement.

    Promotion is a side effect, never a feature: no caller may report it, which is why this
    returns nothing. A presence-hit promotion answers exactly like a fresh save, because explaining
    the flip means teaching the lane taxonomy, and lanes live below the interface boundary.
    """
    if entry_mode not in schema.HUMAN_ATTESTED:
        return                           # a machine-lane ingest attests nothing
    conn.execute(
        "UPDATE atoms SET entry_mode = ?, promoted_at = datetime('now') "
        " WHERE atom_id = ? AND entry_mode = ?", (entry_mode, atom_id, _PROMOTABLE_FROM))
    conn.commit()


def store_atom(conn, embedder, *, atom: dict, snapshot_text: str) -> None:
    """Write one atom + its chunks, embedding first (the step that can fail/cost).

    Now a thin strangler-fig wrapper over a one-shot `AtomSink` (submit + immediate flush), so the
    old synchronous-durable contract — "the atom is written when this returns" — holds unchanged for
    every non-X call site. Only `ingest_x` drives the sink explicitly to capture the batching win."""
    sink = AtomSink(conn, embedder)
    sink.submit(atom, snapshot_text)
    sink.close()


def submit_atom(conn, embedder, sink: "AtomSink | None", *, atom: dict, snapshot_text: str,
                on_written=None) -> None:
    """Write one atom through a caller-owned `sink` when there is one, else synchronously.

    Lets a single-atom mint helper (`atomize_paper`, `github_atom_from_url`) join a caller's
    batch instead of forcing its own embed round-trip. `on_written(atom_id)` fires when the atom
    is durable; a callback, not a return value, since a submitted-but-unflushed atom is in RAM
    only and callers must be told when it lands rather than probing for it. Extended rationale:
"""
    if sink is not None:
        sink.submit(atom, snapshot_text,
                    on_written=(lambda: on_written(atom["atom_id"])) if on_written else None)
        return
    store_atom(conn, embedder, atom=atom, snapshot_text=snapshot_text)
    if on_written:
        on_written(atom["atom_id"])


# Long-form producer-pool policy (blog + Substack), shared so the two loops can't drift apart.
# Deliberately far smaller than `OPYT_INGEST_WORKERS` (20, for X): provider fan-out is already
# capped downstream by the content-gate and VLM semaphores, so this only bounds how many POSTS
# are in flight — and its cost is RAM, since a rendered blog post is far larger than a tweet.
def llm_run_marker() -> dict:
    """Take at run start; hand to `llm_run_stats` at summary time so the summary covers THIS run,
    not the process lifetime (the underlying latency stats are append-only module globals).
    Fail-safe: `{}` means "no offsets", read by `llm_run_stats` as the lifetime view."""
    try:
        from pipeline.llm_client import latency_marker
        return latency_marker()
    except Exception:
        return {}


def llm_run_stats(since: dict | None = None) -> dict:
    """LLM diagnostics for a run summary: per-call latency by role, and by serving upstream.
    Spread as `**` into the summary dict so every adapter reports the same keys. The upstream
    view makes `provider.sort`'s live re-ranking auditable — a slow upstream otherwise has no
    symptom. `since` is this run's `llm_run_marker()`; omit it for the process lifetime. Fail-safe:
    a diagnostic must never break a run summary."""
    try:
        from pipeline.llm_client import latency_distribution, upstream_distribution
        return {"llm_call_latency": latency_distribution(since),
                "llm_upstreams": upstream_distribution(since)}
    except Exception:
        return {"llm_call_latency": {}, "llm_upstreams": {}}


POST_WORKERS = int(os.environ.get("OPYT_POST_WORKERS", "4"))
# Submission window. `run_concurrent` defaults to 4x workers; halved here because each outstanding
# result holds a fully rendered post (snapshot + enriched markdown) rather than a tweet.
POST_INFLIGHT = POST_WORKERS * 2

# `seen` maps atom_id -> raw_hash, but the hash doesn't exist until after the paid work, and the
# producer must CLAIM an atom_id before yielding it or two threads pay for the same post. This
# placeholder holds that claim until the consumer upgrades it to the real hash; `seen` therefore
# holds non-hash values mid-run and must not be read as a hash ledger before the run completes.
PENDING_CLAIM = "\x00pending"


def make_consumer(sink: "AtomSink", seen: dict, counters: dict, on_mark) -> callable:
    """The shared `consume_fn` blog and Substack both hand `run_concurrent`. Upgrades a
    `PENDING_CLAIM` in `seen` to the real hash and writes via `sink`; adapter-specific counters
    stay local to each caller. `counters` is a dict (not `nonlocal` ints) since the returned
    closure is defined outside the caller's lexical scope. Needs int values at `"consumed"`,
    `"submitted"`, `"skipped"`, `"gate_rejected"`."""
    def _consume(res: dict) -> None:
        """Tally + write, SERIALLY on the calling thread — so the sink stays a single writer and
        every counter is owned by one thread."""
        counters["consumed"] += 1
        outcome = res["outcome"]
        if outcome == "gate_rejected":
            counters["gate_rejected"] += 1
            return
        if outcome == "unchanged":
            counters["skipped"] += 1
            return
        seen[res["atom_id"]] = res["raw_hash"]   # upgrade the generator's claim to the real hash
        counters["submitted"] += 1
        sink.submit(res["atom"], res["md_kept"], on_written=on_mark)
    return _consume


def run_concurrent(items, work_fn, consume_fn, *, workers: int, inflight: int | None = None):
    """Fan `items` across `workers` producer threads and feed each non-None result to `consume_fn`.

    `work_fn(item)` is the paid, network-bound per-item work and runs on a pool thread, many at
    once. `consume_fn(result)` is the write path (embed + DB) and runs serially on the calling
    thread, so it is the single owner of the connection and embed batch.

    CONTRACT: the source is advanced on the calling thread too (priming loop and refill point),
    guaranteed behavior, not an implementation detail — a caller can rely on a rate-limited
    generator staying serialized even while `work_fn` fans out. Pinned by
    `tests/kb/test_run_concurrent.py::test_source_is_advanced_only_on_the_calling_thread`.

    A bounded submission window (`inflight`, default 4×workers) keeps peak RAM at O(window),
    independent of `len(items)`. Results are consumed in submission order; a slow item delays
    only the consumer, never the producers. Extended rationale (incl. measured consumer-share
    numbers):

    Fail-safe: a `work_fn` or `consume_fn` that raises is logged and skips that one item."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.ingestion.utils import log

    inflight = inflight or max(workers * 4, workers + 1)

    def _safe_work(item):
        try:
            return work_fn(item)
        except Exception as e:                       # one bad item skips; the run goes on
            log(f"[kb] producer error (item skipped): {e}")
            return None

    src = iter(items)
    _EXHAUSTED = object()

    def _next():
        """Pull the next item, or `_EXHAUSTED`. A source error stops intake but does not discard
        the already-produced window — logged, then drain what's already fetched."""
        try:
            return next(src)
        except StopIteration:
            return _EXHAUSTED
        except Exception as e:
            log(f"[kb] source iterator error — stopping intake, draining the window: {e}")
            return _EXHAUSTED                        # a raised generator is closed → later next() = StopIteration

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest") as ex:
        pending: "deque" = deque()
        for _ in range(inflight):                    # prime the window
            item = _next()
            if item is _EXHAUSTED:
                break
            pending.append(ex.submit(_safe_work, item))
        while pending:
            res = pending.popleft().result()         # in submission order (_safe_work never raises)
            item = _next()                           # top up BEFORE consuming → pool stays full
            if item is not _EXHAUSTED:
                pending.append(ex.submit(_safe_work, item))
            if res is None:
                continue
            try:
                consume_fn(res)
            except Exception as e:                   # one bad write skips; good atoms still land
                log(f"[kb] consumer error (result skipped): {e}")


def snapshot_and_hash(source: str, atom_id: str, markdown: str,
                      seen: dict[str, str]) -> tuple[str, str] | None:
    """Compute the snapshot hash and decide whether to write.

    Returns `(raw_ref, raw_hash)` if this atom is NEW or CHANGED (caller should store),
    or None if unchanged (caller SKIPS — no re-embed, no re-write). Only writes the
    snapshot file when there's something to store, so an unchanged run touches no disk.
    """
    raw_hash = snapshot_hash(markdown)
    if seen.get(atom_id) == raw_hash:
        return None
    raw_ref, _ = write_snapshot(source, atom_id, markdown)
    return raw_ref, raw_hash
