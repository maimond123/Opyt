"""
pipeline/kb/ingest_common.py — the shared atom write-path every source ingester funnels through.

One place owns how an atom lands in the DB (`store_atom`: the atom row and its body, in one
transaction), so a source ingester only differs in how it renders to markdown and derives fields.
"""

from __future__ import annotations

from contextvars import copy_context

import hashlib
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Callable

from . import owed_links, schema, writes

# Image/media by file extension (end of path, before an optional query/fragment).
_IMG_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|avif|bmp|ico|tiff?)(?:[?#]|$)", re.I)
# Known image-CDN / proxy hosts + path markers (Substack's fetch/upload proxies, etc.).
_IMG_CDN_RE = re.compile(r"substackcdn\.com|/image/fetch/|/image/upload/", re.I)


def looks_like_image_url(url: str) -> bool:
    """Is this URL plausibly an image asset? Extension match or a known image-CDN/proxy shape
    (Substack's CDN URLs carry no extension). Used both to exclude images from `outbound_links`
    and to pick what the VLM enricher reads."""
    u = url or ""
    return bool(_IMG_EXT_RE.search(u) or _IMG_CDN_RE.search(u))


# Fetch-outcome classification (shared by the scrape-backed adapters): a fetch has three
# outcomes, so "host challenged us" stays distinct from "genuinely no body".
FETCH_OK = "ok"                      # real content — proceed
FETCH_ABSENT = "absent"              # confirmed empty (podcast / link post / stub) — skip, count
FETCH_UNDETERMINED = "undetermined"  # we could not tell (challenge / transport failure) — skip, alarm

MIN_BODY_CHARS = 200        # below this a "body" is a stub/interstitial, never a real post
CHALLENGE_MAX_CHARS = 600   # a marker only means "challenge" on a short body; a real essay that
                            # mentions captchas/JS is long
CHALLENGE_MARKERS = (
    "just a moment", "checking your browser", "enable javascript",
    "captcha", "attention required", "verify you are human",
)
# Cloudflare sets this response header when it issues a challenge: the one machine-readable
# signal. Absent on non-Cloudflare WAFs, so the body heuristic stays as the fallback.
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
        if text:
            hay = f"{title} {text}".lower()
            if any(m in hay for m in CHALLENGE_MARKERS):
                return FETCH_UNDETERMINED
        return FETCH_ABSENT
    if len(text) < challenge_max_chars:
        hay = f"{title} {text}".lower()
        if any(m in hay for m in CHALLENGE_MARKERS):
            return FETCH_UNDETERMINED
    return FETCH_OK


# Atom-level completeness: is the stored body the whole thing? A truncated body looks identical
# to a short one, so it is stored as data (state + basis), not inferred.
BODY_COMPLETE = "complete"   # the stored body is the whole thing
BODY_PARTIAL = "partial"     # real content, knowingly short of the whole — preview, abstract, teaser
BODY_ABSENT = "absent"       # stored without a body — podcast, link post, README-less repo
BODY_PENDING = "pending"     # stored without a body because we were stopped — retryable

# How the state above is known. `assumed` is the weakest claim: no evidence either way, distinct
# from a source-stated or an observed fact.
BASIS_STATED = "stated"      # the source declared it (audience, has_fulltext, post type)
BASIS_OBSERVED = "observed"  # we determined it from what came back (fetch failed, README missing)
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


# Counters worth surfacing to a caller. `dispatched` vs `added` is the honest read on `limit`
# (caps posts dispatched, not atoms added); `producer_failed` is the only place a post that
# vanished mid-run is counted. `fetched`/`stale`/`covered`/`capped` are adapter-specific (X's
# request unit; GitHub's three reasons a repo was never fetched) — the list is the union worth
# surfacing. `capped` is the work a bounded run still owes.
RUN_STAT_KEYS = ("added", "dispatched", "producer_failed", "undetermined",
                 "skipped", "failed", "paywalled",
                 "fetched", "stale", "covered", "capped")

# Diagnostics (dict-valued), kept separate from the int counters above: RUN_STAT_KEYS is what a
# user is told, this is what an engineer reads (wall-clock breakdown, serving upstream).
RUN_DIAG_KEYS = ("stage_seconds", "stage_latency", "llm_call_latency", "llm_upstreams")


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
    """The p-th percentile (0–100) of an already-sorted list, nearest-rank. Empty → 0.0."""
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
        # Many producer threads time into the same timer; a lost `+=` would undercount a stage.
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
        """Per-stage {count, mean, p50, p95, max} in seconds."""
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


def store_atom(conn, atom: dict, text: str, *,
               on_written: Callable[[], None] | None = None) -> None:
    """Write one atom and its body `text` (the snapshot) in one transaction; inside a caller's
    `write_tx` it joins that transaction. `on_written` fires once the atom is durable.

    The atom row carries the new `raw_hash` that every adapter's hash check reads as done, so it
    never lands without its body. Every KB atom passes here, so this is where a blog, Substack or
    GitHub post first records the paper links it owes (`owed_links.long_form_links`)."""
    if owed := owed_links.long_form_links(atom, text):
        atom["payload"]["links_owed"] = owed
    with writes.write_tx(conn, site="atom"):
        schema.upsert_atom(conn, atom)
        schema.put_body(conn, atom["atom_id"], text)
    if on_written:
        writes.after_commit(conn, on_written)


# The one machine lane an atom can be promoted out of. Named explicitly (an allow-list), so a mode
# added later is never promoted by default.
_PROMOTABLE_FROM = "frontier"


def promote_atom(conn, atom_id: str, entry_mode: str) -> None:
    """Flip a machine-found atom to the human-attested mode of the ingest that just touched it.

    One-way: the WHERE clause only matches a `frontier` row. `entry_mode` is the caller's mode; a
    non-human mode is a no-op, so every presence-hit site can call this unconditionally.
    `promoted_at` is stamped because `latest` and search's `since` key on `COALESCE(promoted_at,
    first_seen)`. Returns nothing: promotion is never reported to a user.
    """
    if entry_mode not in schema.HUMAN_ATTESTED:
        return                           # a machine-lane ingest attests nothing
    with writes.write_tx(conn, site="atom.promote"):
        conn.execute(
            "UPDATE atoms SET entry_mode = ?, promoted_at = datetime('now') "
            " WHERE atom_id = ? AND entry_mode = ?", (entry_mode, atom_id, _PROMOTABLE_FROM))


# ── Long-form producer-pool policy (blog + Substack) ───────────────────────────────
# Far smaller than `OPYT_INGEST_WORKERS` (X): provider fan-out is capped downstream by the
# content-gate and VLM semaphores, so this bounds how many posts are in flight, and its cost is RAM.
POST_WORKERS = int(os.environ.get("OPYT_POST_WORKERS", "4"))


def run_concurrent(items, work_fn, consume_fn, *, workers: int,
                   inflight: int | None = None) -> dict:
    """Fan `items` across `workers` producer threads and feed each non-None result to `consume_fn`.

    Results leave through the caller's closures; the return value reports what went wrong:
    `{"producer_failed": int, "consumer_failed": int, "source_error": Exception | None}`.
    `source_error` is returned unclassified, since only the source knows whether it means blocked
    or error; already-produced work still lands. `producer_failed` counts `work_fn` raises.

    `work_fn(item)` runs on pool threads. `consume_fn(result)` and advancing the source both run
    on the calling thread (contract: a rate-limited generator stays serialized), so the consumer
    is the single owner of the connection. A bounded window (`inflight`, default
    4×workers) caps peak RAM; results are consumed in submission order.

    A `work_fn` or `consume_fn` that raises is logged and skips that one item."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.ingestion.utils import log

    inflight = inflight or max(workers * 4, workers + 1)
    report: dict = {"producer_failed": 0, "consumer_failed": 0, "source_error": None}

    # A raise, distinct from a legitimate `None`; returned as a sentinel so it is counted on the
    # calling thread rather than from a pool thread.
    _FAILED = object()

    def _safe_work(item):
        try:
            return work_fn(item)
        except Exception as e:                       # one bad item skips; the run goes on
            log(f"[kb] producer error (item skipped): {e}")
            return _FAILED

    src = iter(items)
    _EXHAUSTED = object()

    def _next():
        """Pull the next item, or `_EXHAUSTED`. A source error stops intake but does not discard
        the already-produced window — recorded, then drain what's already fetched."""
        try:
            return next(src)
        except StopIteration:
            return _EXHAUSTED
        except Exception as e:
            log(f"[kb] source iterator error — stopping intake, draining the window: {e}")
            report["source_error"] = e               # the only thing separating this from a
            return _EXHAUSTED                        # finished source; a raised generator is
                                                     # closed, so a later next() = StopIteration

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest") as ex:
        pending: "deque" = deque()
        for _ in range(inflight):                    # prime the window
            item = _next()
            if item is _EXHAUSTED:
                break
            pending.append(ex.submit(copy_context().run, _safe_work, item))
        while pending:
            res = pending.popleft().result()         # in submission order (_safe_work never raises)
            item = _next()                           # top up before consuming → pool stays full
            if item is not _EXHAUSTED:
                pending.append(ex.submit(copy_context().run, _safe_work, item))
            if res is _FAILED:
                report["producer_failed"] += 1
                continue
            if res is None:
                continue
            try:
                consume_fn(res)
            except Exception as e:                   # one bad write skips; good atoms still land
                log(f"[kb] consumer error (result skipped): {e}")
                report["consumer_failed"] += 1
    return report


def snapshot_hash(markdown: str) -> str:
    """sha256 of the snapshot text: an atom's `raw_hash`, its change detector."""
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def changed_hash(atom_id: str, markdown: str, seen: dict[str, str]) -> str | None:
    """The snapshot's `raw_hash` when this atom is new or changed (the caller stores it), or None
    when `seen` already holds that hash."""
    raw_hash = snapshot_hash(markdown)
    return None if seen.get(atom_id) == raw_hash else raw_hash
