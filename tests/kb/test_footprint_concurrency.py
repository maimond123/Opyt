"""Blog/Substack footprint ingest went from one serial loop to a producer pool — these pin the
properties that made that safe, none of which the existing wiring tests can see (they run one post
at a time, so a serial implementation passes them all).

Why it changed. Within a post, `content_gate` fans its batches across `_GATE_SEM`'s 4–8 permits and
then waits for all of them. Measured batch counts on a real 6-post site were 1, 2, 2, 3, 5, 8 — so
MOST pages produce fewer batches than there are lanes, the lanes idle, and the page's stage time is
the `max` of its calls rather than their sum spread over the lanes. Queueing posts keeps the lanes
fed. The LLM stages are ~97% of wall-clock; the fetch is ~0.9%.

The three properties that make it safe, one test each:

  1. the work actually overlaps               (otherwise the change bought nothing)
  2. the FETCH does not                       (the safety property — see below)
  3. a post is claimed once, before the fetch (the check-then-act the plan called out)

(2) is the one that would be expensive to get wrong. The original deferral of this work assumed
"process posts concurrently" required "fetch posts concurrently", which risks a Cloudflare soft-ban
the AIMD gate cannot detect — it only catches clean 429s. It does not require that: `run_concurrent`
calls `_next()` on the CALLING thread at both the priming loop and the refill point, so a lazily
fetching generator is drained strictly serially and the host sees the same request rate as before.
A regression here would be silent until a host banned us.

(3) is a check-then-act (TOCTOU) race, not a data race. CPython's GIL makes a single `dict[k] = v`
atomic — but it makes individual operations atomic, NOT sequences. The membership test and the mark
used to be separated by ~45 s of paid work, so a second thread would read stale state and pay for
the same post twice. The fix is thread confinement rather than a lock: both the test and the mark
live in the generator, which only the calling thread runs, and the mark records a CLAIM
(`PENDING_CLAIM`) rather than a completion.
"""

from __future__ import annotations

import threading
import time

import pytest

from pipeline.kb import ingest_blog as fp
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


_BLOG = "https://simonwillison.net"
_CONTENT = (
    "Autonomous agents compose small tools into larger systems, and the interesting engineering "
    "question is how you decompose a task into steps an agent can actually execute reliably. This "
    "post walks through a small agent framework and the tradeoffs it makes around retries and "
    "timeouts, with enough prose to clear the thin-body challenge guard comfortably."
)


def _url(i: int) -> str:
    return f"{_BLOG}/2024/01/post-{i}"


def _patch(monkeypatch, urls, *, fetch=None):
    """Patch discovery + the per-post fetch. `urls` is the sitemap order (duplicates allowed —
    UNION discovery really does surface the same post twice)."""
    from pipeline.ingestion.sources import blog as src

    def _default_fetch(url):
        return {"url": url, "title": f"Post {url[-1]}", "date": "2024-01-15", "content": _CONTENT}

    monkeypatch.setattr(src, "_fetch_sitemap_urls",
                        lambda base: [{"url": u, "lastmod": "2024-01-15"} for u in urls])
    monkeypatch.setattr(src, "harvest_hub_links", lambda base: [])
    monkeypatch.setattr(src, "_fetch_article", fetch or _default_fetch)
    monkeypatch.setattr(fp, "_feed_date_map", lambda base: {})


class _Overlap:
    """Records the maximum number of threads inside a block at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.threads: set[str] = set()

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            self.threads.add(threading.current_thread().name)
        return self

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1
        return False


# ── 1. the paid stages overlap ───────────────────────────────────────────────────


def test_paid_stages_run_concurrently(conn, fake_embedder, monkeypatch):
    """The point of the change. If this fails, posts are still serialized and the lanes still idle."""
    _patch(monkeypatch, [_url(i) for i in range(4)])
    overlap = _Overlap()
    real = fp.content_gate.classify_page

    def _slow_gate(markdown, **kw):
        with overlap:
            time.sleep(0.15)            # long enough that serial execution cannot fake an overlap
            return real(markdown, **kw)

    monkeypatch.setattr(fp.content_gate, "classify_page", _slow_gate)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert out["added"] == 4
    assert overlap.peak >= 2, f"posts still serialized (peak concurrency {overlap.peak})"


# ── 2. the fetch does NOT overlap ────────────────────────────────────────────────


def test_fetch_stays_serial(conn, fake_embedder, monkeypatch):
    """THE safety property. Parallel fetching risks a Cloudflare soft-ban the AIMD gate cannot see
    (it only catches clean 429s), and the failure would be silent until a host blocked us.

    `run_concurrent` calls `_next()` on the CALLING thread — at the priming loop AND at the refill
    point — so a lazily fetching generator is drained one at a time. This asserts that directly:
    peak fetch concurrency is 1, and every fetch happens on the thread that called us."""
    overlap = _Overlap()
    main = threading.current_thread().name

    def _slow_fetch(url):
        with overlap:
            time.sleep(0.05)
            return {"url": url, "title": "T", "date": "2024-01-15", "content": _CONTENT}

    _patch(monkeypatch, [_url(i) for i in range(6)], fetch=_slow_fetch)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert out["added"] == 6
    assert overlap.peak == 1, f"host saw {overlap.peak} concurrent fetches — request rate changed"
    assert overlap.threads == {main}, f"fetch left the calling thread: {overlap.threads}"


def test_db_writes_stay_on_the_calling_thread(conn, fake_embedder, monkeypatch):
    """The consumer is the single owner of the DB connection and the embed batch. sqlite3 connections
    are not shareable across threads by default, so this is a crash, not a subtle corruption — but it
    would only appear once two posts happened to finish together."""
    _patch(monkeypatch, [_url(i) for i in range(4)])
    main = threading.current_thread().name
    threads: set[str] = set()
    real_embed = fake_embedder.embed

    def _tracking_embed(texts, *, role="document"):
        threads.add(threading.current_thread().name)
        return real_embed(texts, role=role)

    monkeypatch.setattr(fake_embedder, "embed", _tracking_embed)
    fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert threads == {main}, f"the write path left the calling thread: {threads}"


# ── 3. claim-before-fetch ────────────────────────────────────────────────────────


def test_duplicate_url_is_fetched_once(conn, fake_embedder, monkeypatch):
    """UNION discovery (sitemap ∪ hub-harvest) surfaces the same post twice, which is exactly the
    check-then-act the claim exists to close. Before the claim, `seen` was marked only AFTER the
    paid work, so a duplicate arriving while the first copy was still in flight paid for a second
    fetch, a second gate, and a second VLM pass."""
    fetched: list[str] = []

    def _counting_fetch(url):
        fetched.append(url)
        return {"url": url, "title": "T", "date": "2024-01-15", "content": _CONTENT}

    dup = _url(0)
    _patch(monkeypatch, [dup, _url(1), dup], fetch=_counting_fetch)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert fetched.count(dup) == 1, f"duplicate post fetched {fetched.count(dup)}x"
    assert out["added"] == 2 and out["skipped"] == 1


def test_claim_is_not_a_hash(conn, fake_embedder, monkeypatch):
    """The claim is a placeholder, because the real hash does not exist until after the paid work.
    `snapshot_and_hash` compares `seen.get(atom_id) == raw_hash`, so the sentinel must never satisfy
    that — otherwise every post would read as 'unchanged' and silently stop being written."""
    from pipeline.kb.ingest_common import PENDING_CLAIM, snapshot_hash

    assert snapshot_hash("anything at all") != PENDING_CLAIM
    _patch(monkeypatch, [_url(0)])
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    assert out["added"] == 1 and out["skipped"] == 0, "the claim was mistaken for an unchanged hash"


# ── counters under concurrency ───────────────────────────────────────────────────


def test_limit_caps_dispatch_exactly(conn, fake_embedder, monkeypatch):
    """`limit` caps DISPATCH now, not atoms — the consumer lags the producer by up to `inflight`, so
    a consumer-side cap would over-spend a whole window of PAID work before noticing. The exact-fetch
    assertion is the point: an off-by-`inflight` here is money, not a rounding error."""
    fetched: list[str] = []

    def _counting_fetch(url):
        fetched.append(url)
        return {"url": url, "title": "T", "date": "2024-01-15", "content": _CONTENT}

    _patch(monkeypatch, [_url(i) for i in range(10)], fetch=_counting_fetch)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon", limit=3)

    assert len(fetched) == 3, f"limit=3 fetched {len(fetched)} posts"
    assert out["dispatched"] == 3 and out["added"] == 3


def test_producer_error_skips_one_post_and_is_counted(conn, fake_embedder, monkeypatch):
    """Fail-safe: one bad post skips, the run continues. Under the old serial loop an unexpected
    exception propagated and killed the whole sync, so this is strictly safer — but it opens a hole,
    because `run_concurrent` logs and drops the result without calling the consumer. `producer_failed`
    is the ONLY place such a post appears in the summary; without it the post vanishes from every
    counter and a broken page looks identical to a page that was never discovered."""
    _patch(monkeypatch, [_url(i) for i in range(4)])
    real = fp.content_gate.classify_page

    def _explode_on_one(markdown, **kw):
        if "post-2" in markdown:
            raise RuntimeError("gate blew up on this page")
        return real(markdown, **kw)

    monkeypatch.setattr(fp.content_gate, "classify_page", _explode_on_one)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert out["added"] == 3, "one bad post took others down with it"
    assert out["dispatched"] == 4
    assert out["producer_failed"] == 1, "the failed post vanished from the summary"


def test_counters_are_exact_under_concurrency(conn, fake_embedder, monkeypatch):
    """`+= 1` is a read-modify-write, which the GIL does NOT make atomic — that is why `_work`
    returns an outcome instead of tallying one. A lost update here would undercount silently, and
    `limit` reads one of these counters. Enough posts that a racy implementation would drop one."""
    _patch(monkeypatch, [_url(i) for i in range(12)])
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    assert out["dispatched"] == 12
    assert out["added"] == 12
    assert out["producer_failed"] == 0
    assert schema.count_atoms(conn, "blog") == 12
