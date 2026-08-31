"""ARC-1 Phase 2 — the concurrent ingest harness end to end.

Two layers:
  • `run_concurrent` in isolation — every item consumed exactly once, the bounded submission window
    never balloons, and a producer OR consumer that raises skips only its own item (the fail-safe
    contract), never aborting the run.
  • `sync_bookmarks` under real threads — the equivalence + single-writer proof: the concurrent path
    produces the SAME atoms/threads as a serial run (no lost/dup atoms), and because a producer that
    touched `conn` would raise SQLite's cross-thread error (→ item skipped → count short), a full
    count IS the single-writer guarantee. Also asserts the image-cache save survives concurrent
    writers (the "dict changed size during iteration" crash the guard exists to prevent).
"""
from __future__ import annotations

import threading
import time

from pipeline.kb.ingest_common import run_concurrent


# ── run_concurrent in isolation ──────────────────────────────────────────────

def test_consumes_every_item_exactly_once():
    got: list[int] = []                                  # consumer is single-threaded → plain append ok
    run_concurrent(range(200), lambda i: i * 10, got.append, workers=8)
    assert sorted(got) == [i * 10 for i in range(200)]   # all present, none duplicated or dropped


def test_source_is_advanced_only_on_the_calling_thread():
    """A CONTRACT of run_concurrent, not an incidental property — and one nothing else pinned.

    Callers rely on it to keep rate-limited work OUT of the pool. A Python generator runs on
    whichever thread calls `next()` on it and pauses at each `yield`, so a caller that puts an HTTP
    fetch inside its generator gets a serial fetch for free — provided this loop keeps advancing the
    source on the CALLING thread, at both the priming loop and the refill point. The blog and
    Substack footprint ingesters depend on exactly that: their per-post fetch sits in the generator
    so the scraped host's request rate is unchanged while the paid LLM stages fan out.

    Move this advance onto a worker thread and that protection vanishes with no other symptom.
    Parallel scraping invites a Cloudflare soft-ban, which arrives as a challenge PAGE rather than a
    clean 429 — so the AIMD gate cannot react to it and nothing fails until a host has already
    blocked us. Asserted HERE, where the guarantee lives, because a refactor of this function is
    exactly when it would break, and its four callers' own tests are the wrong place to find out."""
    main = threading.current_thread().name
    advanced_on: set[str] = set()
    worked_on: set[str] = set()

    def source():
        for i in range(20):
            advanced_on.add(threading.current_thread().name)
            yield i

    def work(i):
        worked_on.add(threading.current_thread().name)
        time.sleep(0.01)                                 # overlap, so the pool really is fanning out
        return i

    run_concurrent(source(), work, lambda r: None, workers=4)

    assert advanced_on == {main}, f"source advanced off the calling thread: {advanced_on}"
    # Negative control: without this, the assertion above would also pass if the pool had silently
    # collapsed to serial execution — in which case it proves nothing about thread placement.
    assert len(worked_on) > 1, "work never left the calling thread; the assertion above is vacuous"


def test_source_error_drains_produced_prefix_not_aborts():
    """A transient SOURCE error mid-iteration (e.g. an X scrape DependencyError on page 2) must stop
    intake but still consume everything already fetched — not discard the in-flight window."""
    def flaky_source():
        for i in range(30):
            yield i
        raise RuntimeError("scrape DependencyError on the next page")   # mid-pagination transient

    got: list[int] = []
    run_concurrent(flaky_source(), lambda i: i, got.append, workers=4)
    assert sorted(got) == list(range(30))            # the fetched prefix survived; run didn't abort


def test_producer_error_skips_only_that_item():
    def work(i):
        if i == 42:
            raise ValueError("bad producer")
        return i
    got: list[int] = []
    run_concurrent(range(100), work, got.append, workers=8)
    assert 42 not in got and len(got) == 99              # one skipped, the run finished


def test_consumer_error_skips_only_that_result():
    def consume(i):
        if i == 7:
            raise ValueError("bad consumer")
        got.append(i)
    got: list[int] = []
    run_concurrent(range(50), lambda i: i, consume, workers=8)
    assert 7 not in got and len(got) == 49


def test_submission_window_stays_bounded():
    """Outstanding results (produced but not yet consumed) never exceed `inflight`, no matter how
    many items — this is the RAM bound (a rendered atom can be ~500 KB, so an unbounded backlog is a
    real hazard). A deliberately slow consumer maximizes the backlog pressure."""
    import time
    lock = threading.Lock()
    live = {"n": 0, "max": 0}

    def work(i):
        with lock:
            live["n"] += 1                               # result now outstanding
            live["max"] = max(live["max"], live["n"])
        return i

    def consume(i):
        time.sleep(0.001)                                # slow consumer → backlog wants to grow
        with lock:
            live["n"] -= 1

    run_concurrent(range(500), work, consume, workers=4, inflight=8)
    # Bound is inflight + 1: we top up BEFORE consuming the popped result, so one extra is briefly
    # outstanding. The point stands — it's bounded by the WINDOW, not by len(items)=500.
    assert live["max"] <= 9


# ── sync_bookmarks under real threads ────────────────────────────────────────

class _ManyConvo:
    """A thread-safe fake _ConvoFetcher: returns a 2-tweet chain for EVEN tids (→ a thread), [] for
    odd, and mirrors the resolved-ledger add. Its counters are locked (many producer threads)."""
    backend = "twitterapi"

    def __init__(self, profile, checked):
        self.checked = checked
        self.n_calls = self.n_failed = self.n_chains = 0
        self._lock = threading.Lock()

    def chain(self, tid):
        with self._lock:
            self.n_calls += 1
        self.checked.add(str(tid))
        if int(tid) % 2 == 0:
            with self._lock:
                self.n_chains += 1
            return [{"id": str(tid), "author": {"userName": "a"}},
                    {"id": f"{tid}9", "author": {"userName": "a"}}]
        return []


def test_sync_bookmarks_concurrent_equivalence_and_single_writer(kb_home, fake_embedder, monkeypatch):
    from pipeline.kb import derive, schema
    from pipeline.kb import ingest_x
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.vision as vision

    # 60 bookmarks: even tids carry replies (→ fetched → thread), odd are provably standalone (skip
    # the fetch but still become atoms + mark the ledger).
    norms = [{"id": str(i),
              "isReply": False,
              "replyCount": 5 if i % 2 == 0 else 0,
              "text": f"body {i}",
              "url": f"https://x.com/u/{i}",
              "entities": {"urls": []},
              "extendedEntities": {"media": []}}
             for i in range(1, 61)]

    monkeypatch.setattr(ingest_x, "_INGEST_WORKERS", 8)   # force real concurrency
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")

    producer_threads: set = set()

    def _fake_enrich(norm, cache, *, describe_all):
        producer_threads.add(threading.get_ident())       # prove producers ran on MANY threads
        return 1 if int(norm["id"]) % 10 == 0 else 0
    monkeypatch.setattr(vision, "enrich_tweet_media", _fake_enrich)

    def _fake_derive(norm):
        tid = norm["id"]
        return {"who_id": f"x:user:{tid}", "who_name": "U", "who_handle": f"u{tid}", "who_site": None,
                "when_ts": "2024-01-01T00:00:00Z", "when_precision": "second",
                "source_tags": [], "about_entities": [], "description": "d"}
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    monkeypatch.setattr(ingest_x, "_ConvoFetcher", _ManyConvo)

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=True)

    # Concurrency was REAL (not accidentally serialized on one thread).
    assert len(producer_threads) > 1
    # Equivalence: every bookmark became a durable atom, correct thread/standalone split, no dup/loss.
    assert summary["added"] == 60
    assert summary["threads"] == 30                        # the 30 even tids returned a chain
    assert summary["funnel"]["thread"]["skipped_standalone"] == 30
    assert summary["funnel"]["vlm"]["images_new"] == 6     # tids 10,20,…,60, tallied on the consumer
    # The DB agrees — and since a producer touching `conn` would have RAISED (cross-thread) and
    # dropped its atom, a full, dup-free count IS the single-writer guarantee.
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE source_type='x'").fetchone()[0] == 60
    assert conn.execute("SELECT COUNT(DISTINCT atom_id) FROM chunks").fetchone()[0] == 60
    conn.close()


def test_image_cache_save_survives_concurrent_writers(kb_home):
    """A producer adding a NEW key via cache_put while save_image_cache dumps the dict must not raise
    'dictionary changed size during iteration' — the exact crash the _CACHE_LOCK guards against."""
    import time
    from pipeline.image_cache import cache_put, save_image_cache

    cache: dict = {}
    stop = {"v": False}
    errors: list = []

    def writer():
        i = 0
        while not stop["v"]:
            cache_put(cache, f"u{i}", "x" * 40)           # keep adding NEW keys → dict RESIZES
            i += 1
            time.sleep(0.0003)                            # throttle: dict stays small, saves stay fast

    w = threading.Thread(target=writer)
    w.start()
    try:
        for _ in range(40):
            save_image_cache(kb_home, cache)              # snapshots under the lock, dumps outside it
    except Exception as e:                                 # pragma: no cover - the bug would land here
        errors.append(e)
    finally:
        stop["v"] = True
        w.join()
    assert not errors
