"""pipeline/kb/vision.py — image→text enrichment: gating (bookmarks all / oracle thin-text),
URL caching, and the fail-safe (None = skip, don't cache).

Every path reads through `chart_read.read_image`. These fake the `ocr` fixture
(tests/kb/conftest.py), and the assertions are about a MediaRead — text plus a kind/substance
verdict — rather than a prose gloss.

That verdict is why the cache is not format-agnostic: `substance` sets an Oracle post's
`media_substance`, and inferring it from `len(prose) > 0` would score every decorative photo as
substantive.
"""
from __future__ import annotations

from pipeline.kb import vision
from pipeline.chart_read import MediaRead


def _norm(text="", n_photos=1):
    media = [{"type": "photo", "media_url_https": f"https://pbs/{i}.jpg"} for i in range(n_photos)]
    return {"id": "1", "text": text, "extendedEntities": {"media": media}}


def test_bookmarks_read_all_photos(ocr):
    norm = _norm(text="a long substantive caption well over the thin-text threshold indeed", n_photos=2)
    cache = {}
    made = vision.enrich_tweet_media(norm, cache)
    assert made == 2
    descs = [m.get("description") for m in norm["extendedEntities"]["media"]]
    assert descs == ["desc(https://pbs/0.jpg)", "desc(https://pbs/1.jpg)"]
    assert len(cache) == 2


def test_bookmarks_now_carry_the_substance_verdict(ocr):
    """A bookmark's photo records what it is, not just prose about it: a kind/substance verdict,
    not 2-4 sentences for every image alike."""
    ocr.respond(lambda url, context: MediaRead("", "not_chart"))
    norm = _norm(text="", n_photos=1)
    vision.enrich_tweet_media(norm, {})
    m = norm["extendedEntities"]["media"][0]
    assert m["media_read"] == {"kind": "not_chart", "substance": False}
    assert "description" not in m, "a decorative photo contributes no atom text"


def test_cache_hit_does_not_recall_the_model(ocr):
    cache = {"https://pbs/0.jpg": {"text": "cached-text", "kind": "chart"}}
    norm = _norm(text="", n_photos=1)
    made = vision.enrich_tweet_media(norm, cache)
    assert made == 0 and ocr.calls == []                     # served from cache, no paid call
    assert norm["extendedEntities"]["media"][0]["description"] == "cached-text"


def test_legacy_bare_string_entry_is_re_read_not_guessed(ocr):
    """`image_descriptions.json` is one key space (bare URL) shared by five ingesters, so it must
    hold one value shape. A bare-string entry would be read as `substance = len(s) > 0`, scoring a
    decorative selfie that the bookmark path glossed as "A man at a whiteboard" as substance=True
    when the footprint path read it, and `_keep_group` would keep a post a real cascade read
    would have dropped. A legacy entry must therefore be a miss: re-read, and overwrite in place."""
    ocr.respond(lambda url, context: MediaRead("", "not_chart"))
    cache = {"https://pbs/0.jpg": "A man standing in front of a whiteboard."}
    norm = _norm(text="", n_photos=1)

    made = vision.enrich_tweet_media(norm, cache)

    assert ocr.calls == ["https://pbs/0.jpg"], "the untrustable entry must be re-read, not trusted"
    assert made == 1
    assert norm["extendedEntities"]["media"][0]["media_read"]["substance"] is False, \
        "the model's verdict must win over the inferred one"
    assert isinstance(cache["https://pbs/0.jpg"], dict), "the re-read must upgrade the entry"


def test_failure_is_not_cached_and_leaves_url(ocr):
    ocr.respond(lambda url, context: None)                   # read failed
    norm = _norm(text="", n_photos=1)
    cache = {}
    made = vision.enrich_tweet_media(norm, cache)
    assert made == 0 and cache == {}                         # poison-value rule
    assert "description" not in norm["extendedEntities"]["media"][0]


def test_empty_read_is_cached_but_not_attached(ocr):
    """A photo verdict is a result, not a failure — cache it so it is never re-paid."""
    ocr.respond(lambda url, context: MediaRead("", "not_chart"))
    norm = _norm(text="", n_photos=1)
    cache = {}
    vision.enrich_tweet_media(norm, cache)
    assert cache == {"https://pbs/0.jpg": {"text": "", "kind": "not_chart"}}
    assert "description" not in norm["extendedEntities"]["media"][0]


def test_no_photos_is_a_noop(ocr):
    norm = {"id": "1", "text": "hi", "extendedEntities": {"media": [{"type": "video"}]}}
    assert vision.enrich_tweet_media(norm, {}) == 0
    assert ocr.calls == []


# ── Quoted-tweet media: the quoted node's own images earn a read too ──────────────

def _with_quote(root_text, quote_text, quote_photos=1):
    root = _norm(text=root_text, n_photos=0)
    root["extendedEntities"]["media"] = []          # root itself has no photos
    root["quoted_tweet"] = _norm(text=quote_text, n_photos=quote_photos)
    root["quoted_tweet"]["media"] = root["quoted_tweet"]["extendedEntities"]["media"]
    return root


def test_quoted_photos_read_even_when_root_has_none(ocr):
    # A text-only quoter quoting a chart must still read the chart —
    # the per-node "no photos → return 0" must not skip the quoted pass.
    ocr.respond(lambda url, context: MediaRead("chart of CPI", "chart"))
    root = _with_quote("my hot take, no image", "", quote_photos=1)
    made = vision.enrich_tweet_media(root, {})
    assert made == 1
    assert root["quoted_tweet"]["extendedEntities"]["media"][0]["description"] == "chart of CPI"


def test_root_and_quoted_share_the_cache(ocr):
    root = _norm(text="", n_photos=1)                          # root has its own photo (pbs/0.jpg)
    root["quoted_tweet"] = _norm(text="", n_photos=1)          # quoted also has pbs/0.jpg → same URL
    cache = {}
    made = vision.enrich_tweet_media(root, cache)
    assert made == 1 and len(cache) == 1                       # same CDN URL → read once
    assert ocr.calls == ["https://pbs/0.jpg"]


def test_no_quoted_tweet_is_still_fine(ocr):
    norm = _norm(text="", n_photos=1)                          # no quoted_tweet key at all
    assert vision.enrich_tweet_media(norm, {}) == 1


# ── the Oracle photo reader: images, not groups, are the unit of parallel work ───────

def test_the_reader_dispatches_one_future_per_image(ocr):
    """The property per-image dispatch buys. Fanning out across groups reads a group's images
    serially inside one thread, so work units are wildly unequal — measured: ~64% of
    groups had no images while the largest took 66s, and the stage ran at 30% pool utilization
    (1190.9 thread-seconds over a 200s wall on 20 threads). No scheduler beats its largest
    indivisible unit. Here every unit is exactly one image."""
    import threading

    peak = {"n": 0, "cur": 0}
    lock = threading.Lock()
    release = threading.Event()

    def _read(url, context):
        with lock:
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
        release.wait(timeout=2.0)
        with lock:
            peak["cur"] -= 1
        return MediaRead(f"t<{url}>", "chart")

    ocr.respond(_read)
    # One group holding six images — the shape that would serialize into one thread.
    group = [_norm(text="a thread", n_photos=6)]
    t = threading.Timer(0.3, release.set)
    t.start()
    out = vision.read_group_photos([group], {}, workers=20)
    t.cancel()

    assert out["read"] == 6
    assert peak["n"] > 1, f"one group's images still read serially (peak in-flight={peak['n']})"


def test_a_render_after_the_reader_is_free(ocr):
    """`_read_node_photos` checks the cache first, so a warm cache turns its serial
    `for m in photos` loop into dict lookups."""
    group = [_norm(text="", n_photos=3)]
    cache = {}
    vision.read_group_photos([group], cache, workers=20)
    assert len(ocr.calls) == 3

    made = vision.enrich_tweet_media(group[0], cache)
    assert made == 0, "the render pass must not pay for a single read"
    assert len(ocr.calls) == 3, "no NEW model calls during render"
    assert group[0]["extendedEntities"]["media"][0]["description"].startswith("desc(")


def test_the_reader_dedupes_before_dispatch(ocr):
    """The dedup a serial pass gets for free (an earlier read filled the cache before a later
    one looked). Fired concurrently, both occurrences miss at once and pay twice."""
    url = "https://pbs/0.jpg"
    a = _norm(text="", n_photos=1)                       # both groups reference pbs/0.jpg
    b = _norm(text="", n_photos=1)
    out = vision.read_group_photos([[a], [b]], {}, workers=20)
    assert ocr.calls == [url], f"same image read {len(ocr.calls)} times, must be 1"
    assert out["images"] == 2 and out["dispatched"] == 1


def test_the_reader_walks_quoted_nodes_too(ocr):
    """A reader that walked fewer levels than the render would leave the quoted photo owed
    forever: the render attaches it, and nothing reads it."""
    root = _with_quote("text-only quoter", "", quote_photos=2)
    out = vision.read_group_photos([[root]], {}, workers=20)
    assert out["read"] == 2, "the quoted node's images must be read too"


def test_the_reader_skips_usable_cache_but_re_reads_a_legacy_entry(ocr):
    cache = {"https://pbs/0.jpg": {"text": "x", "kind": "chart"},
             "https://pbs/1.jpg": "an old describe_image gloss"}
    out = vision.read_group_photos([[_norm(text="", n_photos=2)]], cache, workers=20)
    assert ocr.calls == ["https://pbs/1.jpg"], "only the untrustable entry is re-read"
    assert out["dispatched"] == 1


def test_a_failed_read_is_not_cached_so_the_photo_stays_owed(ocr):
    """Poison-value rule: a failed read leaves its photo pending, for the photo source's retry."""
    ocr.respond(lambda url, context: None)
    cache = {}
    out = vision.read_group_photos([[_norm(text="", n_photos=2)]], cache, workers=20)
    assert out["failed"] == 2 and out["read"] == 0
    assert cache == {}, "a failure must never be cached"


# ── _photos_pending — the third term of ingest_x's skip gate ──────
#
# Deferring VLM to Enrichment makes `atom in seen and tid in convo_checked` true forever after the
# first Enrichment run (`_ConvoFetcher.chain` marks the tid on ANY successful read, an empty chain
# included), so without a third term the photo is never described. These pin that the third term
# asks the SAME question `_read_node_photos` asks — a pending set that disagrees with what the
# renderer actually reads is worse than no pending set at all.

def test_photos_pending_is_false_once_every_photo_is_described(ocr):
    norm = _norm(text="", n_photos=2)
    cache: dict = {}
    assert vision._photos_pending(norm, cache) is True
    vision.enrich_tweet_media(norm, cache)
    assert vision._photos_pending(norm, cache) is False


def test_photos_pending_counts_a_legacy_bare_string_as_a_miss(ocr):
    """Presence is not a hit. `chart_read.from_cache` returns None for a legacy bare-string entry
    and every reader in this file treats that as absent — so a gate that tested `url in cache`
    would skip a photo the renderer is about to re-read inline."""
    norm = _norm(text="", n_photos=1)
    assert vision._photos_pending(norm, {"https://pbs/0.jpg": "an old describe_image gloss"}) is True
    assert vision._photos_pending(
        norm, {"https://pbs/0.jpg": {"text": "x", "kind": "chart"}}) is False


def test_photos_pending_sees_a_quoted_nodes_photo(ocr):
    """Two levels, the same two the renderer walks. One level would leave a quoted chart to a
    silent inline read inside a producer thread."""
    root = _with_quote("text-only quoter", "", quote_photos=1)
    assert vision._photos_pending(root, {}) is True


def test_photos_pending_uses_the_same_url_fallback_as_the_reader(ocr):
    """`media_url_https or url` — not a single key. A pending check reading only `media_url_https`
    would call a `url`-only photo pending forever and never let the bookmark settle."""
    norm = {"id": "1", "text": "", "extendedEntities": {
        "media": [{"type": "photo", "url": "https://pbs/alt.jpg"}]}}
    cache: dict = {}
    assert vision._photos_pending(norm, cache) is True
    vision.enrich_tweet_media(norm, cache)
    assert list(cache) == ["https://pbs/alt.jpg"]
    assert vision._photos_pending(norm, cache) is False


def test_a_failed_read_leaves_the_photo_pending(ocr):
    """The poison-value rule seen from the gate's side: a failure is never cached, so the bookmark
    comes back next Enrichment run. Self-healing, at the cost of retrying a dead image URL."""
    ocr.respond(lambda url, context: None)
    norm = _norm(text="", n_photos=1)
    cache: dict = {}
    vision.enrich_tweet_media(norm, cache)
    assert cache == {}
    assert vision._photos_pending(norm, cache) is True
