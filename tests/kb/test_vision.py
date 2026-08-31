"""pipeline/kb/vision.py — image→text enrichment: gating (bookmarks all / oracle thin-text),
URL caching, and the fail-safe (None = skip, don't cache).

MIGRATED 2026-08-02: every path here now reads through `ocr_cascade.read_image`, not
`describe_images.describe_image`. So these fake the `ocr` fixture (tests/kb/conftest.py), and the
assertions are about a MediaRead — text PLUS a kind/substance verdict — rather than a prose gloss.

That verdict is the point of the migration and it is why the cache stopped being format-agnostic:
`substance` decides artifact-vs-fragment in `_keep_group`, and the old code INFERRED it from
`len(prose) > 0`, which scored every decorative photo as substantive.
"""
from __future__ import annotations

from pipeline.kb import vision
from pipeline.ocr_cascade import MediaRead


def _norm(text="", n_photos=1):
    media = [{"type": "photo", "media_url_https": f"https://pbs/{i}.jpg"} for i in range(n_photos)]
    return {"id": "1", "text": text, "extendedEntities": {"media": media}}


def test_bookmarks_read_all_photos(ocr):
    norm = _norm(text="a long substantive caption well over the thin-text threshold indeed", n_photos=2)
    cache = {}
    made = vision.enrich_tweet_media(norm, cache, describe_all=True)
    assert made == 2
    descs = [m.get("description") for m in norm["extendedEntities"]["media"]]
    assert descs == ["desc(https://pbs/0.jpg)", "desc(https://pbs/1.jpg)"]
    assert len(cache) == 2


def test_bookmarks_now_carry_the_substance_verdict(ocr):
    """The migration's actual gain: a bookmark's photo now records WHAT it is, not just prose about
    it. `describe_image` could not produce this — it returned 2-4 sentences for every image alike."""
    ocr.respond(lambda url, context: MediaRead("", "photo", False))
    norm = _norm(text="", n_photos=1)
    vision.enrich_tweet_media(norm, {}, describe_all=True)
    m = norm["extendedEntities"]["media"][0]
    assert m["media_read"] == {"kind": "photo", "substance": False}
    assert "description" not in m, "a decorative photo contributes no atom text"


def test_oracle_gate_skips_when_text_is_substantive(ocr):
    norm = _norm(text="x" * 200, n_photos=1)   # long caption → gate fails → no read
    made = vision.enrich_tweet_media(norm, {}, describe_all=False)
    assert made == 0 and ocr.calls == []
    assert "description" not in norm["extendedEntities"]["media"][0]


def test_oracle_gate_fires_on_thin_text_and_image_only(ocr):
    assert vision.enrich_tweet_media(_norm(text="wow", n_photos=1), {}, describe_all=False) == 1
    assert vision.enrich_tweet_media(_norm(text="", n_photos=1), {}, describe_all=False) == 1


def test_cache_hit_does_not_recall_the_model(ocr):
    cache = {"https://pbs/0.jpg": {"text": "cached-text", "kind": "document", "substance": True}}
    norm = _norm(text="", n_photos=1)
    made = vision.enrich_tweet_media(norm, cache, describe_all=True)
    assert made == 0 and ocr.calls == []                     # served from cache, no paid call
    assert norm["extendedEntities"]["media"][0]["description"] == "cached-text"


def test_legacy_bare_string_entry_is_re_read_not_guessed(ocr):
    """THE regression guard for the 2026-08-02 finding. `image_descriptions.json` is one key space
    (bare URL) shared by five ingesters, and it used to hold two incompatible value shapes. The old
    `from_cache` accepted a bare string and inferred `substance = len(s) > 0` — so a decorative
    selfie that the BOOKMARK path had glossed as "A man at a whiteboard" scored substance=True when
    the FOOTPRINT path read it, and `_keep_group` kept a post a real cascade read would have
    dropped. A legacy entry must therefore be a MISS: re-read, and overwrite in place."""
    ocr.respond(lambda url, context: MediaRead("", "photo", False))
    cache = {"https://pbs/0.jpg": "A man standing in front of a whiteboard."}
    norm = _norm(text="", n_photos=1)

    made = vision.enrich_tweet_media(norm, cache, describe_all=True)

    assert ocr.calls == ["https://pbs/0.jpg"], "the untrustable entry must be re-read, not trusted"
    assert made == 1
    assert norm["extendedEntities"]["media"][0]["media_read"]["substance"] is False, \
        "the model's verdict must win over the inferred one"
    assert isinstance(cache["https://pbs/0.jpg"], dict), "the re-read must upgrade the entry"


def test_failure_is_not_cached_and_leaves_url(ocr):
    ocr.respond(lambda url, context: None)                   # read failed
    norm = _norm(text="", n_photos=1)
    cache = {}
    made = vision.enrich_tweet_media(norm, cache, describe_all=True)
    assert made == 0 and cache == {}                         # poison-value rule
    assert "description" not in norm["extendedEntities"]["media"][0]


def test_empty_read_is_cached_but_not_attached(ocr):
    """A PHOTO verdict IS a result, not a failure — cache it so it is never re-paid."""
    ocr.respond(lambda url, context: MediaRead("", "photo", False))
    norm = _norm(text="", n_photos=1)
    cache = {}
    vision.enrich_tweet_media(norm, cache, describe_all=True)
    assert cache == {"https://pbs/0.jpg": {"text": "", "kind": "photo", "substance": False}}
    assert "description" not in norm["extendedEntities"]["media"][0]


def test_no_photos_is_a_noop(ocr):
    norm = {"id": "1", "text": "hi", "extendedEntities": {"media": [{"type": "video"}]}}
    assert vision.enrich_tweet_media(norm, {}, describe_all=True) == 0
    assert ocr.calls == []


# ── Quoted-tweet media: the quoted node's own images earn a read too ──────────────

def _with_quote(root_text, quote_text, quote_photos=1):
    root = _norm(text=root_text, n_photos=0)
    root["extendedEntities"]["media"] = []          # root itself has no photos
    root["quoted_tweet"] = _norm(text=quote_text, n_photos=quote_photos)
    root["quoted_tweet"]["media"] = root["quoted_tweet"]["extendedEntities"]["media"]
    return root


def test_quoted_photos_read_even_when_root_has_none(ocr):
    # The short-circuit fix: a text-only quoter quoting a chart must still read the chart —
    # the per-node "no photos → return 0" must not skip the quoted pass.
    ocr.respond(lambda url, context: MediaRead("chart of CPI", "chart", True))
    root = _with_quote("my hot take, no image", "", quote_photos=1)
    made = vision.enrich_tweet_media(root, {}, describe_all=True)
    assert made == 1
    assert root["quoted_tweet"]["extendedEntities"]["media"][0]["description"] == "chart of CPI"


def test_root_and_quoted_share_the_cache(ocr):
    root = _norm(text="", n_photos=1)                          # root has its own photo (pbs/0.jpg)
    root["quoted_tweet"] = _norm(text="", n_photos=1)          # quoted also has pbs/0.jpg → same URL
    cache = {}
    made = vision.enrich_tweet_media(root, cache, describe_all=True)
    assert made == 1 and len(cache) == 1                       # same CDN URL → read once
    assert ocr.calls == ["https://pbs/0.jpg"]


def test_no_quoted_tweet_is_still_fine(ocr):
    norm = _norm(text="", n_photos=1)                          # no quoted_tweet key at all
    assert vision.enrich_tweet_media(norm, {}, describe_all=True) == 1


# ── the prefetch: images, not groups, are the unit of parallel work ───────────────

def test_prefetch_dispatches_one_future_per_image(ocr):
    """THE property the whole change buys. Ingest fans out across GROUPS and reads a group's images
    serially inside one thread, so work units are wildly unequal — measured 2026-08-01, ~64% of
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
        return MediaRead(f"t<{url}>", "document", True)

    ocr.respond(_read)
    # ONE group holding six images — the shape that used to serialize into one thread.
    group = [_norm(text="a thread", n_photos=6)]
    t = threading.Timer(0.3, release.set)
    t.start()
    out = vision.prefetch_group_media([group], {}, workers=20)
    t.cancel()

    assert out["read"] == 6
    assert peak["n"] > 1, f"one group's images still read serially (peak in-flight={peak['n']})"


def test_prefetch_makes_the_render_pass_free(ocr):
    """Phase 3 is UNCHANGED code: `_cascade_node_photos` already checks the cache first, so a warm
    cache turns its serial `for m in photos` loop into dict lookups. If this ever regresses, the
    ingester's `late_reads` counter is what surfaces it."""
    group = [_norm(text="", n_photos=3)]
    cache = {}
    vision.prefetch_group_media([group], cache, workers=20)
    assert len(ocr.calls) == 3

    made, reads = vision.enrich_tweet_media_cascade(group[0], cache)
    assert made == 0, "the render pass must not pay for a single read"
    assert len(ocr.calls) == 3, "no NEW model calls during render"
    assert len(reads) == 3, "but telemetry still sees every image (cache hits included)"
    assert group[0]["extendedEntities"]["media"][0]["description"].startswith("desc(")


def test_prefetch_dedupes_before_dispatch(ocr):
    """The dedup the old serial pass got for free (an earlier read filled the cache before a later
    one looked). Fired concurrently, both occurrences miss at once and pay twice."""
    url = "https://pbs/0.jpg"
    a = _norm(text="", n_photos=1)                       # both groups reference pbs/0.jpg
    b = _norm(text="", n_photos=1)
    out = vision.prefetch_group_media([[a], [b]], {}, workers=20)
    assert ocr.calls == [url], f"same image read {len(ocr.calls)} times, must be 1"
    assert out["images"] == 2 and out["dispatched"] == 1


def test_prefetch_walks_quoted_nodes_too(ocr):
    """A prefetch that walks fewer levels than the render does leaves the missed image to an inline
    read inside a producer thread — the serialization this removes, quietly back."""
    root = _with_quote("text-only quoter", "", quote_photos=2)
    out = vision.prefetch_group_media([[root]], {}, workers=20)
    assert out["read"] == 2, "the quoted node's images must be prefetched, not left to the render"


def test_prefetch_skips_usable_cache_but_re_reads_a_legacy_entry(ocr):
    cache = {"https://pbs/0.jpg": {"text": "x", "kind": "document", "substance": True},
             "https://pbs/1.jpg": "an old describe_image gloss"}
    out = vision.prefetch_group_media([[_norm(text="", n_photos=2)]], cache, workers=20)
    assert ocr.calls == ["https://pbs/1.jpg"], "only the untrustable entry is re-read"
    assert out["dispatched"] == 1


def test_prefetch_failure_is_not_cached_so_the_render_retries(ocr):
    """Poison-value rule, and the reason `late_reads` exists: a failed prefetch leaves the image to
    an inline read, which is correct but silently reintroduces serialization."""
    ocr.respond(lambda url, context: None)
    cache = {}
    out = vision.prefetch_group_media([[_norm(text="", n_photos=2)]], cache, workers=20)
    assert out["failed"] == 2 and out["read"] == 0
    assert cache == {}, "a failure must never be cached"
