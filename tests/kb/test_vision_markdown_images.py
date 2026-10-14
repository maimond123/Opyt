"""
tests/kb/test_vision_markdown_images.py — the markdown image enricher's URL handling.

Two properties, both seen against karpathy.github.io:

  • Relative refs must be described. Self-hosted blogs write `![](/assets/rnn/diags.jpeg)`, and an
    absolute-only pattern saw 2 of that blog's 104 image refs — i.e. it was blind to ~98% of them.
  • Non-image targets must be skipped. trafilatura renders some embeds with image syntax
    (`![@handle](https://twitter.com/handle)`). A vision call on an HTML page fails → returns None
    → is not cached (poison-value rule) → is retried on every run, forever. Skipping is the only
    outcome that converges.
"""
import pytest

from pipeline.kb import vision
from pipeline.chart_read import MediaRead


@pytest.fixture()
def spy(ocr):
    """Every URL handed to the image model, in dispatch order. The shared `ocr` fake already
    returns `desc({url})` with document substance, which is what these assertions expect."""
    return ocr.calls


# ── relative refs: the case that made the enricher blind on self-hosted blogs ─────

def test_relative_ref_is_resolved_against_base_url(spy):
    md = "Intro\n\n![](/assets/rnn/diags.jpeg)\n\nOutro"
    out, n = vision.enrich_markdown_images(
        md, {}, entry_mode="user-saved", base_url="https://karpathy.github.io/2015/05/21/rnn-effectiveness/")

    assert spy == ["https://karpathy.github.io/assets/rnn/diags.jpeg"], \
        "a root-relative ref must resolve against the post's ORIGIN, not be sent as-is"
    assert n == 1
    assert "*Image:* desc(" in out, "the description must be injected into the body"


def test_cache_key_is_the_absolute_url(spy):
    """Keying on the raw relative ref would collide across blogs — `/assets/logo.png` is not the
    same image on two different sites."""
    cache = {}
    vision.enrich_markdown_images("![](/img/a.png)", cache, entry_mode="user-saved", base_url="https://alice.dev/p/1/")
    vision.enrich_markdown_images("![](/img/a.png)", cache, entry_mode="user-saved", base_url="https://bob.dev/p/1/")
    assert set(cache) == {"https://alice.dev/img/a.png", "https://bob.dev/img/a.png"}
    assert len(spy) == 2, "two DIFFERENT images must not share one cache entry"


def test_absolute_ref_still_works_without_base_url(spy):
    """Substack's path: absolute CDN URLs, no base_url passed. Must be unchanged."""
    md = "![](https://substackcdn.com/image/fetch/$s_!33K5!,w_1456,c_limit)"
    _, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")
    assert n == 1 and len(spy) == 1


# ── the non-image guard: what stops a permanent per-run retry ─────────────────────

def test_non_image_target_is_never_sent_to_the_vlm(spy):
    """`![@handle](https://twitter.com/handle)` is image syntax around an HTML page."""
    md = "![@MrChrisJohnson](https://twitter.com/MrChrisJohnson)"
    out, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")
    assert spy == [], "an HTML page must never reach the vision model"
    assert n == 0
    assert out == md, "a skipped ref must be left exactly as-is"


def test_extensionless_image_cdn_is_still_described(spy):
    """The guard cannot be extension-only: Substack CDN URLs carry NO file extension, so an
    extension-only rule would silently drop every Substack image."""
    md = "![](https://substackcdn.com/image/fetch/$s_!pmWZ!,w_1456,f_auto)"
    _, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")
    assert n == 1 and len(spy) == 1


def test_mixed_body_describes_only_the_images(spy):
    md = ("![](/assets/chart.jpeg)\n"
          "![@someone](https://twitter.com/someone)\n"
          "![](https://cdn.example.com/photo.png)\n")
    _, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved", base_url="https://blog.example.com/post/")
    assert n == 2
    assert spy == ["https://blog.example.com/assets/chart.jpeg",
                   "https://cdn.example.com/photo.png"]


# ── the shared predicate both polarities depend on ───────────────────────────────

def test_looks_like_image_url_covers_both_shapes():
    from pipeline.kb.ingest_common import looks_like_image_url
    assert looks_like_image_url("https://x.dev/a/diags.jpeg") is True      # by extension
    assert looks_like_image_url("https://substackcdn.com/image/fetch/$s") is True   # by CDN shape
    assert looks_like_image_url("https://twitter.com/MrChrisJohnson") is False
    assert looks_like_image_url("") is False


# ── ARC-1: one post's images fan out instead of going one at a time ──────────────
# `re.sub` walked matches serially, so `_VLM_GATE`'s 8 permits were unreachable from the long-form
# path. These pin the fan-out AND the invariant it could most easily break.

def test_images_of_one_post_are_described_concurrently(ocr):
    import threading
    state = {"inflight": 0, "peak": 0}
    lock = threading.Lock()
    release = threading.Event()

    def _read(url, context):
        with lock:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        release.wait(timeout=2.0)          # hold calls open so overlap is observable
        with lock:
            state["inflight"] -= 1
        return MediaRead(f"desc({url})", "chart")

    ocr.respond(_read)

    md = "\n\n".join(f"![](https://cdn.example.com/{i}.png)" for i in range(5))
    t = threading.Timer(0.3, release.set); t.start()
    _, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")
    t.cancel()

    assert n == 5
    assert state["peak"] > 1, f"images still described serially (peak in-flight={state['peak']})"


def test_repeated_url_in_one_post_is_charged_once(spy):
    """A sequential pass dedupes for free (an earlier match fills the cache before a later one
    reads it). Fired concurrently, every occurrence would miss the cache at once and pay
    twice for the same image — so the work-list must dedupe before dispatch."""
    url = "https://cdn.example.com/same.png"
    md = f"![](  {url}  )".replace("  ", "") + "\n\n" + f"![]({url})" + "\n\n" + f"![]({url})"
    out, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")

    assert spy == [url], f"the same image was described {len(spy)} times, must be 1"
    assert n == 1
    assert out.count("*Image:*") == 3, "every occurrence must still render the description"


def test_document_order_is_preserved_regardless_of_completion_order(ocr):
    """Descriptions are spliced back by match position, so a fast later image must not overtake a
    slow earlier one in the rendered body."""
    import time

    def _read(url, context):
        time.sleep(0.15 if url.endswith("first.png") else 0.0)   # invert completion order
        return MediaRead(f"D-{url.rsplit('/', 1)[-1]}", "chart")

    ocr.respond(_read)

    md = "![](https://c.dev/first.png)\n\n![](https://c.dev/second.png)"
    out, _ = vision.enrich_markdown_images(md, {}, entry_mode="user-saved")
    assert out.index("D-first.png") < out.index("D-second.png")


def test_one_failing_image_does_not_sink_the_others(ocr):
    """Fail-safe under fan-out: a raising describe must skip only its own image, stay uncached
    (so it retries next run), and leave that ref untouched."""
    def _read(url, context):
        if "bad" in url:
            raise RuntimeError("vision exploded")
        return MediaRead(f"desc({url})", "chart")

    cache = {}
    ocr.respond(_read)

    md = "![](https://c.dev/bad.png)\n\n![](https://c.dev/good.png)"
    out, n = vision.enrich_markdown_images(md, cache, entry_mode="user-saved")

    assert n == 1
    assert "https://c.dev/bad.png" not in cache, "a failure must NOT be cached (poison-value rule)"
    assert out.count("*Image:*") == 1
    assert "![](https://c.dev/bad.png)\n\n" in out, "the failed ref must be left exactly as-is"


def test_cap_bounds_dispatched_calls(monkeypatch, spy):
    """`max_images` is a runaway guard; under fan-out it bounds dispatched calls (a failed call
    costs a request too), and the survivors are the first N in document order."""
    md = "\n\n".join(f"![](https://c.dev/{i}.png)" for i in range(6))
    out, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved", max_images=2)
    assert len(spy) == 2 and n == 2
    assert spy == ["https://c.dev/0.png", "https://c.dev/1.png"], "cap must keep document order"
    assert out.count("*Image:*") == 2


# ── descriptions stay on one line; base64 images never become text ───────────────

def test_a_multiline_description_is_flattened_beside_its_image(monkeypatch, ocr):
    """A real OCR read is multi-line (a table, a slide). It stays one line beside its image."""
    monkeypatch.setattr(vision, "looks_like_image_url", lambda u: True, raising=False)
    ocr.respond(lambda url, context: MediaRead("first line\n\nsecond paragraph", "chart"))
    out, _ = vision.enrich_markdown_images("para one\n\n![chart](/a.png)\n\npara three", {}, entry_mode="user-saved",
                                           base_url="https://x.com/p/")
    assert "*Image:* first line second paragraph" in out


def test_one_line_is_idempotent_and_handles_empty():
    assert vision._one_line("a\n\nb") == "a b"
    assert vision._one_line(vision._one_line("a\n\nb")) == "a b"
    assert vision._one_line("") == ""
    assert vision._one_line(None) == ""


def test_a_base64_image_is_replaced_by_its_alt_text_and_never_read(spy):
    """Inlined base64 (seen on vitalik.eth.limo and sakana.ai) must not be chunked and embedded as text."""
    md = ("Intro\n\n![a chart](data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==)\n\n"
          "![](data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEK)\n\nOutro")
    out, n = vision.enrich_markdown_images(md, {}, entry_mode="user-saved", base_url="https://b.io/p/")
    assert "base64" not in out and "data:" not in out
    assert "a chart" in out and "Outro" in out
    assert spy == [] and n == 0


# ── where a page's images are read ─────────────────────────────────────────────

@pytest.mark.parametrize("entry_mode", ["oracle-footprint", "user-saved"])
def test_a_page_someone_chose_is_read(spy, entry_mode):
    _, n = vision.enrich_markdown_images("![c](https://cdn/a.png)", {}, entry_mode=entry_mode)
    assert spy == ["https://cdn/a.png"] and n == 1


def test_a_linked_page_reads_nothing(spy):
    """An article or Substack post an Oracle linked is pulled in automatically: no read, no cache
    write, the image line kept as it was. A base64 image still becomes its alt text."""
    cache = {}
    md = "![c](https://cdn/a.png)\n\n![inline](data:image/png;base64,AAAA)"
    out, n = vision.enrich_markdown_images(md, cache, entry_mode="author_referenced")
    assert spy == [] and n == 0 and cache == {}
    assert out == "![c](https://cdn/a.png)\n\ninline"


def test_the_article_builder_passes_its_entry_mode(monkeypatch):
    """`build_article_atom` serves the Oracle blog, hopper and the linked-documents worker; the
    caller's entry_mode is what decides."""
    from pipeline.kb import ingest_blog
    seen_modes = []

    def fake(md, cache, *, entry_mode, **kw):
        seen_modes.append(entry_mode)
        return md, 0
    monkeypatch.setattr(vision, "enrich_markdown_images", fake)
    article = {"title": "T", "text": "body " * 50, "date": "2026-01-01", "url": "https://b.io/p"}
    for mode in ("author_referenced", "user-saved"):
        ingest_blog.build_article_atom(dict(article), url="https://b.io/p", atom_id="https://b.io/p",
                                       blog_url="https://b.io", author="b.io", entry_mode=mode)
    assert seen_modes == ["author_referenced", "user-saved"]
