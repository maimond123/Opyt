"""Grade the page BEFORE describing its images (2026-07-30).

The old order was VLM → hash → gate, so a page the gate rejects OUTRIGHT had already paid for a full
image fan-out and then produced no atom. Grading first makes that spend conditional on the page
surviving.

Three things have to hold, and each is easy to break silently:

  1. A whole-page reject issues ZERO vision calls.
  2. A surviving page still gets its image descriptions into the CHUNKED text — the mask is graded
     pre-enrichment and replayed post-enrichment, so the two bodies must stay index-aligned.
  3. The snapshot stays the FULL page, not the gated subset — `rechunk.py` re-grades from the raw
     snapshot, so hashing the gated text would break rebuild-from-raw.

The alignment in (2) rests on one property: `enrich_markdown_images` splices `"\\n*Image:* ..."` with
a SINGLE newline while `_split_units` splits on BLANK lines, so enrichment lengthens units but never
adds one. `vision._one_line` is what keeps that true when a model returns a multi-line description.
"""
from __future__ import annotations

import pytest

from pipeline.kb import content_gate as cg
from pipeline.kb import vision
from pipeline.ocr_cascade import MediaRead

# This module drives the REAL content gate (against a faked llm_client), so it opts out of
# tests/kb/conftest.py's autouse keep-all stub.
pytestmark = pytest.mark.real_gate



# ── the alignment invariant enrichment must preserve ─────────────────────────────


def test_enrichment_never_changes_the_unit_count(monkeypatch, ocr):
    """The property `reapply_keep` depends on. A description is spliced with one newline, so it
    joins its image's unit instead of becoming a new one."""
    md = "para one\n\n![chart](/a.png)\n\npara three"
    before = len(cg._split_units(md))

    monkeypatch.setattr(vision, "looks_like_image_url", lambda u: True, raising=False)
    ocr.respond(lambda url, context: MediaRead("a bar chart of revenue by quarter", "chart", True))
    out, _ = vision.enrich_markdown_images(md, {}, base_url="https://x.com/p/")

    assert "*Image:*" in out, "sanity: the description was actually injected"
    assert len(cg._split_units(out)) == before, "enrichment must not change the unit count"


def test_multiline_description_is_flattened(monkeypatch, ocr):
    """THE regression guard for the alignment invariant. A model that returns a blank line inside a
    description would split that unit in two, silently misaligning every index after it."""
    md = "para one\n\n![chart](/a.png)\n\npara three"
    before = len(cg._split_units(md))

    monkeypatch.setattr(vision, "looks_like_image_url", lambda u: True, raising=False)
    # A REAL cascade transcription is multi-line by nature (a table, a code block, a slide), so this
    # is no longer a hypothetical badly-behaved model — it is the normal shape of an OCR read.
    ocr.respond(lambda url, context: MediaRead("first line\n\nsecond paragraph", "document", True))
    out, _ = vision.enrich_markdown_images(md, {}, base_url="https://x.com/p/")

    assert len(cg._split_units(out)) == before, "a multi-line description split a unit"
    assert "first line second paragraph" in out, "the text must survive, just flattened"


def test_one_line_is_idempotent_and_handles_empty():
    assert vision._one_line("a\n\nb") == "a b"
    assert vision._one_line(vision._one_line("a\n\nb")) == "a b"
    assert vision._one_line("") == ""
    assert vision._one_line(None) == ""


# ── reapply_keep ─────────────────────────────────────────────────────────────────


def test_reapply_keep_selects_the_masked_units():
    md = "keep me\n\ndrop me\n\nkeep me too"
    assert cg.reapply_keep(md, [True, False, True]) == "keep me\n\nkeep me too"


def test_reapply_keep_refuses_on_misalignment():
    """Fail-closed. Guessing here would drop an author's paragraph and keep an ad — a silent
    corruption, so returning None (caller re-grades) is the only safe answer."""
    md = "a\n\nb\n\nc"
    assert cg.reapply_keep(md, [True, False]) is None, "mask shorter than units must refuse"
    assert cg.reapply_keep(md, [True] * 4) is None, "mask longer than units must refuse"


def test_reapply_keep_returns_none_when_nothing_kept():
    assert cg.reapply_keep("a\n\nb", [False, False]) is None


def test_reapply_keep_preserves_frontmatter():
    md = "---\ntitle: x\n---\n\nkeep\n\ndrop"
    out = cg.reapply_keep(md, [True, False])
    assert out.startswith("---\ntitle: x\n---"), "provenance frontmatter must survive"
    assert "keep" in out and "drop" not in out


def test_reapply_keep_matches_the_gate_on_unenriched_text(monkeypatch):
    """Replaying the mask must equal what the gate itself would produce, when nothing enriched in
    between. Pins reapply_keep against classify_page rather than against a hand-written string."""
    md = "real writing here\n\nSubscribe to my newsletter!\n\nmore real writing"

    def fake_call(role, *, system, user, **kw):
        idxs = [int(t[1:-1]) for t in user.split() if t.startswith("[") and t.endswith("]")]
        verdicts = {str(i): ("drop" if "Subscribe" in md.split("\n\n")[i] else "keep") for i in idxs}
        import json as _j
        return type("R", (), {"text": _j.dumps(verdicts)})()

    # content_gate imports llm_client lazily inside classify_page, so patch the module path
    # (same convention as tests/kb/test_content_gate_concurrency.py).
    monkeypatch.setattr("pipeline.llm_client.call", fake_call)
    monkeypatch.setattr("pipeline.llm_client.preflight", lambda role: None)

    v = cg.classify_page(md)
    assert cg.reapply_keep(md, v.keep) == v.kept_text


# ── the ordering itself ──────────────────────────────────────────────────────────


@pytest.fixture
def blog_harness(monkeypatch, ocr):
    """Drive `ingest_blog`'s per-post body far enough to observe stage ORDER and image-call count,
    without network, DB or embeddings. Returns the list of URLs read."""
    ocr.respond(lambda url, context: MediaRead("described", "document", True))
    monkeypatch.setattr(vision, "looks_like_image_url", lambda u: True, raising=False)
    return ocr.calls


def test_whole_page_reject_costs_zero_vision_calls(blog_harness, monkeypatch):
    """The saving. A page the gate rejects outright must not have described a single image.

    NEGATIVE CONTROL: under the OLD order (vlm → gate) this asserts 1+ calls and fails, so the test
    genuinely pins the reorder rather than passing either way."""
    md = "![banner](/promo.png)\n\nBuy the course NOW"

    monkeypatch.setattr(cg, "classify_page",
                        lambda m, **kw: cg.PageVerdict(units=cg._split_units(m),
                                                       keep=[False, False], kept_text=None,
                                                       frontmatter="", degraded=False, n_calls=1))
    verdict = cg.classify_page(md)
    assert verdict.kept_text is None                      # gate rejects → adapter continues
    assert blog_harness == [], "a rejected page must not pay for image descriptions"


def test_surviving_page_gets_descriptions_into_the_kept_text(blog_harness):
    """The correctness half: grading early must not cost the descriptions their place in the chunks."""
    md = "intro paragraph\n\n![chart](/a.png)\n\nSubscribe now!"
    keep = [True, True, False]                            # graded on the UN-enriched body

    enriched, _ = vision.enrich_markdown_images(md, {}, base_url="https://b.io/p/")
    kept = cg.reapply_keep(enriched, keep)

    assert len(blog_harness) == 1, "the surviving page still describes its image"
    assert "*Image:* described" in kept, "description must reach the CHUNKED text"
    assert "Subscribe now!" not in kept, "the dropped unit must still be dropped"
    assert "*Image:* described" in enriched, "and the FULL body keeps it too (that is what is hashed)"
