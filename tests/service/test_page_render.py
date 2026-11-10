"""`service/pages/render.py` — a page of sources drawn from stored snapshots.

The rules worth pinning are the quiet ones: what a scraped body may not carry onto the page (a
script, a handler, a style), the link cards a snapshot repeats, the site name a blog's title
trails, which headings count as a source's sections, and the promise the served policy (`CSP`)
makes about the page's own scripts. Plain values in, HTML out.
"""
from __future__ import annotations

import base64
import hashlib
import re
from datetime import date

from service.pages import cards, render

MADE, DAYS = date(2026, 10, 13), 90


def draw(*items: dict, title: str = "Sources", answer: str | None = None) -> str:
    return render.page(list(items), title=title, note=None, answer=answer, made=MADE, days=DAYS)


def post(text: str, **extra) -> dict:
    return {"source_type": "x", "raw": f"---\nauthor: alice\n---\n{text}", "payload": {},
            **extra}


def blog(raw: str, url: str = "https://merit.systems/blog/agents") -> dict:
    return {"source_type": "blog", "source_url": url, "raw": raw, "payload": {}}


# ── what a scraped body may not bring ─────────────────────────────────────────────

def test_a_body_loses_its_scripts_handlers_styles_and_frames():
    out = draw(blog('# Post\n\n<script>alert(1)</script>\n\n'
                    '<p style="color:red" onclick="steal()">kept text</p>\n\n'
                    '<img src="https://img.test/a.png" onerror="alert(2)">\n\n'
                    '<a href="javascript:alert(3)">a link</a>\n\n'
                    '<iframe src="https://evil.test/"></iframe>'))
    assert "kept text" in out and "https://img.test/a.png" in out
    for gone in ("alert(1)", "steal()", "onerror", "javascript:", "color:red",
                 "https://evil.test/"):
        assert gone not in out, gone
    assert out.count("<script>") == 3          # the page's own three, nothing a source wrote


def test_the_answer_loses_what_a_body_loses():
    """A host writes the answer, and a host can be fed a prompt by what it read."""
    out = draw(post("hi"), answer='Kept.\n\n<script>alert(1)</script>\n\n'
                                  '<a href="javascript:alert(3)">x</a>')
    assert "Kept." in out and "alert(" not in out and "javascript:" not in out
    assert out.count("<script>") == 3


def test_a_source_type_cannot_break_out_of_its_attribute():
    """Items come from any registered install, so even the kind is escaped where it lands."""
    out = draw(post("hi", source_type='x" onmouseover="steal()'))
    assert 'class="item item-x&quot; onmouseover=&quot;steal()"' in out


def test_every_script_on_a_page_is_one_the_policy_names():
    for html in (draw(post("hi")), render.gone(DAYS)):
        for script in re.findall(r"<script>(.*?)</script>", html, re.S):
            digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
            assert f"'sha256-{digest}'" in render.CSP
    assert "default-src 'none'" in render.CSP and "frame-src 'self'" in render.CSP
    # the comments script calls the page's own address, and the policy lets it call nowhere else
    assert "connect-src 'self'" in render.CSP


def test_only_a_live_page_carries_the_comments_script():
    assert "/comments" in draw(post("hi")) and "/comments" not in render.gone(DAYS)


# ── link cards ────────────────────────────────────────────────────────────────────

def test_a_quote_of_link_cards_becomes_one_card_each_and_a_repeat_is_dropped():
    out = draw(post("look\n\n"
                    "> **[Agents at work](https://a.test/post)**\n"
                    "> what the post says\n"
                    "> **[Agents at work](https://a.test/post/)**\n"
                    "> what the post says\n"
                    "> **[Second](https://b.test/)**\n"
                    "> another page"))
    cards = re.findall(r'<a class="linkcard" href="([^"]+)"', out)
    assert cards == ["https://a.test/post", "https://b.test/"]
    assert "<blockquote>" not in out


# ── a blog's title ────────────────────────────────────────────────────────────────

def test_a_title_loses_a_trailing_site_name_and_keeps_any_other_tail():
    out = draw(blog("# Paying agents | Merit Systems\n\ntext"),
               blog("# Rollups — a primer\n\ntext", url="https://ex.test/r"))
    assert '<h2 class="title">Paying agents</h2>' in out
    assert '<h2 class="title">Rollups — a primer</h2>' in out


# ── sections ──────────────────────────────────────────────────────────────────────

def test_sections_are_the_highest_level_holding_two_headings():
    body, heads = cards._anchor("<h1>Lone title</h1><h3>Install</h3><p>x</p><h3>Use</h3>", "s2")
    assert heads == [("s2-h1", "Install"), ("s2-h2", "Use")]
    assert '<h1>Lone title</h1>' in body and '<h3 id="s2-h2">Use</h3>' in body


def test_a_lone_heading_is_no_section_and_eight_is_the_most():
    assert cards._anchor("<h2>Subscribe</h2><p>x</p>", "s1")[1] == []
    many = "".join(f"<h2>Part {i}</h2>" for i in range(1, 12))
    assert [text for _, text in cards._anchor(many, "s1")[1]] == [f"Part {i}"
                                                                   for i in range(1, 9)]


# ── a long body ───────────────────────────────────────────────────────────────────

def test_a_long_body_shows_whole_with_nothing_to_unfold():
    """Every source opens in full: no "Continue reading"."""
    out = draw(post("word " * 5000 + "lastword"))
    assert "lastword" in out
    assert "Continue reading" not in out and "clamped" not in out


# ── a paper's PDF ─────────────────────────────────────────────────────────────────

def test_a_framed_pdf_is_labelled_and_opened_where_it_was_published():
    paper = {"source_type": "paper", "raw": "# A paper\n\n## Abstract\n\nWe study.",
             "payload": {}, "pdf_url": "/p/abc/pdf/1",
             "pdf_source": "https://arxiv.org/pdf/2401.00001"}
    out = draw(paper)
    assert '<iframe src="/p/abc/pdf/1#view=FitH"' in out
    assert "arxiv.org/pdf/2401.00001</span>" in out
    assert '<a href="https://arxiv.org/pdf/2401.00001" target="_blank"' in out


def test_a_paper_with_no_pdf_shows_its_abstract_and_says_so():
    out = draw({"source_type": "paper", "raw": "# A paper\n\n## Abstract\n\nWe study.",
                "payload": {}})
    assert "We study." in out and "no open PDF" in out and "<iframe" not in out


def test_the_expired_page_is_opyts_and_says_so():
    out = render.gone(DAYS)
    assert "This page has expired" in out and "useopyt.com" in out
    assert "deleted 90 days after it was last opened" in out
    assert '<meta name="robots" content="noindex, nofollow">' in out


# ── the answer ────────────────────────────────────────────────────────────────────

def test_the_answer_comes_before_the_sources_and_its_citations_jump_to_them():
    out = draw(blog("# Agents\n\nbody", url="https://merit.systems/blog/agents"),
               answer="## What they say\n\nAgents pay per call ([Merit]"
                      "(https://merit.systems/blog/agents/)), unlike [Stripe](https://stripe.com).")
    assert out.index('class="answer"') < out.index('class="item ')
    assert '<a href="#s1">Merit</a>' in out
    assert '<a href="https://stripe.com" target="_blank" rel="noopener noreferrer">Stripe</a>' \
        in out


def test_the_answers_text_is_item_0_so_comments_can_find_it():
    out = draw(post("hi"), answer="Agents pay per call.")
    assert '<section class="answer" id="s0"' in out
    assert '<div class="body"><div class="prose"><p>Agents pay per call.</p>' in out


def test_a_page_with_no_answer_has_no_answer_card():
    assert 'class="answer"' not in draw(post("hi"))
