"""
tests/kb/test_fetch_classify.py — the shared three-verdict fetch classifier.

`classify_fetch` is the one place blog and Substack agree on what a fetch RESULT means. The
contract it encodes: a fetch has three outcomes, not two. Folding "we were blocked" into "there
is nothing here" is what made a throttled run indistinguishable from a thin archive, and it is
the signal a per-source concurrency gate has to read before it can back off on anything.

The interesting cases are the BOUNDARIES — a long page that merely discusses CAPTCHAs must not
read as a block, and a short challenge shell must not read as an empty page.
"""
from pipeline.kb.ingest_common import (CF_MITIGATED_HEADER, CHALLENGE_MAX_CHARS, FETCH_ABSENT,
                                       FETCH_OK, FETCH_UNDETERMINED, challenge_in_headers,
                                       classify_fetch)

_REAL = ("Autonomous agents compose small tools into larger systems, and the interesting "
         "failures are at the seams between them. " * 8)          # comfortably > 600 chars
_SHELL = "Just a moment... Please enable JavaScript to continue."  # short + carries a marker


# ── the three verdicts ────────────────────────────────────────────────────────────

def test_real_body_is_ok():
    assert classify_fetch(_REAL) == FETCH_OK


def test_empty_and_stub_bodies_are_absent():
    """No content and a too-short body are CONFIRMED empty — a link post, a podcast episode,
    a redirect stub. These are legitimately skippable and say nothing about the host."""
    assert classify_fetch(None) == FETCH_ABSENT
    assert classify_fetch("") == FETCH_ABSENT
    assert classify_fetch("   ") == FETCH_ABSENT
    assert classify_fetch("Coming soon.") == FETCH_ABSENT


def test_challenge_shell_is_undetermined_not_absent():
    """The case the old code got wrong: a Cloudflare shell is SHORT, so a length-only test
    files it as 'empty page' when it actually means 'we were stopped'."""
    assert classify_fetch(_SHELL) == FETCH_UNDETERMINED


# ── the boundaries (why the length gate exists at all) ────────────────────────────

def test_long_essay_mentioning_captchas_is_not_a_block():
    """A real post ABOUT bot-detection contains every marker word. Gating the marker test on a
    short body is what keeps the detector from eating an author's actual writing."""
    essay = _REAL + " This post is about CAPTCHA design and why 'verify you are human' fails."
    assert classify_fetch(essay) == FETCH_OK


def test_marker_in_title_counts_for_a_short_body():
    """Some shells put the giveaway in the <title> and leave the body nearly bare."""
    assert classify_fetch("Please wait.", title="Attention Required! | Cloudflare") == FETCH_UNDETERMINED


# ── the explicit header beats every heuristic ─────────────────────────────────────

def test_cf_mitigated_header_wins_over_a_long_body():
    """The body test is a heuristic a reworded challenge page defeats; the header is not. A
    challenge served with a LONG decoy body would otherwise be stored as a real article."""
    assert classify_fetch(_REAL, headers={CF_MITIGATED_HEADER: "challenge"}) == FETCH_UNDETERMINED


def test_header_lookup_is_case_insensitive_and_none_safe():
    assert challenge_in_headers({"CF-Mitigated": "challenge"}) is True
    assert challenge_in_headers({"cf-mitigated": ""}) is False      # present but empty ≠ challenge
    assert challenge_in_headers({"server": "cloudflare"}) is False  # fronted ≠ challenged
    assert challenge_in_headers(None) is False
    assert challenge_in_headers({}) is False


def test_unreadable_headers_are_not_evidence_of_a_block():
    """Fail-safe: a header mapping we can't parse must not manufacture a block signal that
    would throttle a healthy source."""
    class Hostile:
        def items(self):
            raise RuntimeError("nope")
    assert challenge_in_headers(Hostile()) is False


# ── the blog seam now CARRIES headers, so the explicit marker is reachable there ──

def test_blog_classify_uses_the_response_headers():
    """`_fetch_article` used to discard `resp.headers`, so on the path most likely to meet a WAF
    the only machine-readable challenge signal was unavailable and the body heuristic carried it
    alone. A challenge page defeats the body test by being LONG — a real Substack block page
    measured 5987 chars against the 600-char gate — so a long decoy body would have been STORED
    AS AN ARTICLE. Fails if the headers stop being threaded through."""
    from pipeline.kb.ingest_blog import _classify_article
    long_decoy = "Checking your browser before accessing. " * 200      # ~8000 chars
    assert len(long_decoy) > CHALLENGE_MAX_CHARS * 10

    blocked = {"content": long_decoy, "title": "Just a moment...",
               "headers": {"cf-mitigated": "challenge"}}
    assert _classify_article(blocked) == FETCH_UNDETERMINED

    # Same long body, no marker → a real article about browser checks, not a block.
    assert _classify_article({"content": long_decoy, "title": "On bot detection",
                              "headers": {"server": "cloudflare"}}) == FETCH_OK


def test_blog_classify_without_headers_keeps_the_old_behaviour():
    """A caller that supplies no headers (test double, non-CF WAF) must degrade to the body
    heuristic rather than change verdicts — the pre-existing contract."""
    from pipeline.kb.ingest_blog import _classify_article
    assert _classify_article({"content": _REAL, "title": "A real post"}) == FETCH_OK
    assert _classify_article(None) == FETCH_UNDETERMINED
