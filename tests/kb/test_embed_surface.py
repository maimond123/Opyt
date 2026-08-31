"""embed_surface — the KEEP list from the plan, one named test per failure mode.

The DROP side is the easy half: if a pattern stops matching, the scaffolding survives into the
vector and retrieval quietly gets slightly worse. The KEEP side is where a regression is
CATASTROPHIC and silent — a strip that eats the display name costs 58% of author-query precision
(measured; see the module docstring), and a strip that eats quoted text reduces a quote-tweet atom
to an empty string. So every KEEP row in `docs/plans/2026-08-11-embed-surface.md` is a test here,
named for the thing that breaks.

These are all pure-function tests on purpose: `strip_for_embedding` does no I/O, so nothing here
needs a store, an embedder, or a network.
"""
from __future__ import annotations

from pipeline.kb.embed_surface import STRIP_VERSION, _strip_body, strip_for_embedding

# A real bookmark atom, verbatim from the corpus (frontier-stage1-home, x:1356497675009593345)
# minus the frontmatter, which `_chunk_snapshot` has already removed before this ever runs.
X_POST = """# Deep Thrill — 2021-02-02

As part of AI research for a client, we learned just how many biotech companies got FDA approval.

---
*Bookmarked · [Original post](https://x.com/DeeperThrill/status/1356497675009593345)*
"""

# x:1938360918884421835 — the atom that is ENTIRELY quoted content. A quote-stripping strip
# reduces this one to an empty string, which is why it has its own test.
X_QUOTE = """# David Li — 2025-06-26

**Quoting** [@lecong](https://x.com/lecong/status/1932993693050646963):
> *Le Cong@Stanford, AI+Bio+Gene-Editing*
>
> How do you get an LLM to reason like a CRISPR pro?
> Genome-Bench: 3,000 + curated Q&As on gene-editing fundamentals.

---
*Bookmarked · [Original post](https://x.com/hypersoren/status/1938360918884421835)*
"""

X_MEDIA = """# Peter van Sabben — 2024-03-20

> **Thread** · [2 posts](https://x.com/sabben/status/1770389801486717390)

**[1/2]** Startup Advantage in AI:

## Media

![photo](https://pbs.twimg.com/media/GJGusyxWsAAsr6z.jpg)

*Image:* AI Advantage to Startups: slow industry pace, business model conflict.

---

**[2/2]** Why now? Timing is critical: https://t.co/RpqKgLL75L

---
*Bookmarked · [Original post](https://x.com/sabben/status/1770389801486717390)*
"""


# ── KEEP: the identity tokens (the measured 58%-precision failure) ───────────────

def test_display_name_survives_the_byline_strip():
    """V2 in the probe: dropping the name took Gajesh 0.933 -> 0.000 on author queries."""
    out = strip_for_embedding(X_POST, "x")
    assert "Deep Thrill" in out
    assert "# Deep Thrill" not in out       # the markup went, the name stayed
    assert "2021-02-02" not in out          # the date is `when_ts`, not embedding material


def test_author_handle_is_recovered_from_the_footer_url():
    """The handle is a SECOND identity token and the footer URL is the only place it is written.
    A plain footer DROP would lose it — this is why the footer is a rewrite."""
    assert "@DeeperThrill" in strip_for_embedding(X_POST, "x")
    assert "@hypersoren" in strip_for_embedding(X_QUOTE, "x")


def test_quoted_persons_handle_survives():
    """1,066 occurrences, 950 distinct. Often the ONLY naming of that person in the atom —
    strip it and `@lecong` vanishes from the Genome-Bench atom entirely."""
    assert "@lecong" in strip_for_embedding(X_QUOTE, "x")


def test_quoted_display_name_survives():
    """The quoted author's own name, inside `> *…*`. Same identity argument, second person."""
    assert "Le Cong@Stanford" in strip_for_embedding(X_QUOTE, "x")


def test_github_repo_identity_survives():
    """354 distinct repo identities live in this H1 and nowhere else in the atom."""
    md = ("# AliesTaha/BareNeuralNetwork\n\n> bare metal neural network\n\n"
          "---\n*GitHub · [View repo](https://github.com/AliesTaha/BareNeuralNetwork)*\n")
    out = strip_for_embedding(md, "github")
    assert "AliesTaha/BareNeuralNetwork" in out


def test_paper_title_survives():
    md = "# Selling Information in Competitive Environments\n\n**Authors:** A. Smith\n"
    out = strip_for_embedding(md, "paper")
    assert "Selling Information in Competitive Environments" in out


# ── KEEP: the content the attribution surface would have eaten ───────────────────

def test_quoted_content_survives_and_the_atom_does_not_go_empty():
    """`get_main_body_text` removes 9,841 quoted lines corpus-wide. Quoted lines are 10.8% of the
    corpus, and for this atom they are ALL of it."""
    out = strip_for_embedding(X_QUOTE, "x")
    assert "Genome-Bench" in out and "CRISPR" in out
    assert out.strip()                                   # the whole point: not empty


def test_image_description_survives_but_the_marker_does_not():
    """`*Image:*` descriptions are 16.9% of the corpus and 72% of them are transcribed screenshots
    of text. The MARKER word is template; what follows it is the only searchable content on an
    image-borne post."""
    out = strip_for_embedding(X_MEDIA, "x")
    assert "AI Advantage to Startups" in out
    assert "business model conflict" in out
    assert "*Image:*" not in out and "Image:" not in out


def test_blog_image_alt_text_is_authored_and_survives():
    """X alt text is a media-type slot (5 distinct values corpus-wide, none over 12 chars); blog
    alt text is writing (34 distinct (source, alt) pairs over 69 markups). That asymmetry is why
    `_strip_line` branches on source instead of gating `_IMAGE` as one rule."""
    md = "Some prose.\n\n![Low- v High-Entropy Problem Spaces](https://cdn.example/a.png)\n"
    assert "Low- v High-Entropy Problem Spaces" in strip_for_embedding(md, "blog")


def test_link_text_survives_even_though_the_url_goes():
    md = "See the [DNAnexus and TMA Precision Health collaboration](https://ow.ly/NO8K50Qu87l).\n"
    out = strip_for_embedding(md, "x")
    assert "DNAnexus and TMA Precision Health collaboration" in out
    assert "ow.ly" not in out


# ── DROP: the scaffolding, per the measured inventory ────────────────────────────

def test_x_image_alt_and_cdn_url_are_dropped():
    """Re-measured 2026-08-11: 1,979 X image markups, FIVE distinct alt values, 1,979 CDN hashes."""
    out = strip_for_embedding(X_MEDIA, "x")
    assert "photo" not in out and "pbs.twimg.com" not in out


def test_template_scaffolding_is_dropped():
    out = strip_for_embedding(X_MEDIA, "x")
    assert "## Media" not in out                 # 2 distinct values corpus-wide
    assert "[1/2]" not in out                    # positional only
    assert "Thread" not in out                   # `> **Thread** · [2 posts](url)`
    assert "t.co" not in out                     # opaque shortcodes
    assert "Bookmarked" not in out               # the label IS source_type
    assert "**" not in out and "---" not in out


def test_footer_label_goes_but_the_line_is_not_blindly_deleted():
    out = strip_for_embedding(X_POST, "x")
    assert "Bookmarked" not in out and "Original post" not in out
    assert "@DeeperThrill" in out                # …because the URL was mined first


# ── Fail-safe direction: kept-too-much, never cut-content ────────────────────────

def test_scaffolding_cut_by_a_chunk_boundary_is_left_intact():
    """Every pattern is line-anchored and this runs PER CHUNK, so a footer split across a chunk
    edge fails to match and survives. Kept-too-much is the designed failure mode; a half-matched
    pattern eating the start of the next chunk's real content is not."""
    head = "some real content here\n\n---\n*Bookmarked · [Original post](https://x.com/a/stat"
    tail = "us/1)*\n\nmore real content\n"
    assert "Bookmarked" in strip_for_embedding(head, "x")      # incomplete → untouched
    assert "more real content" in strip_for_embedding(tail, "x")


def test_an_all_scaffolding_chunk_falls_back_to_its_original_text():
    """An empty string is not a free vector — it is ONE vector shared by every emptied chunk,
    which is the clustering failure this module exists to remove, rebuilt at the bottom.

    `_strip_body` is the same strip WITHOUT that fallback, which is the only way a caller can
    COUNT these chunks — after the fallback fires, an all-scaffolding chunk and a chunk the strip
    simply had no work to do on are byte-identical."""
    md = "## Media\n\n![photo](https://pbs.twimg.com/media/x.jpg)\n\n---\n"
    assert _strip_body(md, "x") == ""             # genuinely nothing but scaffolding
    assert strip_for_embedding(md, "x") == md     # …so the original is what gets embedded

    plain = "just some prose with no markup at all\n"
    assert _strip_body(plain, "x") == plain.strip()   # a no-op strip is NOT a fallback


def test_an_x_image_only_chunk_still_falls_back_rather_than_embedding_nothing():
    """The one place the X image rule DOESN'T take the CDN hash — and it must stay that way.

    Splitting the rule out of the url set made a new population of chunks strip to empty: an X chunk
    whose only content was `![photo](pbs.twimg.com/…)` used to keep that line under `scaffolding`
    and now has nothing left. The fail-safe catches it and embeds the original, hash included,
    because one shared empty-string vector across every such chunk is the exact clustering failure
    this module exists to remove — strictly worse than a surviving CDN hash.

    This is a constructed chunk, and deliberately so — on the real corpus the new population is
    EMPTY. Measured over all 5,552 chunks: chunks that strip to nothing are 0 before the split and
    0 after, because real chunks are large enough to always carry prose beside an image. So
    `restrip`'s `all-scaffolding chunks kept whole` counter is expected to stay at 0 when this
    ships, and a nonzero reading there still means what it always meant — a pattern turned greedy.
    The guarantee is pinned here anyway: it is what makes the counter's silence trustworthy."""
    md = "![photo](https://pbs.twimg.com/media/x.jpg)\n"
    assert _strip_body(md, "x", "scaffolding") == ""          # nothing survives the rule…
    assert strip_for_embedding(md, "x", "scaffolding") == md  # …so nothing is what we refuse to embed


def test_empty_and_none_inputs_do_not_raise():
    assert strip_for_embedding("", "x") == ""
    assert strip_for_embedding(None, "x") == ""              # type: ignore[arg-type]
    assert strip_for_embedding("hi", None) == "hi"


def test_unknown_source_type_gets_the_conservative_path():
    """An unrecognized source keeps alt text (the blog rule), because keeping a template word
    costs a token and dropping a caption costs the only description of an image."""
    md = "![a real caption](https://cdn/x.png)\n"
    assert "a real caption" in strip_for_embedding(md, "curation")


# ── profiles: the two arms must differ in exactly the URL rules, and nowhere else ──

def test_scaffolding_profile_keeps_authored_urls_and_drops_machine_emitted_ones():
    """The shipping arm. It removes text that is definitionally not the author's — footer, byline
    markup, `## Media`, thread markers, X's image markup — while leaving every url a PERSON typed.

    The boundary is AUTHOR-WRITTEN vs MACHINE-EMITTED, and this test exists because that is NOT the
    same line as "does the rule touch a url", which is what an earlier version of this test asserted.
    `t.co/RpqKgLL75L` is a link this author chose to post; `pbs.twimg.com/media/…` is X's storage
    address for a file they attached and never typed."""
    out = strip_for_embedding(X_MEDIA, "x", "scaffolding")
    # dropped: the uniform template
    assert "## Media" not in out and "[1/2]" not in out and "Thread" not in out
    assert "Bookmarked" not in out and "**" not in out
    assert "2024-03-20" not in out and "Peter van Sabben" in out
    # KEPT: the url the author wrote — this is what separates the arm from `full`
    assert "t.co" in out
    # DROPPED even here: nobody typed a CDN hash, so it is scaffolding under both profiles
    assert "pbs.twimg.com" not in out and "photo" not in out


def test_full_profile_drops_the_urls_the_scaffolding_profile_keeps():
    """The two profiles must actually differ, or the arm is a re-run of the same experiment.

    `t.co` is now the whole difference on an X atom: the image markup left the skip set, so both
    arms drop it, and the inline/bare urls are what remain to tell the arms apart."""
    full = strip_for_embedding(X_MEDIA, "x", "full")
    scaf = strip_for_embedding(X_MEDIA, "x", "scaffolding")
    assert "t.co" not in full and "t.co" in scaf
    assert len(scaf) > len(full)


def test_the_x_image_rule_runs_under_both_profiles_but_the_blog_one_does_not():
    """The split, stated as one assertion pair. Same markup, two sources, one profile.

    Measured 2026-08-11 over the real corpus: X carries 1,979 image markups with FIVE distinct alt
    values (`photo`/`video`/`image`/`cover`/`gif`, none over 12 chars) and 1,979 `pbs.twimg.com`
    urls — a media-type field and a CDN hash. Non-X carries 69 markups with 34 distinct (source,
    alt) pairs, which are captions. Collapsing these two into one profile-gated rule is what put a
    machine-emitted string in the arm that exists to preserve author-written ones."""
    # Real prose beside the image ON PURPOSE. A chunk that is NOTHING but image markup strips to
    # empty and the fail-safe hands back the original — see
    # `test_an_x_image_only_chunk_still_falls_back_rather_than_embedding_nothing`. That path would
    # make this test pass for the wrong reason, so it is kept out of the fixture.
    x_md = "Startup advantage in AI:\n\n![photo](https://pbs.twimg.com/media/GJGusyxWsAAsr6z.jpg)\n"
    blog_md = "Some prose.\n\n![Low- v High-Entropy Problem Spaces](https://cdn.example/a.png)\n"

    # X: machine-emitted on both halves → gone under scaffolding AND full
    for prof in ("full", "scaffolding"):
        out = strip_for_embedding(x_md, "x", prof)
        assert "pbs.twimg.com" not in out, prof
        assert "photo" not in out, prof
        assert "Startup advantage in AI:" in out, prof     # the author's line is untouched

    # blog: the alt is authored, so scaffolding leaves the line WHOLE and full keeps the caption
    scaf = strip_for_embedding(blog_md, "blog", "scaffolding")
    assert "Low- v High-Entropy Problem Spaces" in scaf and "cdn.example" in scaf
    full = strip_for_embedding(blog_md, "blog", "full")
    assert "Low- v High-Entropy Problem Spaces" in full and "cdn.example" not in full


def test_the_full_profile_did_not_move_when_the_x_image_rule_split():
    """`STRIP_VERSION` is ONE constant for both arms, so the `.2` bump declares a `full` store stale
    that never moved. That over-report is the documented, cheaper error — but it is only correct
    while `full` genuinely is unchanged, so the behavior it rests on is pinned here rather than
    left as a claim in a comment.

    `full` skips nothing, so every branch runs and the source split is invisible to it."""
    assert strip_for_embedding(X_MEDIA, "x", "full") == (
        "Peter van Sabben\n\nStartup Advantage in AI:\n\n"
        "AI Advantage to Startups: slow industry pace, business model conflict.\n\n"
        "Why now? Timing is critical:\n\n@sabben")


def test_scaffolding_profile_keeps_inline_link_syntax_intact():
    """`[text](url)` is left WHOLE — not rewritten to `text`. That rule was 2.27% of the corpus
    and the single biggest deviation from the plan's own DROP inventory, so it is what the arms
    are built to separate."""
    md = "See the [DNAnexus collaboration](https://ow.ly/NO8K50Qu87l).\n"
    out = strip_for_embedding(md, "x", "scaffolding")
    assert "[DNAnexus collaboration](https://ow.ly/NO8K50Qu87l)" in out


def test_identity_tokens_survive_under_both_profiles():
    """The measured 58%-precision failure mode is profile-independent — a new arm must not be a
    new way to lose the author's name."""
    for prof in ("full", "scaffolding"):
        out = strip_for_embedding(X_POST, "x", prof)
        assert "Deep Thrill" in out, prof
        assert "@DeeperThrill" in out, prof
        assert "@lecong" in strip_for_embedding(X_QUOTE, "x", prof), prof


def test_strip_version_encodes_the_profile():
    """Profile is part of what produced the vector, exactly as the regex set is. If both profiles
    stamped one identity, re-embedding from one arm to the other would leave `kb_meta` reporting
    an unchanged store while every vector moved — the silent staleness the guard exists to stop."""
    from pipeline.kb.embed_surface import strip_version
    assert strip_version("full") == STRIP_VERSION
    assert strip_version("scaffolding") != strip_version("full")
    assert "scaffolding" in strip_version("scaffolding")


def test_unknown_profile_raises_rather_than_silently_stripping():
    """A typo'd profile must not fall through to the full strip and quietly move the geometry."""
    import pytest
    with pytest.raises(ValueError):
        strip_for_embedding("hi", "x", "scafolding")
    from pipeline.kb.embed_surface import strip_version
    with pytest.raises(ValueError):
        strip_version("nope")


def test_strip_version_is_set():
    """Guarded in `kb_meta.strip_version`. A pattern change without a bump makes the store
    silently stale — the whole reason the version exists."""
    assert STRIP_VERSION and isinstance(STRIP_VERSION, str)


# ── Idempotency: re-stripping an already-stripped string is a no-op ─────────────

def test_strip_is_idempotent():
    """`restrip_embed_surface.py` detects staleness by recomputing `strip(text)` and comparing to
    the stored `embed_text`. That comparison is only meaningful if the function is stable."""
    for md, st in ((X_POST, "x"), (X_QUOTE, "x"), (X_MEDIA, "x")):
        once = strip_for_embedding(md, st)
        assert strip_for_embedding(once, st) == once


# ── What ships, and the one url claim that needed narrowing ────────────────────

def test_the_only_urls_the_scaffolding_profile_drops_are_opyts_own():
    """The `scaffolding` profile skips every url-BEARING rule (`_IMAGE`/`_LINK`/`_TCO`), so the
    obvious reading is "no url is ever lost". That reading is wrong, and it was written down wrong
    once already.

    Two rules are WHOLE-LINE drops that run before any of the url rules get a say: `_FOOTER` and
    `_THREAD_HEADER`. A url inside one of those lines dies with the line. Measured over the real
    corpus (5,551 chunks, 9,672 urls): 6,722 kept, 2,705 dropped by `_FOOTER`, 245 by
    `_THREAD_HEADER`, and ZERO dropped from anywhere else. Both drop classes are self-references —
    OPYT's own renderer pointing at the atom the reader is already holding — so what leaves is a
    `x.com/…/status/` template plus an opaque 19-digit id, with the handle lifted out first.

    This test pins the DISTINCTION, not the counts: an author-cited url stays, a self-url does
    not."""
    md = (
        "# Ada Lovelace — 2026-01-02\n"
        "\n"
        "> **Thread** · [2 posts](https://x.com/adalove/status/999000111222333444)\n"
        "\n"
        "The proof is in [Knuth's note](https://www-cs-faculty.stanford.edu/~knuth/musings.html).\n"
        "\n"
        "*Bookmarked · [Original post](https://x.com/adalove/status/999000111222333444)*"
    )
    out = strip_for_embedding(md, "x", "scaffolding")

    # the author cited this one — it survives, url and link text both
    assert "https://www-cs-faculty.stanford.edu/~knuth/musings.html" in out
    assert "Knuth's note" in out

    # OPYT rendered these two — the url goes, and the identity inside it does not
    assert "999000111222333444" not in out, "a self-referential status id survived its line drop"
    assert "@adalove" in out, "_identity_from_url must lift the handle out before the footer drops"
    assert "Ada Lovelace" in out, "the display name is the other identity token; it must survive"
    assert "Thread" not in out and "2 posts" not in out


def test_default_profile_is_the_one_the_corpus_was_embedded_with():
    """`DEFAULT_PROFILE` is read by BOTH the writer (`ingest_common._chunk_snapshot`) and the guard
    (`embed.assert_strip_version`), which is what stops them disagreeing. That design also means a
    one-character edit here silently redefines what every future vector is made of, and the guard
    would agree with the edit rather than catch it.

    So the value is pinned. Changing it is a corpus-wide re-embed (`restrip_embed_surface.py
    --profile <name> --apply`) plus a floor recalibration in `sitting_builder`, and this test
    failing is the reminder that both are owed."""
    from pipeline.kb.embed_surface import DEFAULT_PROFILE, strip_version
    assert DEFAULT_PROFILE == "scaffolding"
    assert strip_version(DEFAULT_PROFILE) == f"{STRIP_VERSION}+scaffolding"
