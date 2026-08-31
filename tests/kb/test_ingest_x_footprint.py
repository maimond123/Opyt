"""ingest_x_footprint (Stage-5 root footprint) — a confirmed Oracle's OWN X timeline → opinion atoms.

Two layers:
  • PURE filter/stitch unit tests hit `_filter_and_stitch` directly (no DB / embedder / network) —
    they encode the LOAD-BEARING order (drop reply-to-others BEFORE grouping by conversationId, or a
    self-thread essay welds to its replies-to-commenters) + the RESOLVED curation filter (drop RTs +
    replies-to-others, keep originals + self-threads, NO length gate). The reply target is
    `inReplyToUserId` vs the AUTHOR's id — `x_graphql._normalize` does not emit `inReplyToUsername`
    at all, and twitterapi.io left it EMPTY on profile tweets before that (live-verified
    2026-07-19), so the fixtures mirror the real shape (author.id + inReplyToUserId).
  • Adapter tests stub the X session + timeline walk and the VLM, proving the WIRING: self-thread →
    ONE atom, entry_mode/who_id, thread figure → VLM description in chunk text, content-hash
    idempotency, and the `limit` bound. Not the live scrape.
"""
from __future__ import annotations

import pytest

from pipeline.kb import ingest_x_footprint as fp
from pipeline.kb import schema


# ── PURE: filter + stitch (offline) ───────────────────────────────────────────────

_AUTHOR = "A"     # the timeline owner's numeric author id (uniform: the walk filters to it)


def _tw(tid, *, text="", conv=None, reply_to_uid=None, rt=False, author_id=_AUTHOR):
    """A minimal raw-twitterapi-shaped tweet. `reply_to_uid` = the numeric user id this tweet
    replies to (None = not a reply); reply_to_uid == author_id is a SELF-reply (thread continuation).
    Mirrors the live shape: inReplyToUserId populated, inReplyToUsername absent."""
    return {"id": tid, "text": text, "conversationId": conv or tid,
            "author": {"id": author_id, "userName": "carol"},
            "isRetweet": rt, "isReply": reply_to_uid is not None, "inReplyToUserId": reply_to_uid}


def test_short_aphorism_is_kept():                         # (a) NO length gate — Taleb survives
    groups = fp._filter_and_stitch([_tw("1", text="Skin in the game.")])
    assert [t["id"] for g in groups for t in g] == ["1"]


def test_reply_to_others_dropped_self_reply_kept():        # (b)
    raw = [_tw("1", text="hot take", conv="1"),
           _tw("2", text="@rando you're wrong", conv="9", reply_to_uid="RANDO"),   # reply-to-OTHER → drop
           _tw("3", text="…and another thing", conv="1", reply_to_uid=_AUTHOR)]    # self-reply → keep
    assert {t["id"] for g in fp._filter_and_stitch(raw) for t in g} == {"1", "3"}


def test_self_thread_stitches_to_one_group():              # (c) root + self-replies = ONE group
    raw = [_tw("10", text="1/ thesis", conv="10"),
           _tw("11", text="2/ argument", conv="10", reply_to_uid=_AUTHOR),
           _tw("12", text="3/ conclusion", conv="10", reply_to_uid=_AUTHOR)]
    g = fp._filter_and_stitch(raw)
    assert len(g) == 1 and [t["id"] for t in g[0]] == ["10", "11", "12"]


def test_filter_before_stitch_no_frankenstein():           # THE load-bearing ordering guard
    # A self-thread essay + a reply-to-commenter SHARE conversationId (the live @martin_casado case).
    raw = [_tw("100", text="1/ essay", conv="100"),
           _tw("101", text="2/ essay", conv="100", reply_to_uid=_AUTHOR),   # self-reply → keep
           _tw("200", text="@fan good q", conv="100", reply_to_uid="FAN")]  # reply-to-OTHER, SAME conv
    g = fp._filter_and_stitch(raw)
    assert len(g) == 1 and [t["id"] for t in g[0]] == ["100", "101"]        # reply-to-commenter NOT welded


def test_reply_to_others_dropped_despite_leading_at_mention():
    # A self-continuation carries carried-over @-mentions but replies to the AUTHOR — keep it; a real
    # reply-to-other also leads with @ — drop it. ID comparison separates them; a leading-@ heuristic can't.
    raw = [_tw("1", text="my point", conv="1"),
           _tw("2", text="@x @y Nevermind, I'm an idiot", conv="1", reply_to_uid=_AUTHOR),  # self → keep
           _tw("3", text="@stranger you're wrong", conv="1", reply_to_uid="Z")]             # other → drop
    assert {t["id"] for g in fp._filter_and_stitch(raw) for t in g} == {"1", "2"}


def test_retweets_dropped_and_dedup_by_id():
    raw = [_tw("1", rt=True), _tw("2", text="mine"),
           _tw("2", text="mine")]                          # API repeats across pages → dedup
    g = fp._filter_and_stitch(raw)
    assert {t["id"] for gr in g for t in gr} == {"2"} and sum(len(gr) for gr in g) == 1


# ── PURE: the substance filter (_keep_group, offline) — Step-1 skeleton ─────────────
# The STRUCTURAL filter above keeps short aphorisms; the SUBSTANCE filter here is where a NAKED
# post under the length bar is finally dropped. Only that one length decision is LOCKED — the
# ambiguous middle (decorative media / quote / bare link) is provisionally KEPT (deferred to OCR).

def _grpnode(text="", *, media_substance=None, quoted=None, link=None):
    """A minimal group member for _keep_group: optional substantive/decorative media (carrying the
    OCR cascade's `media_read` verdict), a quoted tweet, or a link (a full URL → entities.urls, the
    way the timeline walk carries expanded links)."""
    t = {"text": text}
    if media_substance is not None:
        t["extendedEntities"] = {"media": [
            {"type": "photo", "media_url_https": "u",
             "media_read": {"kind": "chart" if media_substance else "photo",
                            "substance": media_substance}}]}
    if quoted is not None:
        t["quoted_tweet"] = quoted
    if link:
        t["entities"] = {"urls": [{"expanded_url": link}]}
    return t


def test_keep_naked_short_dropped():                       # the ONE locked drop
    assert fp._keep_group([_grpnode("Skin in the game.")], has_article=False) == (False, "naked<200")


def test_keep_naked_long_kept():
    assert fp._keep_group([_grpnode("x" * 200)], has_article=False) == (True, "naked>=200")


def test_keep_thread_regardless():                         # a 2-post thread of short lines survives
    keep, reason = fp._keep_group([_grpnode("1/ short"), _grpnode("2/ short")], has_article=False)
    assert keep is True and reason == "thread"


def test_keep_article_regardless():
    assert fp._keep_group([_grpnode("tiny")], has_article=True) == (True, "article")


def test_keep_quoted_article_regardless():
    grp = [_grpnode("react", quoted={"id": "9", "article": {"title": "T"}})]
    assert fp._keep_group(grp, has_article=False) == (True, "quoted-article")


def test_keep_substantive_media_regardless():              # OCR found a chart/doc → artifact
    grp = [_grpnode("wow", media_substance=True)]
    assert fp._keep_group(grp, has_article=False) == (True, "media-substance")


def test_keep_decorative_media_short_dropped():            # POINT 1: photo w/o substance → naked bar
    grp = [_grpnode("lol", media_substance=False)]         # decorative photo + thin caption → DROP
    assert fp._keep_group(grp, has_article=False) == (False, "decorative-media<200")


def test_keep_decorative_media_long_survives_on_text():    # ...but ≥200 of the author's words stays
    grp = [_grpnode("x" * 200, media_substance=False)]
    assert fp._keep_group(grp, has_article=False) == (True, "decorative-media>=200")


def test_keep_quote_is_provisional():                      # thin reaction + quote → deferred, kept
    assert fp._keep_group([_grpnode("🔥", quoted={"id": "9"})], has_article=False) == (
        True, "provisional-keep")


def test_keep_bare_link_short_dropped(monkeypatch):         # POINT 2: bare link (news) → naked bar
    # The deep structural probe is the last resort before this drop (see the -deep-paper tests
    # below), so a genuinely offline check of the ORIGINAL drop decision must stub it off.
    monkeypatch.setattr(fp.link_router, "classify_link_deep", lambda u: None)
    grp = [_grpnode("must-read", link="https://www.nytimes.com/x")]   # thin praise + unopenable link
    assert fp._keep_group(grp, has_article=False) == (False, "bare-link<200")


def test_keep_bare_link_long_survives_on_text():           # Balaji's thesis + a bare link → kept
    grp = [_grpnode("x" * 200, link="https://www.nytimes.com/x")]
    assert fp._keep_group(grp, has_article=False) == (True, "bare-link>=200")


def test_keep_github_link_is_provisional():                # dispatchable link → deferred (Step 3), kept
    grp = [_grpnode("🔥", link="https://github.com/ggerganov/llama.cpp")]
    assert fp._keep_group(grp, has_article=False) == (True, "provisional-keep")


def test_keep_paper_link_is_provisional():                 # arXiv is a research paper → dispatchable
    grp = [_grpnode("great", link="https://arxiv.org/abs/2401.00001")]
    assert fp._keep_group(grp, has_article=False) == (True, "provisional-keep")


def test_keep_unlisted_paper_host_rescued_by_deep_probe(monkeypatch):
    # A paper on a host `_PAPER_HOSTS` doesn't list would otherwise sink as bare-link<200 — the
    # deep structural probe is the LAST resort BEFORE that drop, not a third free gate.
    monkeypatch.setattr(fp.link_router, "classify_link_deep",
                        lambda u: ("paper", "https://doi.org/10.1038/x", None))
    grp = [_grpnode("must-read", link="https://nature.com/articles/x")]
    assert fp._keep_group(grp, has_article=False) == (True, "bare-link-deep-paper")


def test_keep_unlisted_bare_link_still_dropped_when_deep_probe_finds_nothing(monkeypatch):
    # The probe is a genuine last resort, not a rubber stamp — a real bare link (news, a blog)
    # still hits the naked bar when the deep check also says "not a paper".
    monkeypatch.setattr(fp.link_router, "classify_link_deep", lambda u: None)
    grp = [_grpnode("must-read", link="https://www.nytimes.com/x")]
    assert fp._keep_group(grp, has_article=False) == (False, "bare-link<200")


# ── ADAPTER: fetch + build (fake embedder + mocked fetch) ─────────────────────────

@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


# A body that clears the naked-post floor (_NAKED_MIN_CHARS=200) so length-agnostic adapter tests
# aren't swept up by the substance filter — used where the test's point is NOT length.
_LONG = ("A substantive take that comfortably clears the two-hundred-character naked-post floor so "
         "the substance filter keeps it as the author's own footprint rather than dropping it as a "
         "thin fragment; this sentence is padded to length here entirely on purpose, yes indeed.")


def _raw(tid, *, text="", conv=None, reply_to_uid=None, rt=False, media=None, quoted=None, link=None):
    """A fuller raw-twitterapi tweet for the adapter path (carries author + createdAt so
    derive_x / tweet_to_markdown render). author.id='99' → who_id x:user:99.

    `quoted` = a nested quoted_tweet dict. Mirrors the LIVE shape: a genuine quote tweet can arrive
    with `isQuote` ABSENT/None while `quoted_tweet` is populated — so a fixture with a quoted object
    leaves isQuote unset, exactly the case the flag-gated render used to drop. `link` = an expanded
    outbound url (entities.urls[].expanded_url), the shape the walk carries and the Step-3 link
    dispatcher classifies."""
    t = {"id": tid, "text": text,
         "author": {"userName": "carol", "name": "Carol", "id": "99"},
         "createdAt": "Mon Jan 06 10:00:00 +0000 2026",
         "conversationId": conv or tid,
         "isRetweet": rt, "isReply": reply_to_uid is not None, "inReplyToUserId": reply_to_uid,
         "likeCount": 5, "replyCount": 1,
         "entities": {"urls": [{"expanded_url": link}]} if link else {},
         "url": f"https://x.com/carol/status/{tid}"}
    if media:
        t["extendedEntities"] = {"media": media}
    if quoted:
        t["quoted_tweet"] = quoted            # NOTE: isQuote deliberately left unset (live shape)
    return t


def _patch_fetch(monkeypatch, tweets, *, articles=False):
    """Stub the X session + the timeline walk so the adapter runs offline.

    Three seams, and all three are needed or the test reaches x.com: the cookie read, the
    handle -> rest_id lookup, and the walk itself. `_pull_own_timeline` is the right stub point —
    it is what `_fetch_profile_tweets` used to be — because everything downstream of it (filter,
    stitch, render, hash, embed, write) is what these tests are actually about.

    `articles=False` also stubs out article DETECTION, since these fixtures carry no X-Articles and
    the real detector would scan every fixture's entities on every test."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_render as xt
    monkeypatch.setattr(core, "read_x_cookies", lambda *a, **k: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "fetch_user_profile", lambda cookies, headers, h: {
        "user_id": "99", "handle": h, "display_name": h, "bio": "", "website": "",
        "bio_urls": [], "verified": False, "followers": 1})
    monkeypatch.setattr(fp, "_pull_own_timeline",
                        lambda cookies, headers, uid, since_ts, cap: [dict(t) for t in tweets])
    if not articles:
        monkeypatch.setattr(xt, "_article_tweet_id", lambda t: None)


def test_self_thread_builds_one_atom(conn, fake_embedder, monkeypatch):
    _patch_fetch(monkeypatch, [
        _raw("10", text="1/ agents compose tools", conv="10"),
        _raw("11", text="2/ into larger systems", conv="10", reply_to_uid="99"),  # self-reply
        _raw("12", text="3/ that is the thesis", conv="10", reply_to_uid="99"),
    ])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol", author_name="Carol")
    assert out["added"] == 1 and out["threads"] == 1

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id='xprofile:10'").fetchone()
    assert atom is not None
    assert atom["source_type"] == "x" and atom["entry_mode"] == "oracle-footprint"
    assert atom["who_id"] == "x:user:99" and atom["what_kind"] == "opinion"
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='xprofile:10'"))
    assert "agents compose tools" in body and "into larger systems" in body and "that is the thesis" in body
    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id='xprofile:10'").fetchone()["who_id"]
    assert who == "x:user:99"


def test_footprint_concurrent_single_writer(conn, fake_embedder, monkeypatch):
    """40 standalone posts through the REAL producer pool. Every one becomes a durable atom (no
    lost/duped atoms), producers ran on MANY threads, and — since a producer touching `conn` would
    raise SQLite's cross-thread error and drop its atom → count short — a full, dup-free count IS the
    single-writer guarantee. Also exercises the locked img-cache write (cache_put) under the pool."""
    import threading

    from pipeline.kb import vision

    _patch_fetch(monkeypatch, [_raw(str(i), text=f"{_LONG} number {i}") for i in range(1, 41)])

    producer_threads: set = set()

    def _fake_cascade(norm, cache):
        producer_threads.add(threading.get_ident())      # prove producers ran on >1 thread
        from pipeline.image_cache import cache_put
        cache_put(cache, f"u{norm['id']}", "x" * 30)      # concurrent locked cache write must not corrupt
        return 0, []
    monkeypatch.setattr(vision, "enrich_tweet_media_cascade", _fake_cascade)

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")

    assert len(producer_threads) > 1                      # concurrency was REAL, not serialized on one
    assert out["added"] == 40 and out["skipped"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE atom_id LIKE 'xprofile:%'").fetchone()[0] == 40
    assert conn.execute("SELECT COUNT(DISTINCT atom_id) FROM chunks").fetchone()[0] == 40


def test_rts_and_replies_to_others_excluded(conn, fake_embedder, monkeypatch):
    _patch_fetch(monkeypatch, [
        _raw("1", text=_LONG, conv="1"),
        _raw("2", text="@rival wrong", conv="9", reply_to_uid="rival-id"),  # reply-to-OTHER → dropped
        _raw("3", text="a retweet", rt=True),                                # RT → dropped
    ])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["added"] == 1
    assert {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")} == {"xprofile:1"}


def test_naked_short_post_is_dropped(conn, fake_embedder, monkeypatch):   # naked<200 — the LOCKED cut
    _patch_fetch(monkeypatch, [_raw("42", text="Skin in the game.")])     # naked + short → DROPPED
    out = fp.sync_x_footprint(conn, fake_embedder, handle="nntaleb")
    assert out["added"] == 0 and out["dropped"] == 1 and out["drop_reasons"] == {"naked<200": 1}
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:42'").fetchone()[0] == 0


def test_naked_long_post_is_kept(conn, fake_embedder, monkeypatch):       # naked>=200 → kept
    _patch_fetch(monkeypatch, [_raw("43", text=_LONG)])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="nntaleb")
    assert out["added"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:43'").fetchone()[0] == 1


def test_thread_figure_gets_ocr_description(conn, fake_embedder, monkeypatch):   # (d) media→OCR cascade
    media = [{"type": "photo", "media_url_https": "https://pbs.twimg.com/media/chart.jpg"}]
    _patch_fetch(monkeypatch, [_raw("7", text="the data speaks for itself", media=media)])
    from pipeline import ocr_cascade
    # A substantive chart read → rendered into the atom AND promotes the post to a kept artifact.
    monkeypatch.setattr(ocr_cascade, "read_image",
                        lambda url, context="": ocr_cascade.MediaRead(
                            "a bar chart of benchmark scores", "chart", True))
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["added"] == 1
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='xprofile:7'"))
    assert "*Image:* a bar chart of benchmark scores" in body


def test_quote_tweet_inner_post_is_extracted(conn, fake_embedder, monkeypatch):
    # A quote tweet's meaning lives in what it QUOTES — the author's bare reaction is
    # decontextualized (and mis-attributable) without the quoted post. The live profile fetch nulls
    # `isQuote` but populates `quoted_tweet`, so the render must gate on the OBJECT, not the flag.
    quoted = {"id": "555", "text": "Kimi K3 weights will be open in the coming days.",
              "author": {"userName": "yulun_du", "name": "Yulun Du"}, "entities": {}}
    _patch_fetch(monkeypatch, [_raw("30", text="Hurray!! cannot wait to play.", quoted=quoted)])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["added"] == 1
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='xprofile:30'"))
    # The atom carries BOTH the author's reaction AND the quoted author + quoted substance.
    assert "Hurray" in body
    assert "yulun_du" in body and "Kimi K3 weights will be open" in body
    # …and the structural flag no longer lies (derived from the object, not the nulled isQuote).
    import json
    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='xprofile:30'").fetchone()["payload"])
    assert payload["is_quote"] is True


def test_quoted_article_body_arrives_with_the_timeline(conn, fake_embedder, monkeypatch):
    """A quote OF an X-Article renders the quoted body, not its teaser.

    This used to cost a second paid `/twitter/article` call per quoted article, because the paid
    profile fetch returned only a stub node. The timeline walk asks for
    `withArticleRichContentState`, and `x_graphql._normalize` recurses into
    `quoted_status_result` carrying the whole `article` node with it — so the body is already here
    and the fetch is gone. Losing it would leave the atom holding a reaction plus a title, and the
    long-form it points at would never enter the corpus."""
    quoted = {"id": "888", "text": "https://t.co/x",
              "author": {"userName": "satyanadella", "name": "Satya Nadella"},
              "entities": {"urls": [{"expanded_url": "http://x.com/i/article/999"}]},
              "article": {"article_results": {"result": {
                  "title": "The Reverse Information Paradox",
                  "content_state": {"blocks": [
                      {"type": "unstyled",
                       "text": "Kenneth Arrow described a paradox in the IP market."}]}}}}}
    _patch_fetch(monkeypatch, [_raw("40", text="Protect the loop.", quoted=quoted)], articles=True)
    out = fp.sync_x_footprint(conn, fake_embedder, handle="ccatalini")
    assert out["added"] == 1
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='xprofile:40'"))
    assert "Protect the loop" in body                                   # the author's reaction
    assert "Kenneth Arrow described a paradox in the IP market" in body  # the quoted BODY


def test_content_hash_idempotent(conn, fake_embedder, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("1", text=_LONG, conv="1")])
    fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")   # unchanged body → skip embed
    assert out["added"] == 0 and out["skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:1'").fetchone()[0] == 1


def test_limit_caps_new_atoms(conn, fake_embedder, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("1", text=_LONG + " one", conv="1"),
                               _raw("2", text=_LONG + " two", conv="2"),
                               _raw("3", text=_LONG + " three", conv="3")])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol", limit=2)
    assert out["added"] == 2
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2


# ── STEP 3: link dispatch — a referenced github/paper → its OWN artifact atom + an Oracle vouch ──
# The reacting tweet already survives the substance filter (a dispatchable link → provisional-keep);
# here the LINK itself is resolved into an artifact atom and the Oracle gets a
# (this-oracle → references → artifact) vouch — the trust edge David asked for.

def _minted(conn, oracle=None):
    """Artifact atoms the Step-3 link dispatcher captured (`entry_mode='author_referenced'`).

    Replaces `_vouch_edges`, which read the Oracle→artifact `references` edge until the `edges`
    table was deleted 2026-08-23 for having no reader. WHICH Oracle pointed at an artifact is no
    longer recorded anywhere; THAT the artifact was captured still is, and capture is what these
    tests are really guarding. `oracle` is accepted and ignored so the call sites still read as
    "this Oracle's reference landed"."""
    # Keyed on "not an X opinion atom" rather than `entry_mode`, because the paper tests stub
    # `atomize_paper` and their fakes do not stamp one. The Oracle's own tweets are the only
    # `source_type='x'` rows here, so everything else is something the dispatcher captured.
    return {r[0] for r in conn.execute(
        "SELECT atom_id FROM atoms WHERE source_type != 'x'")}


def test_github_reference_dispatched_and_vouched(conn, fake_embedder, monkeypatch):
    # An Oracle points at a repo → the repo becomes its OWN artifact atom (attributed to the repo
    # OWNER, entry_mode 'author_referenced') AND the Oracle gets a references vouch onto it.
    _patch_fetch(monkeypatch, [_raw("50", text="best C++ inference repo",
                                    link="https://github.com/ggerganov/llama.cpp")])
    from pipeline.ingestion.sources import github as gh_ing
    from pipeline.kb import ingest_github as gh
    repo = {"name": "llama.cpp", "owner": {"login": "ggerganov"}, "language": "C++",
            "stargazers_count": 60000, "forks_count": 8000, "description": "LLM inference in C/C++",
            "topics": ["llm"], "pushed_at": "2026-01-05T00:00:00Z", "license": {"spdx_id": "MIT"},
            "html_url": "https://github.com/ggerganov/llama.cpp"}
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: repo)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: "# llama.cpp\ninference")

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["dispatched"] == 1 and out["dispatch_kinds"] == {"github": 1}
    row = conn.execute("SELECT who_id, what_kind, entry_mode FROM atoms "
                       "WHERE atom_id='github:ggerganov/llama.cpp'").fetchone()
    assert row is not None and row["who_id"] == "github:ggerganov" and row["what_kind"] == "artifact"
    assert row["entry_mode"] == "author_referenced"
    assert "github:ggerganov/llama.cpp" in _minted(conn)      # the referenced repo was captured
    assert conn.execute(                                                     # the reacting tweet still lands
        "SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:50'").fetchone()[0] == 1


def test_paper_reference_dispatched_and_vouched(conn, fake_embedder, monkeypatch):
    # An arXiv link → atomize_paper (mocked) mints the paper as REFERENCED, and the Oracle vouches.
    _patch_fetch(monkeypatch, [_raw("51", text="must read", link="https://arxiv.org/abs/2401.00001")])
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00001", "url": "https://arxiv.org/abs/2401.00001",
             "externalIds": {"ArXiv": "2401.00001"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    # The artifact PREFETCH pulls the PDF now, so it must be stubbed independently of atomize_paper
    # — mocking the miner no longer shields the fetch it used to own.
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")
    seen_modes = []

    def fake_atomize(conn, embedder, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):                     # **kw: sink/on_written/fulltext
        seen_modes.append(entry_mode)
        schema.upsert_atom(conn, {"atom_id": "paper:arXiv:2401.00001", "source_type": "paper",
                                  "who_id": "scholar:x", "what_kind": "artifact"})
        return "paper:arXiv:2401.00001"
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["dispatched"] == 1 and out["dispatch_kinds"] == {"paper": 1}
    assert seen_modes == ["author_referenced"]                              # minted as referenced, not authored
    assert "paper:arXiv:2401.00001" in _minted(conn)


def test_substack_reference_dispatched_and_vouched(conn, fake_embedder, monkeypatch):
    # A referenced Substack POST → its own opinion atom + the Oracle vouch (same treatment as
    # github/paper: the Oracle pointed at it, who_id is the POST's author, not the Oracle).
    _patch_fetch(monkeypatch, [_raw("52", text="great post", link="https://joe.substack.com/p/x")])
    from pipeline.kb import ingest_substack as isub

    def fake_mint(conn, embedder, url, *, entry_mode="author_referenced", seen=None, img_cache=None):
        assert entry_mode == "author_referenced"
        schema.upsert_atom(conn, {"atom_id": "substack:777", "source_type": "substack"})
        return "substack:777"
    monkeypatch.setattr(isub, "substack_atom_from_url", fake_mint)

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["dispatched"] == 1 and out["dispatch_kinds"] == {"substack": 1}
    assert "substack:777" in _minted(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:52'").fetchone()[0] == 1


def test_quoted_node_link_not_dispatched(conn, fake_embedder, monkeypatch):
    # A github link inside the QUOTED tweet is the quoted author's reference, not this Oracle's act.
    # The dispatcher scans OWN nodes only → nothing minted or vouched.
    quoted = {"id": "9", "text": "check my repo", "author": {"userName": "bob"},
              "entities": {"urls": [{"expanded_url": "https://github.com/bob/thing"}]}}
    _patch_fetch(monkeypatch, [_raw("53", text="nice", quoted=quoted)])     # OUTER tweet carries no link
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["dispatched"] == 0 and _minted(conn) == set()
    assert conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE atom_id='github:bob/thing'").fetchone()[0] == 0


def test_referenced_pdf_is_minted_and_vouched(conn, fake_embedder, monkeypatch):
    # A raw .pdf has no scholarly id and no attested author — IRRELEVANT for a reference: who_id is
    # the artifact's, never the Oracle's, so it mints like any paper (the adapter fetches the full
    # body via the pdf url itself). The Oracle vouches for it.
    from pipeline.kb import ingest_papers as ip
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: {
        "paperId": "url:acme.com/deck.pdf", "url": url, "openAccessPdf": {"url": url}})
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")     # the prefetch pulls the PDF now
    seen_modes = []

    def fake_atomize(conn, embedder, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):                     # **kw: sink/on_written/fulltext
        seen_modes.append(entry_mode)
        schema.upsert_atom(conn, {"atom_id": ip.paper_atom_id(p), "source_type": "paper"})
        return ip.paper_atom_id(p)
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)
    _patch_fetch(monkeypatch, [_raw("55", text="slides", link="https://acme.com/deck.pdf")])

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["dispatched"] == 1 and out["dispatch_kinds"] == {"paper": 1}
    assert seen_modes == ["author_referenced"]
    assert "paper:url:acme.com/deck.pdf" in _minted(conn)


def test_reference_vouch_idempotent_mint_once(conn, fake_embedder, monkeypatch):
    # THE correctness point: a paper ALREADY in the store (a prior Oracle brought it in) must still
    # get a NEW Oracle's vouch — even though atomize_paper's immutability dedup returns None and
    # stores nothing. The vouch goes through upsert_edge directly, not the adapter's edge list.
    schema.upsert_atom(conn, {"atom_id": "paper:arXiv:2401.00001", "source_type": "paper"})  # pre-existing
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00001", "url": "https://arxiv.org/abs/2401.00001",
             "externalIds": {"ArXiv": "2401.00001"}}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    called = []
    monkeypatch.setattr(ip, "atomize_paper", lambda *a, **k: called.append(1))   # must NOT be called

    d = fp.LinkDispatcher(conn, fake_embedder)
    kinds = d.dispatch([_raw("60", text="ditto", link="https://arxiv.org/abs/2401.00001")],
                       "x:user:OTHER")
    assert kinds == {"paper": 1}
    assert not called                                                       # present → vouch-only, no re-mint
    assert "paper:arXiv:2401.00001" in _minted(conn, oracle="x:user:OTHER")


def test_unlisted_paper_host_dispatched_via_deep_probe(conn, fake_embedder, monkeypatch):
    # A Nature link never matches `_PAPER_HOSTS`, so the FAST classifier returns None. The deep
    # probe (mocked here — the network fetch itself is `classify_link_deep`'s own concern, tested
    # in test_link_router.py) finds a citation_doi meta tag and rewrites to a doi.org url that
    # `paper_from_url` mints normally, same as any other referenced paper.
    monkeypatch.setattr(fp.link_router, "classify_link_deep",
                        lambda u: ("paper", "https://doi.org/10.1038/s41586-021-03819-2", None))
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "DOI:10.1038/s41586-021-03819-2",
             "url": "https://doi.org/10.1038/s41586-021-03819-2",
             "externalIds": {"DOI": "10.1038/s41586-021-03819-2"}, "authors": []}
    seen_urls = []

    def fake_paper_from_url(url, enrich=True, content_type=None):
        seen_urls.append(url)                # must be the REWRITTEN doi.org url, not the nature.com one
        return dict(paper)
    monkeypatch.setattr(ip, "paper_from_url", fake_paper_from_url)

    def fake_atomize(conn, embedder, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):
        schema.upsert_atom(conn, {"atom_id": "paper:DOI:10.1038/s41586-021-03819-2",
                                  "source_type": "paper", "who_id": "scholar:x",
                                  "what_kind": "artifact"})
        return "paper:DOI:10.1038/s41586-021-03819-2"
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    d = fp.LinkDispatcher(conn, fake_embedder)
    kinds = d.dispatch([_raw("80", text="huge if true", link="https://nature.com/articles/x")],
                       "x:user:ORACLE")
    assert kinds == {"paper": 1}
    # Called twice (the offline `predicted_atom_id` pre-check, then the mint itself) — same as any
    # other paper reference (see test_paper_reference_dispatched_and_vouched); the point here is
    # every call used the REWRITTEN doi.org url, never the original nature.com one.
    assert seen_urls and all(u == "https://doi.org/10.1038/s41586-021-03819-2" for u in seen_urls)
    assert "paper:DOI:10.1038/s41586-021-03819-2" in _minted(conn, oracle="x:user:ORACLE")


def test_unlisted_pdf_without_pdf_shaped_url_dispatched_via_content_type_hint(conn, fake_embedder,
                                                                              monkeypatch):
    # A `/download?id=123`-style redirect that SERVES a pdf: no host match, no `.pdf` in the path,
    # no citation_doi meta tag either — the ONLY signal is the real Content-Type the deep probe
    # observed. Proves the hint actually reaches `_parse_paper_url`, not just `classify_link_deep`.
    link = "https://acme-journal.example/download?id=123"
    monkeypatch.setattr(fp.link_router, "classify_link_deep",
                        lambda u: ("paper", link, "application/pdf; charset=binary"))
    from pipeline.kb import ingest_papers as ip
    seen_content_types = []

    real_parse = ip._parse_paper_url

    def spying_parse(url, *, content_type=None):
        seen_content_types.append(content_type)
        return real_parse(url, content_type=content_type)
    monkeypatch.setattr(ip, "_parse_paper_url", spying_parse)

    def fake_atomize(conn, embedder, p, **kw):
        aid = ip.paper_atom_id(p)
        schema.upsert_atom(conn, {"atom_id": aid, "source_type": "paper",
                                  "who_id": "scholar:x", "what_kind": "artifact"})
        return aid
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    d = fp.LinkDispatcher(conn, fake_embedder)
    kinds = d.dispatch([_raw("82", text="preprint", link=link)], "x:user:ORACLE")
    assert kinds == {"paper": 1}
    assert "application/pdf; charset=binary" in seen_content_types
    assert f"paper:url:{link}" in _minted(conn, oracle="x:user:ORACLE")


def test_unlisted_non_paper_link_stays_undispatched(conn, fake_embedder, monkeypatch):
    # The deep probe correctly says "not a paper" (a plain news article) — the link stays a bare,
    # undispatched reference, same as before this fallback existed.
    monkeypatch.setattr(fp.link_router, "classify_link_deep", lambda u: None)
    d = fp.LinkDispatcher(conn, fake_embedder)
    kinds = d.dispatch([_raw("81", text="wow", link="https://www.nytimes.com/x")], "x:user:ORACLE")
    assert kinds == {}


# ── Lookback window: default + the hard 2-year ceiling ────────────────────────

def test_resolve_since_default_is_6mo():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert fp._resolve_since(None, now) == now - timedelta(days=fp._DEFAULT_LOOKBACK_DAYS)


def test_resolve_since_clamps_beyond_2yr():
    # A 5-year request is FLOORED at the 2-year ceiling — X is an ephemeral stream, and the
    # timeline walk's own pagination depth runs out well before five years of history anyway.
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert fp._resolve_since(now - timedelta(days=5 * 365), now) == now - timedelta(days=fp._MAX_LOOKBACK_DAYS)


def test_resolve_since_passthrough_within_ceiling():
    # A 1-year request is inside the ceiling → honored unchanged.
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    one_yr = now - timedelta(days=365)
    assert fp._resolve_since(one_yr, now) == one_yr


# ── the media prefetch: images are the unit of dispatch, and `limit` still bounds spend ──────────

def test_prefetch_reads_every_image_before_the_render_pass(conn, fake_embedder, monkeypatch):
    """The unbounded run reads images UP FRONT, one future per image, so `run_concurrent`'s per-group
    threads never serialize on a round-trip. `late_reads` is the escape hatch: anything above 0 means
    an image was read inside a producer thread after all — the serialization the prefetch removes,
    partially back — and a leak costs only latency, so it would otherwise look like a healthy run."""
    media = [{"type": "photo", "media_url_https": f"https://pbs.twimg.com/media/{i}.jpg"}
             for i in range(3)]
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=media)])
    from pipeline import ocr_cascade
    monkeypatch.setattr(ocr_cascade, "read_image",
                        lambda url, context="": ocr_cascade.MediaRead("a chart", "chart", True))

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")

    assert out["media_prefetch"]["read"] == 3
    assert out["late_reads"] == 0, "the render pass must not have paid for a single read"
    assert "media_prefetch" in out["stage_seconds"]


def test_limit_skips_the_prefetch_so_a_cost_cap_stays_a_cost_cap(conn, fake_embedder, monkeypatch):
    """`limit` bounds SPEND on a bounded run, and `_work` enforces it by early-skipping groups before
    they reach the VLM. A prefetch reads every image in the WINDOW first, so running it under `limit`
    would turn the cap into a floor — paying for images of groups that are never ingested. A latency
    optimization must not silently un-bound a cost the caller asked to bound."""
    reads: list[str] = []
    media = [{"type": "photo", "media_url_https": f"https://pbs.twimg.com/media/{i}.jpg"}
             for i in range(6)]
    _patch_fetch(monkeypatch, [_raw(str(i), text=_LONG + f" n{i}", conv=str(i), media=media)
                               for i in range(1, 6)])
    from pipeline import ocr_cascade
    monkeypatch.setattr(ocr_cascade, "read_image",
                        lambda url, context="": (reads.append(url)
                                                 or ocr_cascade.MediaRead("a chart", "chart", True)))

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol", limit=1)

    assert out["added"] == 1
    assert out["media_prefetch"] == {}, "no prefetch may run when `limit` bounds the spend"
    # 5 groups x 6 images = 30 refs in the window; the cap must keep the paid reads far under that.
    assert len(set(reads)) <= 6, f"limit leaked: {len(set(reads))} unique images paid for"


# ── the profile must account for its own wall clock (2026-08-02) ─────────────────────────────────

def test_link_dispatch_is_timed(conn, fake_embedder, monkeypatch):
    """It was not, and that cost a wrong prediction. `_dispatch_referenced_links` fetches + embeds
    every referenced paper/repo SERIALLY on the consumer, but had no `timer.stage()`, so it scored
    zero in a profile that otherwise looked complete — and 'zero' is indistinguishable from 'free'."""
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG + " https://github.com/foo/bar")])

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")

    assert "link_dispatch" in out["stage_seconds"]
    assert out["stage_latency"]["link_dispatch"]["count"] >= 1


def test_every_phase3_second_lands_in_a_parent_stage(conn, fake_embedder, monkeypatch):
    """`process` is phase 3's wall clock; `produce`/`consume` are the two halves under it. Without a
    parent that spans the WHOLE phase there is no denominator, so an untimed child cannot be seen —
    it just makes the timed children look like a bigger share of a total nobody measured."""
    _patch_fetch(monkeypatch, [_raw(str(i), text=_LONG + f" n{i}", conv=str(i)) for i in range(1, 4)])

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    secs = out["stage_seconds"]

    assert {"process", "produce", "consume"} <= set(secs)
    # The consumer is SERIAL, so it runs inside the phase wall. (`produce` is thread-seconds across
    # the pool and may legitimately EXCEED it — comparing the two is the mistake, not a failure.)
    assert secs["consume"] <= secs["process"] + 1e-6


def test_residual_is_reported_and_never_negative(conn, fake_embedder, monkeypatch):
    """Parent−Σ(children), surfaced so unmeasured work is a NUMBER instead of a silence. Negative
    means a child is being counted outside its declared parent — the bookkeeping lying, which would
    make the residual useless exactly when it matters."""
    _patch_fetch(monkeypatch, [_raw(str(i), text=_LONG + f" n{i}", conv=str(i)) for i in range(1, 4)])

    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")

    assert set(out["stage_residual"]) == {"produce", "consume"}
    for parent, left in out["stage_residual"].items():
        assert left >= -1e-6, f"{parent} residual {left} < 0 — a child is timed outside its parent"


# ── batched artifact writes: the vouch must survive deferral (2026-08-02) ────────────────────────

def _sink(conn, fake_embedder, flush_chunks=10_000):
    """A sink that will NOT auto-flush — so a test can observe the window where an atom is
    submitted but not yet durable, which is the entire hazard this feature introduces."""
    from pipeline.kb.ingest_common import AtomSink
    return AtomSink(conn, fake_embedder, flush_chunks=flush_chunks)


def test_vouch_is_deferred_until_the_artifact_is_durable(conn, fake_embedder, monkeypatch):
    """THE regression guard for batching. With a sink the artifact is buffered in RAM, so
    `_atom_present` answers False for an atom that lands seconds later. The old code asked exactly
    that question and vouched on the answer — under batching it would answer "no" every time and
    drop EVERY vouch, leaving papers with no record of who referenced them. Indistinguishable from
    an Oracle who linked nothing, so nothing would have caught it."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00002", "url": "https://arxiv.org/abs/2401.00002",
             "externalIds": {"ArXiv": "2401.00002"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")   # no real PDF pull

    sink = _sink(conn, fake_embedder)
    d = fp.LinkDispatcher(conn, fake_embedder, sink=sink)
    kinds = d.dispatch([_raw("70", text="read this", link="https://arxiv.org/abs/2401.00002")],
                       "x:user:ORACLE")

    assert kinds == {"paper": 1}, "the reference must still be counted while in flight"
    assert _minted(conn, oracle="x:user:ORACLE") == set(), \
        "nothing may be vouched before the atom is durable — that would be a dangling edge"

    sink.close()                                          # flush → the atom lands → callback fires
    assert "paper:arXiv:2401.00002" in _minted(conn, oracle="x:user:ORACLE")


def test_second_reference_in_one_window_rides_the_first_submit(conn, fake_embedder, monkeypatch):
    """Two tweets link the same paper inside one flush window. `_atom_present` MISSES (buffered), so
    without an in-memory in-flight marker the second reference re-mints and re-embeds — paying twice
    for one artifact. `_pending` is that marker, and it is marked at SUBMIT, not at write."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00003", "url": "https://arxiv.org/abs/2401.00003",
             "externalIds": {"ArXiv": "2401.00003"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")
    mints = []
    real = ip.atomize_paper
    monkeypatch.setattr(ip, "atomize_paper",
                        lambda *a, **k: (mints.append(1), real(*a, **k))[1])

    sink = _sink(conn, fake_embedder)
    d = fp.LinkDispatcher(conn, fake_embedder, sink=sink)
    link = "https://arxiv.org/abs/2401.00003"
    assert d.dispatch([_raw("71", text="a", link=link)], "x:user:ORACLE") == {"paper": 1}
    assert d.dispatch([_raw("72", text="b", link=link)], "x:user:ORACLE") == {"paper": 1}

    assert len(mints) == 1, f"the artifact was minted {len(mints)}x for two references"
    sink.close()
    assert "paper:arXiv:2401.00003" in _minted(conn, oracle="x:user:ORACLE")


def test_two_oracles_in_one_window_both_get_their_vouch(conn, fake_embedder, monkeypatch):
    """`_pending` maps an atom to a LIST of who_ids, not one. Collapsing it to a single value would
    silently drop the second Oracle's attestation — the exact 'mint once, vouch every time' rule the
    dispatcher exists to hold, broken by the optimization meant to preserve it."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00004", "url": "https://arxiv.org/abs/2401.00004",
             "externalIds": {"ArXiv": "2401.00004"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")   # no real PDF pull

    sink = _sink(conn, fake_embedder)
    d = fp.LinkDispatcher(conn, fake_embedder, sink=sink)
    link = "https://arxiv.org/abs/2401.00004"
    d.dispatch([_raw("73", text="a", link=link)], "x:user:ONE")
    d.dispatch([_raw("74", text="b", link=link)], "x:user:TWO")
    sink.close()

    assert "paper:arXiv:2401.00004" in _minted(conn, oracle="x:user:ONE")
    assert "paper:arXiv:2401.00004" in _minted(conn, oracle="x:user:TWO")


def test_a_failed_artifact_write_leaves_no_dangling_vouch(conn, fake_embedder, monkeypatch):
    """Fail-safe, restated for the deferred path. An edge points at an atom BY ID; if the atom never
    lands, the edge points into empty space and 'what has this Oracle referenced?' returns an id
    that resolves to no row. On the synchronous path `_atom_present` guarded that. On the sink path
    the guard is that `on_written` simply never fires for an atom the flush dropped."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00005", "url": "https://arxiv.org/abs/2401.00005",
             "externalIds": {"ArXiv": "2401.00005"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")   # no real PDF pull

    sink = _sink(conn, fake_embedder)
    d = fp.LinkDispatcher(conn, fake_embedder, sink=sink)
    d.dispatch([_raw("75", text="doomed", link="https://arxiv.org/abs/2401.00005")], "x:user:ORACLE")

    from pipeline.kb.embed import EmbedError
    monkeypatch.setattr(fake_embedder, "embed",
                        lambda *a, **k: (_ for _ in ()).throw(EmbedError("dead")))
    sink.close()                                  # batch fails, then per-atom isolation also fails

    assert _minted(conn, oracle="x:user:ORACLE") == set(), \
        "an artifact that never landed must never be vouched to"


def test_prefetched_payload_skips_the_inline_fetch(conn, fake_embedder, monkeypatch):
    """The point of the prefetch: dispatch must CONSUME the parallel fetch, not redo it on the
    serial consumer. If this regresses, `prefetch_hits` in the run summary is what surfaces it."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00006", "url": "https://arxiv.org/abs/2401.00006",
             "externalIds": {"ArXiv": "2401.00006"}, "authors": []}
    fetches = []
    monkeypatch.setattr(ip, "paper_from_url",
                        lambda url, enrich=True: (fetches.append(enrich), dict(paper))[1])
    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda p: pytest.fail("resolve_fulltext ran despite a prefetched fulltext"))
    link = "https://arxiv.org/abs/2401.00006"

    d = fp.LinkDispatcher(conn, fake_embedder,
                          prefetched={link: {"paper": dict(paper), "fulltext": "body text"}})
    assert d.dispatch([_raw("76", text="x", link=link)], "x:user:ORACLE") == {"paper": 1}

    assert d.prefetch_hits == 1
    assert fetches == [False], "only the free id-parse may run; the S2 enrich was prefetched"


def test_a_missing_prefetch_entry_falls_back_to_an_inline_fetch(conn, fake_embedder, monkeypatch):
    """Fail-safe: a url whose prefetch failed is simply ABSENT from the map. Dispatch must still
    mint it — slowly, inline — rather than skip the artifact. Correct, and invisible without
    `prefetch_hits`, since a silently-skipped artifact looks like a post that linked nothing."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00007", "url": "https://arxiv.org/abs/2401.00007",
             "externalIds": {"ArXiv": "2401.00007"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "inline body")

    d = fp.LinkDispatcher(conn, fake_embedder, prefetched={})     # prefetch found nothing
    assert d.dispatch([_raw("77", text="x", link="https://arxiv.org/abs/2401.00007")],
                      "x:user:ORACLE") == {"paper": 1}
    assert d.prefetch_hits == 0
    assert "paper:arXiv:2401.00007" in _minted(conn, oracle="x:user:ORACLE")


def test_prefetch_dispatches_one_future_per_artifact(conn, monkeypatch):
    """Granularity, one layer down from the images: the unit of parallel work is ONE artifact."""
    import threading
    peak = {"n": 0, "cur": 0}
    lock = threading.Lock()
    release = threading.Event()

    def _fake(url, kind):
        with lock:
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
        release.wait(timeout=2.0)
        with lock:
            peak["cur"] -= 1
        return {"paper": {"paperId": url}, "fulltext": ""}
    monkeypatch.setattr(fp, "_prefetch_one_artifact", _fake)
    groups = [[_raw(str(i), text="x", link=f"https://arxiv.org/abs/2401.0000{i}")] for i in range(1, 5)]
    t = threading.Timer(0.3, release.set)
    t.start()
    out = fp.prefetch_referenced_artifacts(groups, conn, workers=20)
    t.cancel()

    assert out["fetched"] == 4 and out["unique"] == 4
    assert peak["n"] > 1, f"artifacts still fetched serially (peak in-flight={peak['n']})"


def test_prefetch_dedupes_and_ignores_bare_links(conn, monkeypatch):
    """438 of 455 links in a real run are bare (x.com, t.co, a company site). They classify to None
    with pure string matching and must never reach the pool — and one url referenced twice is one
    fetch, since concurrent misses cannot dedup against each other the way a serial pass did."""
    calls = []
    monkeypatch.setattr(fp, "_prefetch_one_artifact",
                        lambda url, kind: (calls.append(url), {"paper": {}, "fulltext": ""})[1])
    link = "https://arxiv.org/abs/2401.00009"
    groups = [[_raw("80", text="a", link=link)], [_raw("81", text="b", link=link)],
              [_raw("82", text="c", link="https://x.com/someone/status/1")],
              [_raw("83", text="d", link="https://acme.com/blog/post")]]
    out = fp.prefetch_referenced_artifacts(groups, conn, workers=4)

    assert calls == [link], f"prefetched {calls}, must be the one dispatchable url"
    assert out["links"] == 2 and out["unique"] == 1


def test_limit_skips_the_artifact_prefetch_too(conn, fake_embedder, monkeypatch):
    """Same bound as the media prefetch, same reason: `limit` caps SPEND, and a pass that walks the
    whole window would pay to fetch artifacts of groups the run will never ingest."""
    monkeypatch.setattr(fp, "prefetch_referenced_artifacts",
                        lambda *a, **k: pytest.fail("artifact prefetch ran under `limit`"))
    _patch_fetch(monkeypatch, [_raw(str(i), text=_LONG + f" n{i}", conv=str(i)) for i in range(1, 4)])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol", limit=1)
    assert out["artifact_prefetch"] == {}


def test_residual_grows_when_a_child_goes_unregistered(conn, fake_embedder):
    """The drift property. `_STAGE_NESTING` is hand-maintained, so it WILL fall behind the code; the
    only question is which way it fails. An unregistered child must inflate its parent's residual
    ('look inside this parent'), never shrink it ('nothing to see') — over-reporting is recoverable,
    under-reporting is the exact bug this whole mechanism exists to catch."""
    totals = {"consume": 10.0, "link_dispatch": 4.0, "embed": 2.0, "brand_new_stage": 3.0}
    assert fp._stage_residual(totals)["consume"] == 4.0     # 10 − 4 − 2, the new child NOT subtracted


# ── An unreadable account is BLOCKED, not ERROR ──────────────────────────────

def test_an_unreadable_profile_is_blocked_not_error(conn, fake_embedder, monkeypatch):
    """An account whose profile will not load must come back as a BLOCKED run — "nothing written,
    nothing marked seen, retry" — not as ERROR, which tells the user a caller fault needs looking
    at. `classify_run` keys BLOCKED off `undetermined`.

    `undetermined` is the honest verdict because this path cannot tell suspended from protected
    from renamed, and those have different answers. Guessing one would be worse than saying so.

    Also pins the returned-not-raised contract: an adapter that RAISES sinks the caller's OTHER
    sources, which is why every hard stop comes back as a summary. (This test used to stand on
    twitterapi.io answering 200-with-empty-body under a soft quota; the provider is gone, the
    contract is not.)"""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.kb import ingest_common

    monkeypatch.setattr(core, "read_x_cookies", lambda *a, **k: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "fetch_user_profile", lambda *a, **k: None)

    summ = fp.sync_x_footprint(conn, fake_embedder, handle="someone")
    assert ingest_common.classify_run(summ) == ingest_common.RUN_BLOCKED
    assert summ["added"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0   # nothing written
