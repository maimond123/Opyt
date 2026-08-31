"""The frozen proof that the free GraphQL path returns what twitterapi.io returned.

This is the evidence the cutover rested on, made permanent. On 2026-08-30 the same account and the
same window were pulled through both paths and compared across eight fields: 0 mismatches over 52
shared tweets. That measurement was only reproducible while the paid path still existed, so it was
frozen before the deletion — the RAW `UserTweets` result nodes on one side, and the eight-field
probe of twitterapi.io's answer for the same tweets on the other.

The test replays the raw nodes through `x_graphql._normalize` and asserts they reproduce the
recorded answers. Nothing here imports twitterapi.io, and nothing touches the network: the
expectations are frozen values, so this keeps working after the vendor is gone. What it defends is
the live risk, which is not the vendor — it is `_normalize` quietly dropping a field that only a
side-by-side pull would have caught.

The 13 fixture tweets are a COVERING subset of the 52, not a sample: an X Article, tweets with
media, with expanded urls, with a quoted tweet, and plain ones. Repetition adds bytes, not power.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.ingestion import x_graphql as xg

FIXTURE = Path(__file__).parent.parent / "fixtures" / "x" / "twitterapi_parity.json"


def _probe(t: dict) -> dict:
    """The eight fields the live comparison ran over, verbatim. Kept identical to the harness that
    produced the recorded side — a probe that drifted would compare two different questions."""
    return {
        "text":    (t.get("text") or "").strip(),
        "created": t.get("createdAt"),
        "author":  ((t.get("author") or {}).get("userName") or ""),
        "likes":   t.get("likeCount"),
        "quoted":  bool(t.get("quoted_tweet")),
        "media":   len(((t.get("extendedEntities") or {}).get("media")) or t.get("media") or []),
        "urls":    sorted(u.get("expanded_url", "")
                          for u in (t.get("entities") or {}).get("urls", [])),
        "article": bool(t.get("article")),
    }


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(FIXTURE.read_text())


def test_every_field_matches_what_twitterapi_returned(frozen):
    """THE parity claim. `likes` is included even though a like count moves after capture — both
    sides were read within seconds of each other, so the recorded pair is a real comparison; if
    this ever fails on `likes` alone, the fixture is stale, not the code."""
    mismatches = []
    for tid, node in sorted(frozen["graphql_raw"].items()):
        norm = xg._normalize(node)
        assert norm and str(norm.get("id")) == tid, f"{tid}: _normalize returned nothing usable"
        got, want = _probe(norm), frozen["twitterapi_expected"][tid]
        for field in frozen["fields"]:
            if got[field] != want[field]:
                mismatches.append((tid, field, want[field], got[field]))
    assert not mismatches, "\n".join(
        f"{tid}.{f}: twitterapi={w!r} graphql={g!r}" for tid, f, w, g in mismatches)


def test_the_fixture_still_covers_every_field_shape(frozen):
    """A guard on the evidence itself. Every field above is a no-op on a tweet that has none of
    that thing — an all-plain fixture would pass the parity test while proving nothing about
    articles, media, quotes or link expansion. If a later trim breaks this, the parity test above
    has quietly stopped testing what its name claims."""
    exp = frozen["twitterapi_expected"].values()
    assert sum(1 for e in exp if e["article"]) >= 1, "no X Article left in the fixture"
    assert sum(1 for e in exp if e["media"]) >= 1, "no tweet with media left"
    assert sum(1 for e in exp if e["urls"]) >= 1, "no tweet with expanded urls left"
    assert sum(1 for e in exp if e["quoted"]) >= 1, "no quoted tweet left"


def test_an_x_article_renders_a_real_body_not_a_teaser(frozen):
    """The fidelity claim that motivated the whole cutover — and the one that nearly shipped wrong.

    The free timeline walk returns an `article` node either way. Without
    `withArticleRichContentState` that node is a 1.1 KB TEASER: cover image, title, `preview_text`,
    no body. The request still returns 200, `article` is still truthy, and every "does this have an
    article" check still passes — so the body goes missing in silence. The captured fixture held
    exactly that stub until the toggle was found.

    So this asserts on the RENDERED MARKDOWN, not on the node. A shape check would have been
    satisfied by the stub; only rendering it shows there is nothing inside."""
    from pipeline.ingestion.x_render import _article_shape, _render_article

    articles = [xg._normalize(n) for tid, n in frozen["graphql_raw"].items()
                if frozen["twitterapi_expected"][tid]["article"]]
    assert articles, "fixture lost its X Article"
    for a in articles:
        title, blocks = _article_shape(a["article"] or {})
        assert title, "article rendered with no title"
        assert len(blocks) > 20, (
            f"article body is a teaser: {len(blocks)} block(s). The timeline walk has lost "
            f"`withArticleRichContentState`.")
        md = _render_article(a["article"])
        assert len(md) > 5000, f"article markdown is {len(md)} chars — a preview, not a body"
