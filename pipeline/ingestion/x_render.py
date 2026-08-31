"""
pipeline/ingestion/x_render.py — the X tweet RENDERER. No network, no provider.

One normalized tweet (or a stitched thread) → the markdown an atom's body is made of: URL
expansion, quoted tweets, link cards, media, and X-Article bodies. Every function here is PURE —
give it a tweet dict, get a string back.

It was `x_twitterapi.py` until 2026-08-30, when twitterapi.io was removed
(docs/plans/2026-08-30-cut-x-ingestion-over-to-internal-graphql.md). Every X read now runs on the
user's own browser session through x.com's internal GraphQL API, at $0, with no third party seeing
which accounts are read. What survived the deletion is the ~450 lines of rendering that were never
provider-specific — so the module was renamed for what it does, because a module named after a
provider it no longer contains reads to the next person as though that vendor is still live.

⚠️ Do not add a fetch here. The transport lives in `pipeline/ingestion/x_graphql_core.py`; pulling
an Oracle's timeline INTO the store is `pipeline/kb/ingest_x_footprint.py`'s job. A renderer that
can also fetch is how this module grew a network layer the first time.
"""

from datetime import datetime, timezone

from pipeline.ingestion.utils import log


def _parse_twitter_date(created_at: str) -> datetime | None:
    """Parse Twitter date string: 'Tue Dec 10 07:00:30 +0000 2024'"""
    try:
        return datetime.strptime(
            created_at, "%a %b %d %H:%M:%S +0000 %Y"
        ).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ── URL expansion ─────────────────────────────────────────────────────────────

def _expand_urls(text: str, entities: dict | None) -> str:
    """Replace t.co short URLs in text with their expanded equivalents."""
    if not entities:
        return text
    for url_obj in entities.get("urls", []):
        short    = url_obj.get("url", "")
        expanded = url_obj.get("expanded_url", "") or url_obj.get("display_url", "")
        if short and expanded and short in text:
            text = text.replace(short, expanded)
    return text


# ── Article detection and rendering ──────────────────────────────────────────

def _article_tweet_id(tweet: dict) -> str | None:
    """
    Return the tweet ID to use for article fetching, or None if this tweet
    doesn't contain an X Article.

    Detection: look for x.com/i/article or /articles/ in entities.urls.
    The article endpoint uses the outer tweet's ID, not the article URL itself.
    """
    for url_obj in (tweet.get("entities") or {}).get("urls", []):
        expanded = url_obj.get("expanded_url", "")
        if "/i/article" in expanded or "/articles/" in expanded:
            return str(tweet.get("id", ""))
    return None


# Runaway guard against a hostile/malformed article body, not a quality truncation — it fires far
# above any real article (~125K tokens).
_ARTICLE_MAX_CHARS = 500_000


def _article_shape(article: dict) -> tuple[str, list]:
    """The raw `article` node → (title, Draft.js blocks). Unknown shape → ("", []), which renders
    nothing rather than crashing.

    ⚠️ AN EMPTY BLOCK LIST IS THE TEASER, AND IT LOOKS LIKE A NORMAL RESPONSE. X ships this node
    two ways: with `content_state.blocks` when the request asked for `withArticleRichContentState`,
    and without it otherwise — cover image, title and `preview_text` only, about 1.1 KB. The
    request still returns 200 and the node is still truthy, so `bool(article)` cannot tell them
    apart. Callers that need to know whether a body arrived must ask THIS function and check the
    blocks; `ingest_x_footprint` does, and reports `article_incomplete` when they are empty.

    `cover_media` and `preview_text` are on the node and deliberately not returned. Neither has
    ever been rendered — the shape that carried them was twitterapi.io's `/twitter/article`
    response, deleted with the provider on 2026-08-30 — and adding them now would change every
    article atom's snapshot hash and re-embed the lot for a cover image."""
    res = ((article.get("article_results") or {}).get("result") or {})
    cs = res.get("content_state") or {}
    return res.get("title", ""), (cs.get("blocks") or [])


def _render_article(article: dict) -> str:
    """Render an X-Article's content blocks to markdown.

    The full body is the point — it becomes the atom's chunked, searchable surface — so we render all
    of it, bounded only by the runaway guard."""
    title, blocks = _article_shape(article)
    parts = []
    if title:
        parts.append(f"## {title}\n")

    total = 0
    for block in blocks:
        btype = block.get("type", "unstyled")
        text  = block.get("text", "")
        url   = block.get("url", "")

        if btype == "divider":
            piece = "\n---\n"
        elif btype == "header-one":
            piece = f"\n# {text}\n"
        elif btype == "header-two":
            piece = f"\n## {text}\n"
        elif btype in ("header-three", "header-four"):
            piece = f"\n### {text}\n"
        elif btype == "unordered-list-item":
            piece = f"- {text}"
        elif btype == "ordered-list-item":
            piece = f"1. {text}"
        elif btype == "blockquote":
            piece = f"> {text}"
        elif btype == "code-block":
            piece = f"```\n{text}\n```"
        elif btype == "image":
            piece = f"![image]({url})\n"
        elif btype == "gif":
            piece = f"![gif]({block.get('previewUrl', url)})\n"
        elif btype == "markdown":
            piece = text
        elif btype == "atomic":       # Draft.js media block — body lives in entityMap, skip in v1
            continue
        else:                          # unstyled / fallback
            if not text:
                continue
            piece = text

        total += len(piece)
        if total > _ARTICLE_MAX_CHARS:
            log(f"  [article] body exceeded {_ARTICLE_MAX_CHARS} chars — truncating as a runaway "
                f"guard (likely a malformed/hostile payload).")
            break
        parts.append(piece)

    return "\n".join(parts) + "\n"


# ── Markdown rendering helpers ────────────────────────────────────────────────

def _render_quoted_tweet(qt: dict) -> str:
    """Render a nested quoted_tweet: author + text, plus its own link card, media, and X-Article
    body — a quote's meaning often lives in what it quotes, so a text-only render would drop that
    payload. Reuses the same three helpers as the root tweet. One level only (no quote-of-quote
    recursion)."""
    author   = qt.get("author", {})
    username = author.get("userName", "unknown")
    name     = author.get("name", "Unknown")
    qt_url   = (
        qt.get("url") or qt.get("twitterUrl") or
        f"https://x.com/{username}/status/{qt.get('id', '')}"
    )
    qt_text  = _expand_urls(qt.get("text", ""), qt.get("entities"))
    lines    = qt_text.strip().split("\n")
    quoted   = "\n".join(f"> {line}" for line in lines)

    out = (
        f"**Quoting** [@{username}]({qt_url}):\n"
        f"> *{name}*\n"
        f">\n"
        f"{quoted}\n"
    )
    # The quoted tweet's own card / media / article. Each helper returns "" when its payload is
    # absent (fail-safe), so a plain quoted text-post adds nothing and this stays a no-op.
    card_md = _render_link_cards(qt)
    if card_md:
        out += "\n" + card_md
    media_md = _render_media(qt)
    if media_md:
        out += "\n" + media_md
    if qt.get("article"):
        out += "\n" + _render_article(qt["article"]) + "\n"
    return out


def _parse_card(card: dict | None, entities: dict | None) -> tuple[str, str, str]:
    """
    Extract (title, description, url) from a twitterapi.io card object.

    twitterapi.io puts card data in card.binding_values — a list of
    {key: str, value: {string_value|image_value|...}} objects.
    The relevant keys are 'title', 'description', and 'vanity_url'.
    The URL comes from entities.urls[0].expanded_url.
    """
    if not card or not isinstance(card, dict):
        return "", "", ""

    bv: dict[str, str] = {}
    for item in card.get("binding_values") or []:
        key = item.get("key", "")
        val = item.get("value") or {}
        if "string_value" in val:
            bv[key] = val["string_value"]

    title = bv.get("title", "")
    desc  = bv.get("description", "")

    # Get expanded URL from entities.urls (the t.co maps here)
    url = ""
    for u in (entities or {}).get("urls", []):
        expanded = u.get("expanded_url", "")
        if expanded and "x.com" not in expanded and "twitter.com" not in expanded:
            url = expanded
            break

    return title, desc, url


def _render_link_cards(tweet: dict) -> str:
    """
    Render external link card from card.binding_values.
    Falls back to bare expanded URLs from entities.urls if no card data.
    """
    entities = tweet.get("entities")
    card     = tweet.get("card")

    title, desc, url = _parse_card(card, entities)

    if title and url:
        block = f"> **[{title}]({url})**"
        if desc:
            block += f"\n> {desc}"
        return "## Links\n\n" + block + "\n\n"

    # Fallback: bare external URLs (no card metadata available)
    bare_links = []
    for u in (entities or {}).get("urls", []):
        expanded = u.get("expanded_url", "")
        display  = u.get("display_url", expanded)
        if not expanded:
            continue
        if "x.com" in expanded or "twitter.com" in expanded:
            continue
        bare_links.append(f"- [{display}]({expanded})")
    if bare_links:
        return "## Links\n\n" + "\n".join(bare_links) + "\n\n"

    return ""


def _render_media(tweet: dict) -> str:
    """
    Render media attachments.

    Input is the twitterapi.io shape: extendedEntities.media, each item carrying
    type (photo/video/animated_gif), media_url_https, video_info.variants. Free-engine
    (GraphQL) tweets must be mapped to this shape by x_graphql._normalize first.

    Never silently drops: if media is present but unrenderable — hiding under a key we
    don't read (snake_case drift), an unknown type, or a missing url/variant — it emits a
    [media-drop] log line instead of an empty string that's indistinguishable from "no
    media". That distinction is the whole point; a bare "" is how the last drop hid.
    """
    # Primary: extendedEntities.media (twitterapi.io shape). Fallback: top-level media.
    ext = tweet.get("extendedEntities") or {}
    media_list = ext.get("media") or []
    if not media_list:
        media_list = tweet.get("media") or []

    if not media_list:
        # Empty under the keys we render from. Before declaring "no media", check whether
        # the tweet carries media under a key we DIDN'T read — the historic snake_case
        # mismatch, or a future producer/shape drift. That's a drop, not an absence.
        shadow = ((tweet.get("extended_entities") or {}).get("media")
                  or (tweet.get("entities") or {}).get("media"))
        if shadow:
            log(f"  [media-drop] {tweet.get('id', '?')}: {len(shadow)} media item(s) present "
                f"under an unread key — check the _normalize → _render_media field mapping.")
        return ""

    parts = ["## Media\n"]
    rendered = 0
    for m in media_list:
        if not isinstance(m, dict):
            continue
        mtype = m.get("type", "photo")
        if mtype == "photo":
            url = m.get("media_url_https") or m.get("url", "")
            if url:
                parts.append(f"![photo]({url})\n")
                # A VLM description (attached upstream by the ingest layer) is what makes an
                # image-borne post searchable — a bare CDN URL carries no meaning into the chunk.
                desc = m.get("description")
                if desc:
                    parts.append(f"*Image:* {desc}\n")
                rendered += 1
        elif mtype in ("video", "animated_gif"):
            # Use highest-bitrate mp4 variant if available
            video_info = m.get("video_info") or {}
            variants = [
                v for v in video_info.get("variants", [])
                if v.get("content_type") == "video/mp4"
            ]
            if variants:
                best = max(variants, key=lambda v: v.get("bitrate", 0))
                thumb = m.get("media_url_https", "")
                vid_url = best["url"]
                if thumb:
                    parts.append(f"[![video]({thumb})]({vid_url})\n")
                else:
                    parts.append(f"[Video]({vid_url})\n")
                rendered += 1
            else:
                # Fallback to thumbnail
                thumb = m.get("media_url_https", "")
                if thumb:
                    parts.append(f"![{mtype}]({thumb})\n")
                    rendered += 1

    # We were handed N items but rendered fewer — an unknown type or a missing
    # url/variant swallowed the difference. Surface it rather than drop it silently.
    if rendered < len(media_list):
        log(f"  [media-drop] {tweet.get('id', '?')}: rendered {rendered}/{len(media_list)} "
            f"media item(s) — {len(media_list) - rendered} unrenderable "
            f"(unknown type / missing url or variant).")

    return "\n".join(parts) + "\n" if rendered else ""


# ── Core renderer ─────────────────────────────────────────────────────────────

def tweet_to_markdown(
    root_tweet:    dict,
    article:       dict | None     = None,
    thread_tweets: list[dict] | None = None,
    source:        str = "x-profile",
    footer_label:  str = "Profile extract",
) -> str:
    """
    Render a tweet (or thread) to the pipeline markdown format.

    root_tweet:    The canonical tweet for this note (earliest in thread, or solo).
    article:       Pre-fetched article dict from /twitter/article, if applicable.
    thread_tweets: Full ordered list of tweets in the thread (root → last reply).
                   Only supplied when len > 1 (i.e., this is actually a thread).
    source:        Provenance stamped into frontmatter `source:` — "x-profile" (the default,
                   a credible-people crawl), "x-bookmark", or "x-probe". Frontmatter is
                   stripped before chunking (`chunk.strip_frontmatter`), so this is never
                   indexed: it is provenance for a human opening the archived snapshot.
    footer_label:  Leading word of the footer line ("Profile extract" / "Bookmarked" /
                   "Candidate probe"), matching whichever `source` the caller passed.
    """
    author   = root_tweet.get("author", {})
    username = author.get("userName", "unknown")
    name     = author.get("name", "Unknown")
    tweet_id = str(root_tweet.get("id", ""))
    url      = (
        root_tweet.get("url") or root_tweet.get("twitterUrl") or
        f"https://x.com/{username}/status/{tweet_id}"
    )
    created_at = _parse_twitter_date(root_tweet.get("createdAt", ""))
    date_str   = created_at.strftime("%Y-%m-%d") if created_at else "unknown"
    is_quote   = root_tweet.get("isQuote", False)
    quoted_obj = root_tweet.get("quoted_tweet")

    # Post type precedence: article > thread > quote > post. Recognize a quote by the OBJECT too —
    # the profile fetch nulls isQuote (same reason the quote render gates on quoted_obj below).
    if article:
        post_type = "article"
    elif thread_tweets and len(thread_tweets) > 1:
        post_type = "thread"
    elif is_quote or (quoted_obj and isinstance(quoted_obj, dict)):
        post_type = "quote"
    else:
        post_type = "post"

    frontmatter = (
        f"---\n"
        f"source: {source}\n"
        f'author: "@{username}"\n'
        f'author_name: "{name}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"likes: {root_tweet.get('likeCount', 0)}\n"
        f"type: {post_type}\n"
        f"tags: []\n"
        f"---\n\n"
    )

    body = f"# {name} — {date_str}\n\n"

    # Gate on the PRESENCE of the quoted object, not the isQuote flag: the profile-fetch endpoint
    # can leave isQuote=None on genuine quote tweets while still populating quoted_tweet.
    if quoted_obj and isinstance(quoted_obj, dict):
        body += _render_quoted_tweet(quoted_obj) + "\n"
    elif is_quote:
        # Flag says quote but no object — the genuine rare gap. Mark it visibly (never a silent drop).
        log(f"  [warn] isQuote set but no quoted_tweet object on {tweet_id}")
        body += f"*[quoted tweet unavailable]*\n\n"

    # Thread body
    if thread_tweets and len(thread_tweets) > 1:
        body += f"> **Thread** · [{len(thread_tweets)} posts]({url})\n\n"
        for i, t in enumerate(thread_tweets, 1):
            t_text = _expand_urls(t.get("text", ""), t.get("entities"))
            body  += f"**[{i}/{len(thread_tweets)}]** {t_text}\n\n"
            # Per-tweet card (each thread post can have its own link card)
            t_card = _render_link_cards(t)
            if t_card:
                body += t_card
            # Per-tweet media
            t_media = _render_media(t)
            if t_media:
                body += t_media
            if i < len(thread_tweets):
                body += "---\n\n"
    else:
        text  = _expand_urls(root_tweet.get("text", ""), root_tweet.get("entities"))
        body += f"{text}\n\n"

    # X Article content
    if article:
        body += _render_article(article) + "\n"

    # Link cards and media — only for standalone posts/quotes.
    # Threads render these inline per-post in the loop above.
    if not (thread_tweets and len(thread_tweets) > 1):
        link_md = _render_link_cards(root_tweet)
        if link_md:
            body += link_md

    media_md = _render_media(root_tweet) if not (thread_tweets and len(thread_tweets) > 1) else ""
    if media_md:
        body += media_md

    body += f"---\n*{footer_label} · [Original post]({url})*\n"
    return frontmatter + body



def _dedupe_tweets(tweets: list[dict]) -> list[dict]:
    """Unique by id, newest first. Centralized here since advanced_search repeats tweets across
    pages and the probe page overlaps the newest shard by construction. Snowflake ids sort by
    time, so an id sort gives a deterministic 'Latest' ordering."""
    seen: set[str] = set()
    out: list[dict] = []
    for t in tweets:
        tid = str(t.get("id", ""))
        if tid and tid not in seen:
            seen.add(tid)
            out.append(t)
    out.sort(key=lambda t: int(t.get("id", 0) or 0), reverse=True)
    return out


# ── Thread stitching ──────────────────────────────────────────────────────────

def _stitch_threads(tweets: list[dict]) -> dict[str, list[dict]]:
    """
    Group tweets by conversationId → sorted list (chronological).
    Single-tweet conversations stay as single-item lists.
    """
    groups: dict[str, list[dict]] = {}
    for t in tweets:
        # conversationId "may be empty" per docs — fall back to tweet's own id
        conv_id = str(t.get("conversationId") or t.get("id", ""))
        groups.setdefault(conv_id, []).append(t)

    # Sort by tweet ID — snowflake IDs are ms-precise and monotonically increasing,
    # more reliable than createdAt which is only second-precision
    for conv_id in groups:
        groups[conv_id].sort(key=lambda t: int(t.get("id", 0) or 0))

    return groups
