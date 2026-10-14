"""Resume sitemap, feed and hub listings one HTTP page at a time, then read individual articles.

Which listed pages are posts is `post_pages`'s decision: the sitemap phase skips child sitemaps a
CMS files under a non-post type, the feed phase keeps the feed's post links, the select phase marks
each page by whether its address fits them, and a document off the feed is saved only if the page
labels itself a dated article."""
import time
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import feedparser
import requests

from pipeline.host_access import request
from pipeline.kb import ingest_blog, link_discovery, schema
from pipeline.kb.ingest_common import FETCH_ABSENT, FETCH_OK
from pipeline.platforms.url_canon import host_of
from pipeline.platforms.web import article as web_article
from pipeline.platforms.web import discover, links
from pipeline.timeparse import parse_ts
from pipeline.oracles.sources.archive import post_pages
from pipeline.oracles.sources.archive.adapter import Adapter, Page


SITEMAPS = ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml", "/post-sitemap.xml", "/sitemap-posts.xml")
FEEDS = ("/feed.xml", "/feed", "/atom.xml", "/rss.xml", "/rss", "/index.xml")


def transition(cursor, phase, urls):
    return {**cursor, "phase": phase, "urls": urls}


def entries(root, base):
    result = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "url":
            continue
        fields = {c.tag.rsplit("}", 1)[-1]: c.text or "" for c in node}
        url = urljoin(base, fields.get("loc", "").strip())
        path = urlsplit(url).path
        if path not in ("", "/", "/index.html") and not discover.is_nav_path(path):
            result.append({"url": url, "lastmod": fields.get("lastmod", "").strip()})
    return result


class Blog(Adapter):
    def initial(self, spec):
        base = spec["url"].rstrip("/")
        return {"phase": "sitemap", "urls": [base + p for p in SITEMAPS],
                "baseline": [], "links": [], "visited": [], "dates": {}, "indexes": []}

    def host(self, spec, kind, data):
        url = data["url"] if kind == "document" else next(iter(data.get("urls") or []), spec["url"])
        return urlsplit(url).hostname

    def page(self, spec, cursor, ctx):
        c = {**cursor, "urls": list(cursor["urls"]), "visited": list(cursor["visited"]),
             "baseline": list(cursor["baseline"]), "links": list(cursor["links"])}
        base, phase = spec["url"].rstrip("/"), c["phase"]
        if phase == "select":
            selected = c["baseline"] + link_discovery.select_hub_urls(
                base, c["baseline"], c["links"], c["indexes"], author_name=spec.get("author"))
            since = parse_ts(spec.get("since"))
            shapes = post_pages.learn(c.get("feed", []), base)
            docs = []
            for entry in selected:
                if since and entry.get("lastmod") and ingest_blog._lastmod_before(entry["lastmod"], since):
                    continue
                key = ingest_blog.canon_post_url(entry["url"])
                docs.append((key, {**entry, "feed_date": c["dates"].get(key, ""),
                                   "fits_feed": post_pages.fits(entry["url"], shapes) if shapes else None}))
            return Page(docs)
        if c["urls"]:
            url = c["urls"].pop(0)
            if url in c["visited"]:
                return Page(cursors=[c])
            response = request(requests.get, url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if phase == "sitemap" and c.get("indexed") and response.status_code in (404, 410):
                raise RuntimeError("a listed child sitemap disappeared")
            if response.status_code not in (404, 410):
                response.raise_for_status()
                if response.headers.get("cf-mitigated") == "challenge":
                    raise RuntimeError("archive page is a challenge")
                if phase == "sitemap":
                    try:
                        root = ElementTree.fromstring(response.content)
                    except ElementTree.ParseError:
                        root = None
                    if c.get("indexed") and (root is None or not root.tag.endswith(("urlset", "sitemapindex"))):
                        raise RuntimeError("a child sitemap was malformed")
                    if root is not None and root.tag.endswith("sitemapindex"):
                        children = [child for child in (urljoin(url, n.text.strip()) for n in root.iter()
                                                        if n.tag.rsplit("}", 1)[-1] == "loc" and n.text)
                                    if not post_pages.filed_as_nonpost(child)]
                        # A valid index replaces the alternative sitemap probes.
                        c["urls"] = list(dict.fromkeys(children + (
                            c["urls"] if c.get("indexed") else [])))
                        c["indexed"] = True
                    elif root is not None and root.tag.endswith("urlset"):
                        c["baseline"] += entries(root, url)
                        if c["baseline"] and not c.get("indexed"):
                            c["urls"] = []
                elif phase == "feed":
                    feed = feedparser.parse(response.content)
                    if feed.entries:
                        fallback = []
                        c["feed"] = [item.get("link") for item in feed.entries if item.get("link")]
                        for item in feed.entries:
                            link = item.get("link")
                            if not link:
                                continue
                            parsed = item.get("published_parsed") or item.get("updated_parsed")
                            day = time.strftime("%Y-%m-%d", parsed) if parsed else ""
                            fallback.append({"url": link, "lastmod": day})
                            if day:
                                c["dates"][ingest_blog.canon_post_url(link)] = day
                        if not c["baseline"]:
                            c["baseline"] = fallback
                        c["urls"] = []
                else:
                    c["links"] += [{**link, "via": url} for link in links.extract_links(response.text, url)]
            c["visited"].append(url)
        if c["urls"]:
            return Page(cursors=[c])
        if phase == "sitemap":
            c = transition(c, "feed", [base + p for p in FEEDS])
        elif phase == "feed":
            c = transition(c, "hub", [base] + [base + p for p in discover.HUB_INDEX_PATHS])
        elif phase == "hub":
            seeds = [e["url"] for e in c["baseline"]] + [e["url"] for e in c["links"]]
            c["indexes"] = link_discovery._pick_index_pages(
                seeds, host_of(base), link_discovery.hub_page_keys(base))
            c = transition(c, "index", c["indexes"])
        else:
            c = transition(c, "select", [])
        return Page(cursors=[c])

    def document(self, conn, spec, data, ctx, hold):
        url = data["url"]
        aid = ingest_blog.canon_post_url(url)
        seen = schema.load_hashes(conn, "blog")
        if aid in seen:
            return "present"
        article = web_article.fetch_article(url)
        verdict = ingest_blog._classify_article(article)
        if verdict == FETCH_ABSENT:
            return "skipped"
        if verdict != FETCH_OK:
            raise RuntimeError("article could not be read")
        if not post_pages.is_post(data.get("fits_feed"), article.get("declares") or {}):
            return "skipped"
        result = ingest_blog.build_article_atom(
            article, url=url, atom_id=aid, blog_url=spec["url"],
            author=spec.get("author") or spec["entity"], author_name=spec.get("author"),
            handle=spec.get("handle"), lastmod=data.get("lastmod", ""),
            feed_dates={aid: data["feed_date"]} if data.get("feed_date") else {},
            img_cache=ctx.cache["images"], seen=seen, entry_mode="oracle-footprint")
        if result["outcome"] != "ok":
            return result["outcome"]
        hold(result["atom"], result["md"])
        return aid
