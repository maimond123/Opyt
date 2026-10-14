"""
pipeline/oracles/sources/web/platform.py — the web outage platform of a URL: what an item's
failures are counted against.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pipeline.platforms.source_classify import domain_label

# The quota every web source charges: discovery, the archives and their refresh, linked documents.
# It lives here, beneath them all, so the milestones can name it without importing a source.
QUOTA = "web"


def platform_key(source_type: str, url: str) -> str:
    """The platform an item's failures are counted against for the outage rule: its source type,
    or for a blog its registrable domain (`alice.wordpress.com` → `blog:wordpress.com`), by the
    heuristic `source_classify.domain_label` uses, `.co.uk` included."""
    if source_type != "blog":
        return source_type

    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    host = host.removeprefix("www.")
    parts = host.split(".")
    label = domain_label(host)
    if label and label in parts:
        host = ".".join(parts[len(parts) - 1 - parts[::-1].index(label):])
    return f"blog:{host}"


def own_site(source_type: str, url: str) -> bool:
    """Is this source served from its owner's own domain though its platform is another's: a
    Substack that is not on `substack.com`. Such a site can die alone, so its unanswered
    requests are its own failure (`Item.own_site`)."""
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return source_type == "substack" and not (host == "substack.com"
                                              or host.endswith(".substack.com"))
