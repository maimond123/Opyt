"""
pipeline/oracles/discovery/shaping.py — what discovery found, as sources: the source and profile
shapes, URL normalizing and classifying, and dedupe. Plain functions of values.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from urllib.parse import urlparse

from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import parse_url


@dataclass
class DiscoveredSource:
    source_type: str                 # blog, substack, github, scholar, orcid
    url: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DiscoveredProfile:
    username: str
    display_name: str = ""
    bio: str = ""
    website: str = ""
    sources: list = field(default_factory=list)
    discovered_at: str = ""


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    parsed = parse_url(url)
    if parsed is None:
        return ""
    url = parsed._replace(scheme="https").geturl().rstrip("/")
    # Fix double slashes in path (but not after protocol)
    parsed = urlparse(url)
    if "//" in parsed.path:
        clean_path = parsed.path.replace("//", "/")
        url = parsed._replace(path=clean_path).geturl()
    return url


def classify_url(url: str) -> Optional[str]:
    """Classify source links, including artifacts that identify a known account.

    Blog anchors to known account platforms remain ownership declarations by policy.
    Platform roots have neither a profile nor an account handle and are excluded.
    """
    profile = classify_source(normalize_url(url))
    return profile.type if profile and (profile.is_profile or profile.handle) else None


def dedupe_sources(sources: list) -> list:
    """Deduplicate sources by (source_type, normalized URL)."""
    seen = set()
    deduped = []
    for s in sources:
        key = (s.source_type, normalize_url(s.url))
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped
