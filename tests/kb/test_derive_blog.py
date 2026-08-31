"""derive_blog + the blog_entity_id join key. Pure, offline."""
from __future__ import annotations

from pipeline.kb import derive


def test_blog_entity_id_is_canonical_host():
    assert derive.blog_entity_id("https://karpathy.github.io") == "blog:karpathy.github.io"
    assert derive.blog_entity_id("https://www.simonwillison.net/") == "blog:simonwillison.net"
    assert derive.blog_entity_id("http://stratechery.com/2024/x") == "blog:stratechery.com"  # host only
    assert derive.blog_entity_id("") == "blog:unknown"


def test_derive_blog_fields():
    article = {"url": "https://simonwillison.net/2024/01/agents",
               "title": "On\nAgents", "date": "2024-01-15", "content": "..."}
    m = derive.derive_blog(article, blog_url="https://simonwillison.net",
                           handle=None, author_name="Simon Willison")
    assert m["who_id"] == "blog:simonwillison.net"      # the blog HOME, not the per-post path
    assert m["who_name"] == "Simon Willison"
    assert m["who_site"] == "https://simonwillison.net"
    assert m["when_ts"] == "2024-01-15"
    assert m["when_precision"] == "day"
    assert m["about_entities"] == []
    # Mechanical description: newline flattened, structural fields only.
    assert "\n" not in m["description"]
    assert "Simon Willison" in m["description"] and "On Agents" in m["description"]


def test_derive_blog_precision_honest_on_missing_date():
    # A hub-harvested post with no extractable date → when_precision must NOT claim "day" (that
    # silently corrupts time-aware search — an undated atom would sort as if precisely dated).
    m = derive.derive_blog({"url": "https://b.com/p", "title": "T", "date": "", "content": "..."},
                           blog_url="https://b.com")
    assert m["when_ts"] == "" and m["when_precision"] == "unknown"
    # A real date → day precision, as before.
    m2 = derive.derive_blog({"url": "https://b.com/p", "title": "T", "date": "2019-04-01", "content": "."},
                            blog_url="https://b.com")
    assert m2["when_ts"] == "2019-04-01" and m2["when_precision"] == "day"


def test_derive_blog_name_falls_back_to_host_then_handle():
    article = {"url": "https://karpathy.github.io/p", "title": "T", "date": "", "content": "..."}
    # No author_name, no handle → name falls back to the canonical host.
    m = derive.derive_blog(article, blog_url="https://karpathy.github.io")
    assert m["who_name"] == "karpathy.github.io"
    # A handle (no name) → "@handle".
    m2 = derive.derive_blog(article, blog_url="https://karpathy.github.io", handle="@karpathy")
    assert m2["who_name"] == "@karpathy"
