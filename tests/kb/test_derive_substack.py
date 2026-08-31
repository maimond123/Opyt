"""derive_substack + the substack_entity_id join key (T6). Pure, offline."""
from __future__ import annotations

from pipeline.kb import derive


def test_entity_id_prefers_handle():
    assert derive.substack_entity_id("stratechery", "https://x.substack.com") == "substack:stratechery"
    assert derive.substack_entity_id("@Noahpinion", "") == "substack:noahpinion"


def test_entity_id_falls_back_to_subdomain_then_host():
    assert derive.substack_entity_id(None, "https://noahpinion.substack.com") == "substack:noahpinion"
    assert derive.substack_entity_id("", "https://www.astralcodexten.com") == "substack:www.astralcodexten.com"
    assert derive.substack_entity_id(None, "") == "substack:unknown"


def test_derive_substack_fields():
    rec = {
        "id": 42, "title": "The Bull Case\nfor Agents", "subtitle": "why",
        "post_date": "2026-06-26T11:45:17.702Z", "author_handle": "alice",
        "author_name": "Alice A", "publication_name": "Alice's Letter",
        "publication_url": "https://alice.substack.com", "wordcount": 1200,
        "audience": "everyone", "preview": "prev", "slug": "the-bull-case",
    }
    m = derive.derive_substack(rec)
    assert m["who_id"] == "substack:alice"
    assert m["who_name"] == "Alice A"
    assert m["who_site"] == "https://alice.substack.com"
    assert m["when_ts"] == "2026-06-26"
    assert m["when_precision"] == "day"
    assert m["about_entities"] == []
    # Mechanical description: newline flattened, structural fields only.
    assert "\n" not in m["description"]
    assert "Alice A" in m["description"] and "The Bull Case" in m["description"]
