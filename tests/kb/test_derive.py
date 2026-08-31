"""derive — lean, deterministic metadata from source structure (no LLM/NER)."""
from __future__ import annotations

from pipeline.kb.derive import derive_github, derive_x, slugify


def test_slugify_normalizes_to_kebab():
    assert slugify("AI Agents") == "ai-agents"
    assert slugify("#AIAgents") == "aiagents"
    assert slugify("ai_agents") == "ai-agents"
    assert slugify("  Rollups & ZK  ") == "rollups-zk"
    assert slugify("") == ""


def test_derive_x_pulls_author_topics_mentions_and_mechanical_desc():
    norm = {
        "id": "123",
        "author": {"userName": "karpathy", "name": "Andrej", "id": "33836629"},
        "text": "great thread on agents\nand tools",
        "createdAt": "Wed Oct 09 12:00:00 +0000 2024",
        "entities": {
            "hashtags": [{"text": "AIAgents"}, {"text": "LLMs"}],
            "user_mentions": [{"screen_name": "ylecun"}],
        },
    }
    meta = derive_x(norm)
    assert meta["who_id"] == "x:user:33836629"      # author entity, not the saver
    assert meta["when_precision"] == "day"
    # Hashtags are the AUTHOR's labels → `source_tags` (§6 of the capture audit).
    assert meta["source_tags"] == ["aiagents", "llms"]
    assert meta["about_entities"] == ["x:@ylecun"]
    assert meta["description"].startswith("@karpathy (Andrej) · great thread on agents")
    assert "\n" not in meta["description"]           # newlines flattened for a one-line card


def test_derive_x_missing_author_degrades_not_crashes():
    meta = derive_x({"id": "1", "text": "x"})
    assert meta["who_id"] == "x:user:unknown"


def test_derive_github_uses_owner_and_flags_push_precision():
    repo = {
        "name": "nanoGPT",
        "owner": {"login": "karpathy"},
        "language": "Python",
        "stargazers_count": 42000,
        "description": "The simplest, fastest repository for training GPTs.",
        "topics": ["gpt", "deep-learning"],
        "pushed_at": "2024-05-01T10:00:00Z",
    }
    meta = derive_github(repo)
    assert meta["who_id"] == "github:karpathy"
    assert meta["when_ts"] == "2024-05-01"
    assert meta["when_precision"] == "push"          # NOT a publish date
    assert meta["source_tags"] == ["gpt", "deep-learning"]       # the repo's OWN topics (§6)
    assert meta["about_entities"] == ["lang-python"]
    assert meta["description"].startswith("karpathy/nanoGPT · Python ·")
    assert meta["description"].endswith("★42000")
