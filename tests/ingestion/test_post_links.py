"""discovery.post_links — the sites an Oracle links from their own X posts.

The fixtures are real snapshots from the live store, in the shape `x_render` writes:
a quoted post renders first, with its own `## Links`, and the Oracle's own text follows. That
order is the only thing that tells a quoted author's link from the Oracle's, so it is pinned here.
"""
from __future__ import annotations

from pipeline.oracles.discovery.post_links import Candidate, candidate_key, candidates, links_in_post

QUOTE_POST = """---
source: x-profile
author: "@alexolegimas"
author_name: "Alex Imas"
url: https://x.com/alexolegimas/status/2106777719400779892
date: 2026-10-04
likes: 99
type: quote
tags: []
---

# Alex Imas — 2026-10-04

**Quoting** [@knowerofmarkets](https://x.com/knowerofmarkets/status/2106508662487601450):
> *knower*
>
> DeepMind Institute is doing some of the best longform writing on artificial intelligence right now, especially recommend this writeup from Davide and Alexander:
> https://institute.deepmind.com/essays/cheaters-and-whistleblowers-in-the-agent-swarm/

## Links

> **[Cheaters and whistleblowers in the agent swarm](https://institute.deepmind.com/essays/cheaters-and-whistleblowers-in-the-agent-swarm/)**
> In a 100-agent virtual math conference, a cascade of cheating emerged.

## Media

![photo](https://pbs.twimg.com/media/abc.jpg)

*Image:* a chart

More in my latest: https://aleximas.substack.com/p/agents-and-markets

## Links

> **[Agents and markets](https://aleximas.substack.com/p/agents-and-markets)**

---
*Profile extract · [Original post](https://x.com/alexolegimas/status/2106777719400779892)*
"""

BARE_POST = """---
source: x-profile
author: "@RhysSullivan"
url: https://x.com/RhysSullivan/status/1
type: post
---

# Rhys — 2026-09-01

shipped https://executor.sh and the repo https://github.com/RhysSullivan/executor, details at https://rhys.dev/posts/executor.

## Links

- [executor.sh](https://executor.sh)
- [github.com/RhysSullivan/executor](https://github.com/RhysSullivan/executor)

---
*Profile extract · [Original post](https://x.com/RhysSullivan/status/1)*
"""

SELF_QUOTE = """---
author: "@alexolegimas"
---

**Quoting** [@alexolegimas](https://x.com/alexolegimas/status/9):
> my new essay https://aleximas.substack.com/p/older

## Links

> **[Older](https://aleximas.substack.com/p/older)**

still true.
"""


# ── links_in_post ─────────────────────────────────────────────────────────────

def test_a_quoted_authors_links_are_not_the_oracles():
    links = links_in_post(QUOTE_POST)
    assert not any("institute.deepmind.com" in u for u in links)
    assert "https://aleximas.substack.com/p/agents-and-markets" in links


def test_both_link_forms_and_inline_text_links_are_read_once_each():
    links = links_in_post(BARE_POST)
    assert links.count("https://executor.sh") == 1
    assert "https://github.com/RhysSullivan/executor" in links
    assert "https://rhys.dev/posts/executor" in links          # trailing "." trimmed


def test_media_lines_are_not_links():
    assert not any("twimg.com" in u for u in links_in_post(QUOTE_POST))


def test_quoting_yourself_keeps_the_quoted_links():
    assert "https://aleximas.substack.com/p/older" in links_in_post(SELF_QUOTE)


def test_no_frontmatter_and_empty_input():
    assert links_in_post("") == []
    assert links_in_post("see https://a.example.dev/x") == ["https://a.example.dev/x"]


# ── candidate_key ─────────────────────────────────────────────────────────────

def test_candidate_key_table():
    cases = {
        "https://aleximas.substack.com/p/agents": "aleximas.substack.com",
        "https://open.substack.com/pub/aleximas/p/agents?r=1": "aleximas.substack.com",
        "https://open.substack.com/live-stream/123": None,
        "https://substack.com/@someone": None,
        "https://github.com/RhysSullivan/executor": "github.com/rhyssullivan",
        "https://github.com/features/actions": None,
        "https://rhys.dev/posts/executor": "rhys.dev",
        "https://www.dwarkesh.com/p/sarah": "dwarkesh.com",
        "https://medium.com/@karpathy/yes-you-should": "medium.com/@karpathy",
        "https://x.com/someone/status/1": None,
        "https://twitter.com/someone": None,
        "https://arxiv.org/abs/2401.00001": None,
        "https://www.youtube.com/watch?v=x": None,
        "http://localhost:3000/x": None,
        "https://api.example.com/v1": None,
        "http://192.168.1.4/x": None,
    }
    for url, want in cases.items():
        assert candidate_key(url) == want, url


# ── candidates ────────────────────────────────────────────────────────────────

def test_candidates_count_posts_not_links_and_sort_most_linked_first():
    posts = [
        ["https://aleximas.substack.com/p/a", "https://aleximas.substack.com/p/b", "https://nytimes.com/x"],
        ["https://open.substack.com/pub/aleximas/p/c"],
        ["https://nytimes.com/y", "https://x.com/a/status/1"],
        ["https://zeta.dev"],
    ]
    got = candidates(posts)
    assert got[0] == Candidate("aleximas.substack.com", "https://aleximas.substack.com", 2,
                               "https://aleximas.substack.com/p/a")
    assert [(c.key, c.post_count) for c in got] == [
        ("aleximas.substack.com", 2), ("nytimes.com", 2), ("zeta.dev", 1)]


def test_no_posts_no_candidates():
    assert candidates([]) == []
    assert candidates([[], ["https://x.com/a"]]) == []


def test_what_a_code_sample_leaves_behind_is_not_a_candidate():
    """Measured on Rhys's posts: `127`, `github` and `${server.port}` came out as candidates."""
    for url in ("http://127.0.0.1:3000/x", "https://github/x", "http://${server.port}/a"):
        assert candidate_key(url) is None
