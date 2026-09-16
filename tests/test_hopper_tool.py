"""
tests/test_hopper_tool.py

`hopper`'s TOOL DESCRIPTION, which is the only part of this surface a host reads before deciding
to act. The engine is covered in `tests/kb/test_hopper.py`; what is pinned here is the one thing
the engine cannot enforce — that the description tells the host to route by the same facts
`link_router` routes by.

The description tells the host to OFFER a deposit unprompted after a web search, and it names the
hosts that clear that bar. Naming them is a REPORT of what `classify_link` sniffs, and a report
that can disagree with the code is the bug — the same rule `_effective_x_since` follows by
calling the clamp instead of restating it. A docstring cannot call anything, so the agreement is
asserted here instead.
"""

import pytest

from mcp_server import hopper_tools
from pipeline.kb import link_router


class _MCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def doc():
    m = _MCP()
    hopper_tools.register_hopper_tools(m)
    return m.tools["hopper"].__doc__


@pytest.mark.parametrize("host", link_router._PAPER_HOSTS)
def test_every_sniffed_paper_host_is_named_in_the_offer_bar(doc, host):
    """A paper host the router sniffs but the description omits is a link the host model never
    offers to keep — a silent miss, since nothing errors and the user is simply not asked."""
    assert host in doc


def _bar(doc: str) -> str:
    """Just the offer bar. Scoped deliberately: the rest of the docstring names hosts for other
    reasons — `kind_hint`'s note mentions a `*.substack.com` glob to explain a CUSTOM-domain
    Substack, which is the one case the sniff cannot see — and a whole-docstring scan reads that
    as a claim about routing."""
    head = "OFFER IT UNPROMPTED"
    tail = "Those four are the bar"
    assert head in doc and tail in doc, "the offer bar's own delimiters moved"
    return doc[doc.index(head):doc.index(tail)]


@pytest.mark.parametrize("url", ["https://huggingface.co/papers/2005.14165",
                                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/",
                                 "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8371605/",
                                 "https://europepmc.org/article/MED/34265844",
                                 "https://europepmc.org/article/PMC/PMC8371605",
                                 "https://europepmc.org/article/PPR/PPR217527",
                                 "https://openalex.org/W2741809807",
                                 # Moved off the bare-host list 2026-09-16 — see
                                 # `link_router._PAPER_PATH_RES`. Each host has a real non-paper
                                 # section (an author page, a community, an about page) that a
                                 # bare entry routed to the paper adapter.
                                 "https://semanticscholar.org/paper/CorpusID:13756489",
                                 "https://zenodo.org/records/3509134",
                                 "https://alphaxiv.org/abs/2404.16130",
                                 "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3482150"])
def test_every_path_scoped_paper_form_the_bar_promises_actually_routes(doc, url):
    """The host-level test above cannot see these: they are hosts that are only PARTLY papers, so
    the bar names them WITH a path and the promise is only kept for that path. Same silent-miss
    risk in both directions, checked against the router rather than the host list."""
    from urllib.parse import urlparse
    assert link_router.classify_link(url) == "paper"
    stem = urlparse(url).netloc + urlparse(url).path.rsplit("/", 2)[0]
    assert stem in doc, f"{stem!r} routes but the bar never offers it"


def test_the_offer_bar_names_no_host_the_router_does_not_sniff(doc):
    """The other direction, and the worse one. A host named in the bar but absent from
    `classify_link` routes to the `article` FALLBACK instead, so the model offers a deposit under
    a promise the router does not keep — and a wrong route is silent: a paper filed as a blog
    post never errors, it just sits wrong forever."""
    named = {"github.com", "substack.com", "x.com", "twitter.com", *link_router._PAPER_HOSTS}
    for token in _bar(doc).replace("**", " ").replace("`", " ").split():
        token = token.strip(".,;:()/").lower()
        if "/" not in token and any(token.endswith(s) for s in (".com", ".org", ".net", ".gov")):
            assert token in named, f"{token!r} is offered but `classify_link` ignores it"


def test_the_bar_excludes_the_article_fallback(doc):
    """`classify_reference` falls back to "article" for ANY http url, so a bar phrased as
    "routable" would admit every web-search result and the offer would fire on all of them. An
    offer after every search is noise the user learns to skip, which is worse than not asking."""
    assert link_router.classify_reference("https://example.com/news/story")[0] == "article"
    assert "Do NOT offer on an ordinary article" in doc


def test_the_user_facing_offer_never_says_hopper(doc):
    """The tool's name means nothing to a new user, and the whole reason the offer exists is that
    they have never heard of a deposit surface. The example wording is what the host reads back."""
    example = doc.split('"That first repo')[1].split('"')[0]
    assert "hopper" not in example.lower()
    assert 'NEVER say "hopper" to the user' in doc
