"""
pipeline/frontier/adapters.py — the registry: which sources Frontier stage 2 searches, built once
per pass. The contract they meet is `source_search.Adapter`; each source's mapping to
`Candidate`s lives in its own module.
"""

from __future__ import annotations

from pipeline.frontier.arxiv_candidates import ArxivAdapter
from pipeline.frontier.github_candidates import GitHubAdapter
from pipeline.frontier.openalex_candidates import OpenAlexAdapter
from pipeline.frontier.source_search import Adapter
from pipeline.frontier.substack_candidates import SubstackAdapter
from pipeline.frontier.x_candidates import XAdapter


# A `target_sources` value with no entry here is counted as `no_adapter`, never silently dropped.
# A watch names only sources in this registry, so that is a source removed since.
def adapters() -> dict[str, Adapter]:
    return {a.slug: a for a in (ArxivAdapter(), GitHubAdapter(), OpenAlexAdapter(), XAdapter(),
                                SubstackAdapter())}
