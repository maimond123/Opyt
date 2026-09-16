"""
mcp_server/hopper_tools.py — the `hopper` tool: ONE deposit surface for the atom KB.

Registered with `register_hopper_tools(mcp)` like every other tool group (`atoms_tools`,
`oracle_tools`, `sitting_tools`, `frontier_tools`). Thin delegate by design: everything real
lives in `pipeline/kb/hopper.py` (importable and testable with no MCP server), and routing in
`pipeline/kb/link_router.py`, shared with the X footprint puller.

Hopper replaced `save_paper`/`save_repo`, the last manual "keep this artifact" tools and the
last vault writers on the tool surface, and must not reintroduce a vault writer: atoms only.
"""
from __future__ import annotations


def register_hopper_tools(mcp) -> None:

    @mcp.tool()
    def hopper(reference: str, confirm: bool = False, kind_hint: str | None = None) -> dict:
        """SAVE something into the knowledge base — a link the user wants to KEEP. Hand it a
        research paper, a GitHub repo, a Substack post, a plain article or blog post, or a single
        X post; it works out which and routes it to the right ingester. This is the ONLY manual
        "keep this" path; there is no separate save-a-paper or save-a-repo tool.

        NOT "any URL", which this line claimed until 2026-09-15 and the bar ten lines below has
        always contradicted. The overclaim cost two different things: video and podcast links fell
        through to the article ingester and came back "no substantive units", and the same words
        on the marketing site promised a reach the router never had. One page, one atom — a whole
        SITE or a whole PERSON is `oracle`'s job, not this one.

        Reach for this whenever the user says keep / save / add / remember / "put this in OPYT"
        about a link — including a link YOU just surfaced from a web search. It persists a link
        into the trusted corpus so the knowledge-base search tool can route to it later.

        OFFER IT UNPROMPTED after a web search, when the results include a link OPYT ingests
        natively. Exactly four kinds, and the test is the URL's host — a fact, not a judgement:
          • **github.com** — a repo.
          • a paper host, where every page is a paper — **arxiv.org**, **doi.org**,
            **biorxiv.org**, **medrxiv.org**, **openreview.net**, **pubmed.ncbi.nlm.nih.gov**,
            **aclanthology.org** — or the paper SECTION of a host that is only partly papers,
            where the path is part of the promise: **huggingface.co/papers/…**,
            **semanticscholar.org/paper/…**, **zenodo.org/records/…**,
            **alphaxiv.org/abs/…**, **papers.ssrn.com/…?abstract_id=…**, PubMed Central
            (**pmc.ncbi.nlm.nih.gov/articles/…**, **www.ncbi.nlm.nih.gov/pmc/articles/…**),
            **europepmc.org/article/{MED|PMC|PPR}/…** (an article, a PMC copy, a preprint), and
            **openalex.org/W…**. A model page, a gene record, an AUTHOR page or a community page
            on those same hosts is NOT a paper. Or any direct **.pdf**, or **any URL with a DOI in
            its path** (`…/10.1021/jacs.4c01234`), which is how a publisher link — ACS, Wiley,
            Springer, ACM — announces a paper without being on that list.
          • **substack.com** — a post.
          • **x.com** / **twitter.com** — a single status.
        Those four are the bar. Do NOT offer on an ordinary article, news page or company site:
        every search returns those, an offer after every search is noise the user learns to skip,
        and a person worth keeping belongs on the roster (`oracle`), not saved one post at a time.

        Name the ONE or TWO results actually worth keeping, not the whole page. Offer in plain
        words and NEVER say "hopper" to the user — the tool's name means nothing to them:

            "That first repo looks like what you need. Want me to save it to your knowledge
             base? It gets read and indexed, so you can search it later, and it tells OPYT
             more about what to watch for you."

        Then preview only the link they pick — a preview is free and writes nothing, and it is
        what reports `already_present`, so do not preview all ten to find out.

        Why this is YOUR job and not a setting: a new user has never heard of a deposit surface
        and will not ask for one. The moment a link is in front of them is the only moment the
        offer means anything.

        TWO-PHASE:
          • confirm=False (the default) → a PREVIEW. It reports which adapter the reference routes
            to, WHY it routed there, the atom id, whether the KB already has it, and what a confirm
            would do. It NEVER writes. It fetches nothing for a paper, repo or Substack post — you
            can already read those yourself, so describe them to the user in your own words
            alongside the routing. A link that would fall through to `article` costs ONE bounded
            read of the page head, to see whether the page declares itself a paper (a
            `citation_doi` tag) — that is how a nature.com URL routes to `paper` rather than
            being filed as a blog post.
            The X exception: for an x.com status link the preview reads the post and returns a
            `description`. You cannot fetch x.com, and `x:2086520133909168332` is
            unverifiable by a human — so read that `description` back before confirming; it is the
            only way the user can catch a wrong link. If `unreadable` comes back instead, the post
            is deleted / protected / keyless: say so and do NOT confirm.
            Paywalls are your job, not the preview's. This tool stores PUBLIC content only —
            a paywalled Substack post is skipped by the adapter and comes back "failed". It reads
            the same cookie-less public endpoint you do, so it cannot see past a wall you hit
            either. If the page you read was a subscriber teaser, say so BEFORE confirming instead
            of making a round trip to be told no.
          • confirm=True → runs the ingest: an embedding always, plus a content gate for articles
            and thread resolution for an X post. Show the preview first — a wrong
            route is SILENT (a paper filed as a blog post never errors, it just sits wrong).

        Skip straight to confirm=True only when the user has already said "yes, save it" about
        THAT specific link.

        `already_present: true` in a preview means a confirm is a no-op — say so and do not rerun it.
        Repeat calls are idempotent: an unchanged item is never re-fetched or re-embedded.

        What it will not do — do not ask it to, and do not work around it:
          • It never adds a PERSON to the tracked roster. Saving someone's article does not start
            following them. `add_oracle` is the only way in, and it asks the user first.
          • It never writes vault notes. Atoms only.
          • It never guesses. A reference that is not a URL comes back `unroutable` with nothing
            written, rather than being filed somewhere plausible.

        Every saved atom is stamped `entry_mode='user-saved'` — the same mark an X bookmark gets,
        because both mean the user personally saved it. That is load-bearing downstream: hand-saved
        items steer the Frontier's standing research queries.

        Args:
            reference: the URL to save — an article, paper (arXiv / DOI / .pdf), GitHub repo,
                Substack post, or x.com status link.
            confirm: False (default) = preview only, no fetch and no writes; True = run the ingest.
            kind_hint: OPTIONAL, and only consulted when the URL host matches nothing known — a
                recognized host always wins, because the host is a fact and your read is not.
                In practice there is ONE case worth passing it for: pass "substack" when you can
                see the page is a Substack post on a custom domain (a `/p/{slug}` path, a
                subscribe widget) rather than a `*.substack.com` URL. Nothing can detect that
                without fetching, and it matters — routed as a plain article the post gets a
                different atom id and will never dedupe against the same post saved from a
                bookmark. The other values ("paper", "github", "x") cannot override anything:
                those adapters check the host themselves and refuse a URL that is not theirs.

        Returns {status, kind, atom_id, entry_mode, …}. On a preview, status="preview" plus
        `already_present` and `why`. On a confirm, status is one of "saved",
        "already_present", "rejected" (the page was nav/promo boilerplate — not an error),
        "blocked" (a bot-check; retryable), "failed", or "unroutable". Every non-saved status
        wrote NOTHING.

        A `warning` key on a "saved" result means a DEGRADED success — most often a paper whose
        full text landed but whose metadata lookup was throttled, so it is stored with no title,
        no date and no author. Tell the user when you see it. The atom is searchable by its body,
        but re-saving will not repair it, so a silent "saved" would be misleading.
        """
        from pipeline.kb import hopper as hopper_impl, schema
        from pipeline.kb.embed import get_kb_embedder

        conn = schema.connect()
        try:
            # No embedder needed to preview — build it only for the ingest path (same as add_oracle).
            embedder = get_kb_embedder() if confirm else None
            return hopper_impl.save(conn, embedder, reference,
                                    kind_hint=kind_hint, confirm=confirm)
        finally:
            conn.close()
