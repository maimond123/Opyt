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
        """SAVE something into the knowledge base — a link the user wants to KEEP. Hand it any URL
        and it works out what the thing is and routes it to the right ingester: a research paper,
        a GitHub repo, a Substack post, a plain article or blog post, or a single X post. This is
        the ONLY manual "keep this" path; there is no separate save-a-paper or save-a-repo tool.

        Reach for this whenever the user says keep / save / add / remember / "put this in OPYT"
        about a link — including a link YOU just surfaced from a web search. It persists a link
        into the trusted corpus so the knowledge-base search tool can route to it later.

        TWO-PHASE, and the first phase is free or near-free:
          • confirm=False (the default) → a PREVIEW. It reports which adapter the reference routes
            to, WHY it routed there, the atom id, whether the KB already has it, and what a confirm
            would spend. It NEVER writes. It fetches nothing for an article, paper, repo or
            Substack post — you can already read those yourself, so describe them to the user in
            your own words alongside the routing.
            The X exception: for an x.com status link the preview reads the post (~$0.00015)
            and returns a `description`. You cannot fetch x.com, and `x:2086520133909168332` is
            unverifiable by a human — so read that `description` back before confirming; it is the
            only way the user can catch a wrong link. If `unreadable` comes back instead, the post
            is deleted / protected / keyless: say so and do NOT confirm.
            Paywalls are your job, not the preview's. This tool stores PUBLIC content only —
            a paywalled Substack post is skipped by the adapter and comes back "failed". It reads
            the same cookie-less public endpoint you do, so it cannot see past a wall you hit
            either. If the page you read was a subscriber teaser, say so BEFORE confirming instead
            of spending a round trip to be told no.
          • confirm=True → runs the ingest. This spends: a metered embedding always, plus a content
            gate for articles and ~$0.003 for an X post's thread. Show the preview first — a wrong
            route is SILENT (a paper filed as a blog post never errors, it just sits wrong).

        Skip straight to confirm=True only when the user has already said "yes, save it" about
        THAT specific link.

        `already_present: true` in a preview means a confirm is a no-op — say so and don't spend.
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
        `already_present`, `why` and `cost`. On a confirm, status is one of "saved",
        "already_present", "rejected" (the page was nav/promo boilerplate — not an error),
        "blocked" (a bot-check; retryable), "failed", or "unroutable". Every non-saved status
        wrote NOTHING.

        "budget_paused" can come back from EITHER phase, preview included. It means this tool's
        daily runaway guard tripped — that much spend in one day means something is looping, not
        that the user saved a lot. Nothing was fetched and nothing was written. Show the `message`
        and stop: do not retry, do not try the other phase, and do not reach for another tool to
        save the link. It resets at UTC midnight.

        A `warning` key on a "saved" result means a DEGRADED success — most often a paper whose
        full text landed but whose metadata lookup was throttled, so it is stored with no title,
        no date and no author. Tell the user when you see it. The atom is searchable by its body,
        but re-saving will not repair it, so a silent "saved" would be misleading.
        """
        from pipeline.kb import hopper as hopper_impl, schema
        from pipeline.kb.embed import get_kb_embedder

        conn = schema.connect()
        try:
            # No embedder needed to preview — build it only for the paid path (same as add_oracle).
            embedder = get_kb_embedder() if confirm else None
            return hopper_impl.save(conn, embedder, reference,
                                    kind_hint=kind_hint, confirm=confirm)
        finally:
            conn.close()
