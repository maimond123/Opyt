"""
pipeline/kb/run_ingest.py — the atom-KB ingest CLI.

    python -m pipeline.kb.run_ingest --x-limit 80
    python -m pipeline.kb.run_ingest --github karpathy,openai,ggerganov --min-stars 5

Free sources only (X bookmark cookie-scrape + GitHub public API). Honors `$OPYT_HOME`, so
a sandbox run is `OPYT_HOME=/tmp/opyt-kb-slice python -m pipeline.kb.run_ingest ...`. The
one embedder (from the `embeddings:` config block) is shared by BOTH ingesters and, later,
by query — that shared instance is what guarantees the single-subspace invariant.
"""

from __future__ import annotations

import argparse
import sys

from . import schema
from .embed import get_kb_embedder


def _csv(v: str | None) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def _substack_target(v: str | None) -> tuple[str, str | None]:
    """`--substack` value → (publication_url, handle). Accepts a full URL
    (`https://carol.substack.com`, or a custom domain) or a bare handle (`carol` →
    `https://carol.substack.com`). Returns ("", None) when empty. The handle is best-effort
    (the `{sub}.substack.com` subdomain); a custom domain yields no handle → the entity keys
    on the host, which resolve unifies via the shared publication URL."""
    v = (v or "").strip()
    if not v:
        return "", None
    if v.startswith("http"):
        import re
        m = re.match(r"https?://([^./]+)\.substack\.com", v, re.I)
        return v.rstrip("/"), (m.group(1).lower() if m else None)
    handle = v.lstrip("@").lower()
    return f"https://{handle}.substack.com", handle


def _blog_target(v: str | None) -> str:
    """`--blog` value → a normalized blog home URL. Accepts a full URL or a bare host
    (`simonwillison.net` → `https://simonwillison.net`). Returns "" when empty. Unlike Substack
    there is no `{handle}.host` convention — a blog is just its home URL."""
    v = (v or "").strip()
    if not v:
        return ""
    if not v.startswith("http"):
        v = "https://" + v
    return v.rstrip("/")




def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest free sources into the atom-KB.")
    ap.add_argument("--x-limit", type=int, default=0,
                    help="Ingest up to N X bookmarks (0 = skip X).")
    ap.add_argument("--x-profile", default=None, help="X cookie profile to scrape.")
    ap.add_argument("--github", default="", help="Comma-separated GitHub handles/orgs.")
    ap.add_argument("--min-stars", type=int, default=0, help="Skip repos under N stars.")
    ap.add_argument("--substack", default="",
                    help="A confirmed Oracle's publication URL or handle → ingest their FULL "
                         "archive as footprint atoms (e.g. https://carol.substack.com or carol).")
    ap.add_argument("--substack-name", default=None,
                    help="Display name for the Substack author (optional label).")
    ap.add_argument("--substack-since", default=None,
                    help="Only list Substack posts on/after YYYY-MM-DD (bounds the archive walk).")
    ap.add_argument("--substack-limit", type=int, default=0,
                    help="Cap NEW Substack atoms per run (0 = all). NOTE: a re-run ADVANCES to the "
                         "next batch (already-ingested skipped, next N added) — omit for a full, "
                         "idempotent archive pull.")
    ap.add_argument("--blog", default="",
                    help="A confirmed Oracle's blog home URL → ingest their FULL archive as "
                         "footprint atoms via sitemap discovery (e.g. https://simonwillison.net).")
    ap.add_argument("--blog-name", default=None,
                    help="Display name for the blog author (optional label).")
    ap.add_argument("--blog-since", default=None,
                    help="Only ingest blog posts whose sitemap lastmod is on/after YYYY-MM-DD "
                         "(best-effort — posts with no lastmod always proceed).")
    ap.add_argument("--blog-limit", type=int, default=0,
                    help="Cap NEW blog atoms per run (0 = all). NOTE: a re-run ADVANCES to the "
                         "next batch — omit for a full, idempotent archive pull.")
    ap.add_argument("--force", action="store_true",
                    help="Ingest a footprint source (--substack/--blog) even when the "
                         "single-author eligibility gate would SKIP it (multi-author/org) or route "
                         "it to needs-review (author mismatch / unclassifiable) — an operator "
                         "override for a solo publication the classifier mislabels. Skips the "
                         "classify entirely (no fetch, no LLM spend).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned run and exit (no fetch, no embed, no write).")
    ap.add_argument("--oracle-id", default=None,
                    help="The Oracle's entity id (e.g. x:user:123) this footprint belongs to. When "
                         "a source is SKIPPED as multi-author/org, record it as an `affiliated_with` "
                         "edge from this entity instead of discarding it. Omit → skip as today (no "
                         "affiliation edge without a known Oracle to hang it on).")
    args = ap.parse_args(argv)

    handles = _csv(args.github)

    substack_url, substack_handle = _substack_target(args.substack)
    blog_url = _blog_target(args.blog)

    if args.dry_run:
        gated = "-" if not (substack_url or blog_url) else (
            "forced (gate bypassed)" if args.force else "single-author gate ON")
        print(f"[dry-run] x-limit={args.x_limit} github={handles} "
              f"min-stars={args.min_stars} substack={substack_url or '-'} "
              f"blog={blog_url or '-'} footprint-eligibility={gated}")
        return 0

    if not args.x_limit and not handles and not substack_url and not blog_url:
        ap.error("nothing to do: pass --x-limit and/or --github and/or --substack and/or --blog")

    embedder = get_kb_embedder()
    print(f"[kb] embedder: model={embedder.model} provider={embedder.provider}")
    conn = schema.connect()
    try:
        if args.x_limit:
            from . import ingest_x
            summary = ingest_x.sync_bookmarks(conn, embedder, limit=args.x_limit,
                                              profile=args.x_profile)
            print(f"[kb] x: {summary}")
        if handles:
            from . import ingest_github
            summary = ingest_github.sync_github(
                conn, embedder, handles=handles, min_stars=args.min_stars)
            print(f"[kb] github: {summary}")
        if substack_url:
            from datetime import datetime, timezone

            from . import eligibility, resolve, source_adapters
            since = None
            if args.substack_since:
                since = datetime.fromisoformat(args.substack_since).replace(tzinfo=timezone.utc)
            decision, summary = source_adapters.gate_and_sync_website(
                conn, embedder, "substack", substack_url, author_name=args.substack_name,
                since=since, limit=args.substack_limit, force=args.force, handle=substack_handle)
            if summary is None:
                print(f"[kb] substack {decision.decision.upper()} ({decision.reason}): "
                      f"{substack_url}  → re-run with --force to ingest anyway")
                if decision.decision == "skip" and args.oracle_id:   # multi/org → keep as affiliation
                    org = eligibility.record_affiliation(conn, args.oracle_id, substack_url,
                                                         org_name=args.substack_name)
                    if org:
                        print(f"[kb] affiliation recorded: {args.oracle_id} —affiliated_with→ {org}")
            else:
                print(f"[kb] substack ({decision.reason}): {summary}")
                # Re-resolve so the new substack:{handle} row unifies into the Oracle's canonical
                # (attribution materializes here, not the adapter — mirrors Stage-3's separation).
                stats = resolve.resolve_entities(conn)
                print(f"[kb] resolve: {stats.as_dict()['duplicate_rows_collapsed']} rows collapsed")
        if blog_url:
            from datetime import datetime, timezone

            from . import eligibility, resolve, source_adapters
            since = None
            if args.blog_since:
                since = datetime.fromisoformat(args.blog_since).replace(tzinfo=timezone.utc)
            decision, summary = source_adapters.gate_and_sync_website(
                conn, embedder, "blog", blog_url, author_name=args.blog_name,
                since=since, limit=args.blog_limit, force=args.force)
            if summary is None:
                print(f"[kb] blog {decision.decision.upper()} ({decision.reason}): "
                      f"{blog_url}  → re-run with --force to ingest anyway")
                if decision.decision == "skip" and args.oracle_id:   # multi/org → keep as affiliation
                    org = eligibility.record_affiliation(conn, args.oracle_id, blog_url,
                                                         org_name=args.blog_name)
                    if org:
                        print(f"[kb] affiliation recorded: {args.oracle_id} —affiliated_with→ {org}")
            else:
                print(f"[kb] blog ({decision.reason}): {summary}")
                # Re-resolve so the new blog:{host} row unifies into the Oracle's canonical (the
                # X-website→blog-home attested edge; blog links now count as `self` in resolve).
                stats = resolve.resolve_entities(conn)
                print(f"[kb] resolve: {stats.as_dict()['duplicate_rows_collapsed']} rows collapsed")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
