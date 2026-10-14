"""Public Substack archive pages and one essay, using the existing footprint renderer."""
from urllib.parse import urlsplit

import requests

from pipeline.host_access import request
from pipeline.kb import ingest_substack, schema
from pipeline.kb.ingest_common import StageTimer
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack import publication as substack_publication
from pipeline.timeparse import parse_ts
from pipeline.oracles.sources.archive.adapter import Adapter, Page


class Substack(Adapter):
    def initial(self, spec):
        return {"offset": 0}

    def host(self, spec, kind, data):
        return urlsplit(spec["url"]).hostname

    def page(self, spec, cursor, ctx):
        base = spec["url"].rstrip("/")
        session = ctx.cache.setdefault("substack_sessions", {}).setdefault(base, requests.Session())
        response = request(session.get, f"{base}/api/v1/archive",
                           params={"sort": "new", "offset": cursor["offset"], "limit": substack_publication.ARCHIVE_PAGE_SIZE},
                           headers={"User-Agent": substack_http.USER_AGENT}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not all(isinstance(p, dict) and p.get("id") for p in batch):
            raise substack_contracts.SubstackListingError("archive page was not a list of posts")
        since = parse_ts(spec.get("since"))
        documents = [(str(p["id"]), p) for p in batch if since is None or not substack_publication.older_than(p, since)]
        more = batch and not (since and substack_publication.older_than(batch[-1], since))
        return Page(documents, [{"offset": cursor["offset"] + len(batch)}] if more else [])

    def document(self, conn, spec, data, ctx, hold):
        aid = f"substack:{data['id']}"
        seen = schema.load_hashes(conn, "substack")
        if aid in seen:
            return "present"
        if substack_publication.is_paywalled(data) or not data.get("slug"):
            return "skipped"
        # The shared retry ladder owns retries; this turn spends one body request.
        full = substack_http.authed_get_json_retry(
            f"{spec['url'].rstrip('/')}/api/v1/posts/{data['slug']}", {}, {},
            referer=spec["url"].rstrip("/") + "/", retries=1, backoff=0)
        if not isinstance(full, dict):
            raise substack_contracts.SubstackFetchError("post response was not an object")
        if substack_publication.is_paywalled(full) or not (full.get("body_html") or "").strip():
            return "skipped"
        result = ingest_substack.build_footprint_post(
            {"post": data, "full": full, "post_full": {**data, **full}, "atom_id": aid},
            publication_url=spec["url"], handle=spec.get("handle"), author_name=spec.get("author"),
            who_id=spec["entity"], seen=seen, img_cache=ctx.cache["images"], timer=StageTimer())
        if result["outcome"] != "ok":
            return result["outcome"]
        hold(result["atom"], result["md"])
        return aid
