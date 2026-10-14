"""A GitHub listing page or one repository README, with the existing fork/date rules."""
import base64

import requests

from pipeline.host_access import request
from pipeline.ingestion.sources import github
from pipeline.kb import ingest_github, schema
from pipeline.kb.ingest_common import changed_hash
from pipeline.platforms.github import api as github_api
from pipeline.timeparse import parse_ts
from pipeline.oracles.sources.archive.adapter import Adapter, Page


def get(path, **kwargs):
    response = request(requests.get, "https://api.github.com" + path,
                       headers=github_api.gh_headers(), timeout=15, **kwargs)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise github_api.GitHubRateLimited("GitHub rate limit reached")
    if response.status_code != 404:
        response.raise_for_status()
    return response


class Github(Adapter):
    def initial(self, spec):
        return {"page": 1, "route": "users"}

    def host(self, spec, kind, data):
        return "api.github.com"

    def page(self, spec, cursor, ctx):
        repo = ingest_github.github_owner_repo(spec["url"])
        if repo:
            response = get(f"/repos/{repo[0]}/{repo[1]}")
            if response.status_code == 404:
                return Page()
            records = [response.json()]
            more = False
        else:
            handle = ingest_github._github_owner(spec["url"])
            response = get(f"/{cursor['route']}/{handle}/repos",
                           params={"sort": "pushed", "direction": "desc", "per_page": 100, "page": cursor["page"]})
            if response.status_code == 404:
                if cursor["route"] == "users" and cursor["page"] == 1:
                    return Page(cursors=[{"page": 1, "route": "orgs"}])
                raise RuntimeError("repository listing disappeared")
            records = response.json()
            more = bool(records)
        if not isinstance(records, list) or not all(isinstance(r, dict) and r.get("full_name") for r in records):
            raise RuntimeError("repository listing was not a list of repositories")
        since = parse_ts(spec.get("since"))
        documents = [(r["full_name"], r) for r in records
                     if repo or (not r.get("fork") and (since is None or ingest_github._pushed_at(r) is None
                                                        or ingest_github._pushed_at(r) >= since))]
        return Page(documents, [{**cursor, "page": cursor["page"] + 1}] if more else [])

    def document(self, conn, spec, data, ctx, hold):
        owner, name = data["full_name"].split("/", 1)
        aid = f"github:{owner}/{name}"
        if spec.get("entry_mode") == "author_referenced" and conn.execute(
                "SELECT 1 FROM atoms WHERE atom_id=?", (aid,)).fetchone():
            return "present"
        response = get(f"/repos/{owner}/{name}/readme")
        readme = None
        if response.status_code != 404:
            body = response.json()
            if body.get("encoding") != "base64" or "content" not in body:
                raise RuntimeError("README response did not carry its content")
            readme = base64.b64decode(body["content"]).decode("utf-8", errors="replace")
        md = github._repo_to_markdown(data, readme, author=f"@{owner}", author_name=owner)
        aid = f"github:{owner}/{name}"
        raw_hash = changed_hash(aid, md, schema.load_hashes(conn, "github"))
        if raw_hash is None:
            return "present"
        atom, who = ingest_github._repo_atom(data, atom_id=aid, raw_hash=raw_hash,
                                           entry_mode=spec.get("entry_mode", "oracle-footprint"), readme=readme)
        schema.upsert_entity(conn, who, name=owner)
        hold(atom, md)
        return aid
