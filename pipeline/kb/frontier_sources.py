"""
pipeline/kb/frontier_sources.py — the artifact adapters Frontier stage 2 executes against.

One job each: take a standing query plus "everything since T", return candidates. No judgement, no
storage, no dedup — the loop in `frontier_execute.py` owns all three. Keeping the adapters this
dumb is what lets a new source be added without touching the loop.

Not `pipeline/artifacts/`'s adapters: their transport is fine (`github_client` is reused below)
but their dedup checks a markdown file in the vault, which is scheduled for deletion. This rail
keys on real external ids in SQLite instead.

Every source below is free, but not all are unmetered: OpenAlex allows roughly 100 anonymous
requests a day against a hard stop (no balance, nothing chargeable). So the scarce resources are
request budget, a published allowance, and the host's patience — the loop bounds requests, the
per-source TTL bounds the day, and a persisted breaker handles a source that is down.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pipeline.timeparse import utc_now

from pipeline.ingestion.utils import log

_UA = "opyt/1.0 (+https://github.com/opyt)"
_TIMEOUT = 20.0

# How many of one paper's authors ride the candidate payload. A bound on payload size, not a
# judgement about who matters: hyperauthored physics papers carry thousands of names and a JSON
# blob of three thousand dicts in `frontier_candidates` buys nothing. Matched to
# `ingest_papers.MAX_ATOM_AUTHORS` so one number governs how many authors of a paper OPYT records
# — a smaller cap here would silently truncate the atom's list for every paper Semantic Scholar
# cannot resolve, which is the COMMON case: 67 of 82 live paper atoms carry no S2 author id.
MAX_PAYLOAD_AUTHORS = 20


def payload_authors(payload: dict) -> list[dict]:
    """The candidate payload's author list, always as `[{"name": …, …}, …]`.

    Tolerant of the pre-2026-09-08 `list[str]` shape because rows staged before then are still
    in the queue (118 on the live store the day this landed) and a candidate payload is FROZEN at
    stage-2 write time — nothing re-parses it from the adapter. Drop this leniency once no
    `list[str]` row can remain; there is no producer of that shape any more.
    """
    out = []
    for a in payload.get("authors") or []:
        if isinstance(a, str):
            if a.strip():
                out.append({"name": a.strip()})
        elif isinstance(a, dict) and (a.get("name") or "").strip():
            out.append(a)
    return out


@dataclass
class Candidate:
    """One artifact found by one query. `candidate_id` is the dedup identity and must be a REAL
    external id, stable across runs and independent of any file we happen to write.

    `source` and `kind` answer two unrelated questions and are two fields for that reason.
    `source` is the FINDER — who turned this up, which is what stage 4 groups and explains by.
    `kind` is the ATOM KIND — which minter materializes it, which is what stage 3 dispatches on.
    They coincided while `arxiv` and `github` were the only adapters, and a third paper source
    made the coincidence a third identical `if source == …` arm in two stage-3 functions.

    `kind` is per-CANDIDATE, not per-adapter, so one adapter can emit several kinds. It carries no
    default: an adapter that forgets it fails at construction rather than staging an artifact no
    minter will claim.
    """
    candidate_id: str
    source: str
    kind: str                          # atom kind: 'paper' | 'repo' — see above
    title: str
    url: str
    published: str | None = None       # ISO date; stored on the candidate, shown by `frontier`
    summary: str = ""
    payload: dict = field(default_factory=dict)


class SourceError(RuntimeError):
    """An adapter could not complete a search. The loop records it and does NOT stamp."""


class RateLimited(RuntimeError):
    """The source told us to back off. Distinct from a generic failure because the correct
    response is different: stop asking, rather than try the next one."""


def _get(url: str, *, headers: dict | None = None) -> bytes | None:
    """One GET. Returns None on a normal failure; raises `RateLimited` on 429.

    The split matters. A 404 or a timeout is about one request, and the next query is unaffected.
    A 429 is about US, and the only useful response is to stop hitting that host — see
    `_BreakerBacked`.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(f"HTTP 429 from {urllib.parse.urlsplit(url).netloc}") from None
        log(f"[frontier-exec] GET failed {url[:110]}: HTTP {e.code}")
        return None
    except Exception as e:
        log(f"[frontier-exec] GET failed {url[:110]}: {type(e).__name__}: {e}")
        return None


# ── A persisted breaker, for the adapters that own their own transport ──────────
class _BreakerBacked:
    """Mixin: one PERSISTED circuit breaker keyed on the adapter's host.

    Persisted rather than in-memory because each run is a fresh child process — an in-memory
    cooldown never survives to the run it is supposed to stop. That matters more here than the
    word "breaker" suggests: a failed pull deliberately does NOT stamp `last_pulled_at`
    (`frontier_execute.record_pull`), so a failing source is due again on the very next pass and
    would be re-asked forever without this.

    Factored out when OpenAlex became the second adapter needing it; `GitHubAdapter` does not,
    because `github_client` carries its own breaker on github.com.
    """
    # The ONE per-adapter knob. threshold and cooldown are deliberately not attributes: both
    # adapters want the same 3-strikes/15-minutes, and a second class attribute in this block
    # would read as an override point nothing overrides.
    breaker_host: str = ""

    def __init__(self, breaker=None):
        self._breaker = breaker

    def _get_breaker(self):
        if self._breaker is None:
            from pipeline.circuit_breaker import CircuitBreaker
            self._breaker = CircuitBreaker(self.breaker_host, threshold=3, cooldown=900.0)
        return self._breaker

    def available(self) -> bool:
        """False while the breaker is open, so a caller can skip this source WITHOUT paying the
        politeness delay first.

        `peek`, never `allow`. `allow` CLAIMS the half-open trial, so asking it here consumed the
        very trial the real call was about to make — the call then found HALF_OPEN, refused, and
        recorded no outcome, leaving the breaker stranded with nothing able to reopen it.
        Measured on the live store 2026-09-08: `api.openalex.org` stranded 257 hours and
        `export.arxiv.org` 30, against a 15-minute cooldown. Both paper adapters silently dead.
        """
        try:
            return bool(self._get_breaker().peek())
        except Exception:
            return True          # a broken breaker must not silence a working source


# ── arXiv ───────────────────────────────────────────────────────────────────────
_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivAdapter(_BreakerBacked):
    """arXiv's Atom API, WINDOWED.

    The repo's existing `_search_arxiv` cannot serve stage 2: it has no date filter, so every run
    would re-rank the same all-time results and "what is new since Tuesday" would be unanswerable.
    arXiv supports a real window inside `search_query` as
    `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]`, ANDed with the terms, and that is what makes an
    incremental pull possible at all.

    Sorted by `submittedDate` descending so the first page is the newest — with a window applied,
    relevance ordering would bury the recent work the whole rail exists to catch.
    """
    slug = "arxiv"
    # arXiv enforces a hard 1-request-per-3s limit; violating it costs 429s.
    min_interval_s = 3.0
    # Pacing alone was measured insufficient: arXiv applies a TIMED BLOCK on the caller (not a
    # per-request limit) once tripped, after which a 3s gap and a 5s gap both got 0 of 5.
    breaker_host = "export.arxiv.org"

    def search(self, query: str, *, since: datetime | None, limit: int = 25) -> list[Candidate]:
        terms = _and_terms(query)
        if not terms:
            # A query that reduces to nothing searchable is a stage-1 bug. Failing loudly costs one
            # error row; falling through would send a bare `submittedDate:[...]` and re-pull the
            # entire window of arXiv under one query's name.
            raise SourceError(f"arxiv: no searchable terms in {query!r}")
        if since:
            until = utc_now() + _one_day()
            terms += (f" AND submittedDate:[{since.strftime('%Y%m%d%H%M')}"
                      f" TO {until.strftime('%Y%m%d%H%M')}]")
        url = (f"{_ARXIV_API}?search_query={urllib.parse.quote(terms)}"
               f"&start=0&max_results={int(limit)}&sortBy=submittedDate&sortOrder=descending")
        from pipeline.circuit_breaker import CircuitOpenError
        breaker = self._get_breaker()
        try:
            body = breaker.call(lambda: _get(url))
        except CircuitOpenError as e:
            raise SourceError(f"arxiv breaker open ({e}) — backing off, retried next run") from None
        except RateLimited as e:
            raise SourceError(str(e)) from None
        if body is None:
            raise SourceError("arxiv request failed")
        return _parse_arxiv(body)


def _one_day():
    from datetime import timedelta
    return timedelta(days=1)


# arXiv reads these as boolean operators, so a query containing one as a WORD would change the
# structure of the search rather than narrow it. Dropping them costs nothing: they carry no
# meaning in an all-ANDed query anyway.
_ARXIV_OPERATORS = {"AND", "OR", "NOT", "ANDNOT"}


def _and_terms(query: str) -> str:
    """Every term ANDed against `all:`. NEVER quoted as a phrase — a quoted string makes arXiv
    demand the words consecutively and in order, and stage-1 queries are topic DESCRIPTORS, not
    verbatim title strings, so phrase-quoting silently zeroed most queries. Flip back to phrase
    matching only if bare terms start flooding results with loosely-matched papers.
    """
    cleaned = re.sub(r'["\\()]', " ", query)
    terms = [t for t in cleaned.split() if t.upper() not in _ARXIV_OPERATORS]
    return " AND ".join(f"all:{t}" for t in terms)


def _parse_arxiv(body: bytes) -> list[Candidate]:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise SourceError(f"arxiv returned unparseable XML: {e}") from None
    out = []
    for e in root.findall(f"{_ATOM_NS}entry"):
        raw_id = (e.findtext(f"{_ATOM_NS}id") or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        if not arxiv_id:
            continue
        # Strip the version suffix: v1 and v2 of a paper are the same artifact, and keying on the
        # versioned id would re-surface every revision as a brand-new candidate.
        bare = re.sub(r"v\d+$", "", arxiv_id)
        out.append(Candidate(
            candidate_id=f"arxiv:{bare}",
            source="arxiv",
            kind="paper",
            title=" ".join((e.findtext(f"{_ATOM_NS}title") or "").split()),
            url=f"https://arxiv.org/abs/{bare}",
            published=(e.findtext(f"{_ATOM_NS}published") or "")[:10] or None,
            summary=" ".join((e.findtext(f"{_ATOM_NS}summary") or "").split())[:2000],
            payload={"authors": [{"name": n} for a in e.findall(f"{_ATOM_NS}author")
                                 if (n := (a.findtext(f"{_ATOM_NS}name") or "").strip())
                                 ][:MAX_PAYLOAD_AUTHORS],
                     "updated": (e.findtext(f"{_ATOM_NS}updated") or "")[:10]}))
    return out


# ── GitHub ──────────────────────────────────────────────────────────────────────
class GitHubAdapter:
    """GitHub repo search, windowed with the `pushed:>DATE` qualifier.

    Reuses `pipeline/artifacts/github_client.py` for transport only — it already carries the
    on-disk cache, the token, and a `CircuitBreaker` keyed on github.com, and duplicating that
    would mean a second unguarded client hammering the same rate limit.

    `pushed` rather than `created` on purpose: the rail wants a thread that MOVED, not one that
    merely exists. Sorted by stars, not push time, because `pushed:>DATE` already selects on
    recency, so sorting by recency too returns almost no signal about which result is worth
    reading — GitHub has no admission bar of its own the way arXiv's moderation does.

    This REORDERS, never excludes: no minimum-star gate. Quality stays a query-time filter the
    host applies, never a write-time one that decides for it.
    """
    slug = "github"
    # 6.0 s = 10 requests/minute, GitHub's UNAUTHENTICATED limit on `/search/repositories`
    # (docs.github.com/en/rest/search/search; authenticated is 30/min). This was 0.0 until
    # 2026-09-05, justified as "token'd search is 30/min" — a budget no install can reach, because
    # `GITHUB_TOKEN` has no acquisition path outside `opyt-keys` and most installs have none. The
    # justification was also wrong for a second reason: `MAX_REQUESTS_PER_RUN` is 40, and 40
    # unpaced requests exceed 30/min as well, so a token would have raised the ceiling and left
    # this adapter over it either way. The cache and the breaker do not prevent a 403 — the
    # breaker trips AFTER one, which costs the pass.
    #
    # The credential is a THROUGHPUT choice, never a correctness one: with this pacing an
    # anonymous install stays inside the limit, and a token only buys 3x the rate.
    min_interval_s = 6.0

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from pipeline.artifacts.github_client import GitHubApiClient
            self._client = GitHubApiClient()
        return self._client

    def search(self, query: str, *, since: datetime | None, limit: int = 25) -> list[Candidate]:
        q = query if not since else f"{query} pushed:>{since.strftime('%Y-%m-%d')}"
        try:
            rows = self._get_client().search_repos(q, limit=limit, sort="stars")
        except Exception as e:
            raise SourceError(f"github search failed: {type(e).__name__}: {e}") from None
        out = []
        for r in rows or []:
            full = r.get("full_name")
            if not full:
                continue
            out.append(Candidate(
                candidate_id=f"repo:{full}",
                source="github",
                kind="repo",
                title=full,
                url=r.get("html_url") or f"https://github.com/{full}",
                published=r.get("pushed_at"),
                summary=(r.get("description") or "")[:2000],
                # Key names follow the payload contract, NOT GitHub's field names:
                # `code_language` because `language` means a natural language to substack/blog,
                # and `source_tags` because author-declared labels get one name across every
                # source. Same spellings `ingest_github` writes, so the repo's two GitHub writers
                # agree. Pinned by tests/kb/test_payload_key_names.py.
                payload={"stars": r.get("stars"), "code_language": r.get("language"),
                         "source_tags": r.get("topics") or [],
                         "archived": bool(r.get("archived"))}))
        return out


# ── OpenAlex ────────────────────────────────────────────────────────────────────
_OPENALEX_HOST = "https://api.openalex.org"
_OPENALEX_API = f"{_OPENALEX_HOST}/works"
# Asked for explicitly so the response stays small; every field below is read.
_OPENALEX_SELECT = ("id,doi,title,publication_date,authorships,abstract_inverted_index,"
                    "primary_location,best_oa_location,type,cited_by_count,relevance_score")


class OpenAlexAdapter(_BreakerBacked):
    """OpenAlex works search — ~250M records across every discipline, keyless.

    Its coverage is the point: it indexes published, peer-reviewed literature in fields arXiv does
    not touch, so it subsumes the discipline-specific sources (RePEc for economics, INSPIRE for
    physics, ERIC for education) that would otherwise each need an adapter.

    WHY THIS ADAPTER LOOKS BACK FURTHER THAN THE RESUME POINT, AND SORTS BY RELEVANCE.
    Both filtering and sorting on INDEX date (`from_created_date`, `sort=created_date`) are behind
    OpenAlex's paid plans — verified 2026-08-26, each returns HTTP 429 "Plan upgrade required".
    The free tier can only window on PUBLICATION date, and OpenAlex indexes a work well after it
    is published. Measured over 400 works in two one-week publication windows (2026-02 and
    2026-05, both old enough that indexing had finished): the lag is 1-2 days at the median, but
    8-9% of works are indexed more than 7 days after publication, 4-5% more than 14, and 1-1.5%
    more than 30. A `since_for`-width window would miss every one of those PERMANENTLY and
    silently — they are published before the window opens and indexed after it closes.

    So this adapter declares `min_lookback_days` and the loop widens its window to match (see
    `frontier_execute._lookback_floor` — declared here, applied there, so `window_ok` still
    validates the window that is actually sent). 30 days covers ~98.5% of the lag distribution and
    is not an arbitrary number: it is stage 4's own `RECENCY_HALF_LIFE_DAYS`, i.e. exactly as far
    back as the ranker still treats a candidate as fresh.

    A wide window then FORCES the sort. Sorted by date it would return the same newest 25 every
    run and the late-indexed tail — the whole reason the window is wide — could never reach the
    page. Sorted by relevance a work competes on match quality whenever it was indexed, and
    recency is scored where it belongs, in stage 4. The cost is that consecutive pulls return
    largely the same page; `upsert_candidate` dedups it for free and the run reports it honestly
    as `candidates_seen`.

    ANONYMOUS QUOTA. OpenAlex allows about 100 anonymous requests before the daily allowance is
    exhausted. At 30 standing queries on a 48h beat that is well inside the allowance.

    THE ALLOWANCE RESETS AT MIDNIGHT UTC, NOT ON A ROLLING WINDOW. Measured 2026-08-27 by
    exhausting it: `Retry-After` came back 21,946s, which lands within 22 SECONDS of the next
    midnight UTC, and the body says so outright ("Resets at midnight UTC"). This matters in two
    ways a rolling window would not. A day that burns its credits stays burned until midnight —
    nothing trickles back — which is what the persisted breaker below exists to prevent, since a
    FAILING source does not stamp and is due again on the next pass. And the allowance is per
    IP, so ANYTHING ELSE on this machine querying OpenAlex anonymously uses the same pool: a
    one-off measurement script starved the live loop for the rest of that day.
    """
    slug = "openalex"
    # A courtesy gap between requests. NOT the binding limit, and it is worth being clear about
    # which limit is which, because the two fail completely differently.
    #
    # OpenAlex moved to usage-based pricing in February 2026, and the binding constraint is now a
    # DAILY BUDGET, not a rate. Read off a live 429 on 2026-09-12: `X-RateLimit-Limit: 1000`,
    # `"Insufficient budget… Resets at midnight UTC"`, `Retry-After: 68682`. So an anonymous
    # caller gets on the order of a thousand requests a DAY, and no amount of spacing them out
    # buys a single extra one. (This comment previously said "OpenAlex documents 10 requests/
    # second" and treated that as the constraint. That is what a stale claim about an external
    # service looks like: still plausible, no longer what governs us.)
    #
    # Spacing still earns its keep for the OTHER failure — a burst tripping the per-second cap and
    # opening the persisted breaker for every caller on this machine. Budget exhaustion is the one
    # that has actually bitten: a 2026-09-11 measurement run spent the day's allowance in an hour,
    # and every OpenAlex read returned None afterwards.
    min_interval_s = 1.0
    min_lookback_days = 30
    breaker_host = "api.openalex.org"

    def search(self, query: str, *, since: datetime | None, limit: int = 25) -> list[Candidate]:
        terms = _openalex_terms(query)
        if not terms:
            # Same fail-safe as arXiv: a bare filter with no terms re-pulls the entire window of
            # OpenAlex under one query's name. One error row is the cheaper outcome.
            raise SourceError(f"openalex: no searchable terms in {query!r}")
        params = {"search": terms, "per-page": str(int(limit)), "select": _OPENALEX_SELECT,
                  # Stated rather than left to the API default, because choosing relevance over
                  # date is the load-bearing decision in this adapter (see the class docstring).
                  "sort": "relevance_score:desc"}
        if since:
            params["filter"] = f"from_publication_date:{since.strftime('%Y-%m-%d')}"
        url = f"{_OPENALEX_API}?{urllib.parse.urlencode(params)}"

        from pipeline.circuit_breaker import CircuitOpenError
        breaker = self._get_breaker()
        try:
            body = breaker.call(lambda: _get(url))
        except CircuitOpenError as e:
            raise SourceError(f"openalex breaker open ({e}) — backing off, "
                              f"retried next run") from None
        except RateLimited as e:
            raise SourceError(str(e)) from None
        if body is None:
            raise SourceError("openalex request failed")
        return _parse_openalex(body)


def _openalex_terms(query: str) -> str:
    """The query, with the two characters OpenAlex reads as SYNTAX removed.

    A double quote makes OpenAlex demand an exact phrase, and stage-1 queries are topic
    descriptors rather than title strings — the same punctuation that silently zeroed 24 of 26
    arXiv pairs (see `_and_terms`). Terms are NOT ANDed here: OpenAlex ranks by relevance over the
    whole phrase, which is what this adapter wants, so there is nothing to join.
    """
    return " ".join(re.sub(r'["\\]', " ", query or "").split())


def _abstract_from_inverted(index: dict | None) -> str:
    """OpenAlex ships abstracts as a `{word: [positions]}` inverted index (a licensing artifact,
    not a compression one). Rebuilt because abstract length IS the substance signal stage 4 ranks
    papers by, and because it is the body an admitted atom gets when no open PDF resolves."""
    if not isinstance(index, dict) or not index:
        return ""
    words: dict[int, str] = {}
    for word, positions in index.items():
        for p in positions or []:
            if isinstance(p, int):
                words[p] = word
    return " ".join(words[i] for i in sorted(words))


def _openalex_authors(work: dict) -> list[dict]:
    """One work's authorships → `[{"name", "openalex_id", "orcid", "position"}, …]`, capped.

    The id is the point. It is the only stable handle on a PERSON that OpenAlex gives, and
    without it the only way back to an author is a `display_name` search — which returned 16
    people for "Frances Arnold" on 2026-09-08, one with 928 works and one with 2. Carrying the id
    that already sits in the same dict means that lookup never has to run.

    The ORCID rides along for the same reason and at the same price: it is the one identifier
    nobody can claim on another person's behalf, which is what makes it the only safe key for
    merging an `openalex:` entity with a `scholar:` one. Measured 2026-09-08 over 256
    authorships: 73% carry one, free, in this same response.

    `openalex_id` and `orcid` are stored BARE (`A5043841592`, `0000-0002-4027-364X`), not as the
    URLs OpenAlex returns, because bare is what `/works?filter=author.id:` takes and what an
    entity id is built from. `position` is OpenAlex's own `first`/`middle`/`last`.

    Truncation is plain byline order. `is_corresponding` is deliberately NOT read: it is flagged
    on NONE of six measured hyperauthored works (2026-09-08 — both ATLAS Higgs papers, Gemini,
    Gemini 1.5, Gemini-in-medicine, Code Llama), so preferring corresponding authors above the cap
    would keep nobody on exactly the papers a cap exists for. OpenAlex truncates at 100 itself.
    """
    out = []
    for a in (work.get("authorships") or [])[:MAX_PAYLOAD_AUTHORS]:
        author = a.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if not name:
            continue
        # An absent key rather than a null: a payload with no nulls in it has ONE shape, so no
        # reader needs a "present but None" branch. Nothing distinguishes the two here — OpenAlex
        # has never returned an authorship whose author carries a name but no id.
        oid = str(author.get("id") or "").rsplit("/", 1)[-1]
        orcid = str(author.get("orcid") or a.get("raw_orcid") or "").rsplit("/", 1)[-1]
        pos = (a.get("author_position") or "").strip()
        out.append({"name": name,
                    **({"openalex_id": oid} if oid else {}),
                    **({"orcid": orcid} if orcid else {}),
                    **({"position": pos} if pos else {})})
    return out


def _parse_openalex(body: bytes) -> list[Candidate]:
    import json as _json
    try:
        payload = _json.loads(body)
    except ValueError as e:
        raise SourceError(f"openalex returned unparseable JSON: {e}") from None
    out = []
    for w in payload.get("results") or []:
        work_id = str(w.get("id") or "").rsplit("/", 1)[-1]
        loc = w.get("primary_location") or {}
        # The DOI first: it is the canonical identifier, and `_parse_paper_url` folds an arXiv DOI
        # back onto its arXiv id, so an arXiv preprint reached this way dedups against the one the
        # arXiv adapter already staged. The landing page is the fallback for the works OpenAlex
        # holds without a DOI. Neither present → no offline route to an atom id, so no candidate.
        url = w.get("doi") or loc.get("landing_page_url")
        if not work_id or not url:
            continue
        source = (loc.get("source") or {}).get("display_name")
        # The OPEN pdf, so the minter can build a full-document atom instead of an abstract-only
        # one. `best_oa_location` first because that is OpenAlex's own "best open version" — the
        # primary location can be the paywalled publisher copy. Measured over 68 live results
        # 2026-08-26: 47 carry one here (46 via primary, 47 via best_oa), and 25 of those are
        # works `_fulltext_pdf_urls` cannot otherwise reach, since its list is arXiv's mirror then
        # S2's openAccessPdf and S2 does not index them.
        pdf_url = ((w.get("best_oa_location") or {}).get("pdf_url")
                   or loc.get("pdf_url"))
        out.append(Candidate(
            candidate_id=f"openalex:{work_id}",
            source="openalex",
            kind="paper",
            title=" ".join((w.get("title") or "").split()),
            url=url,
            published=w.get("publication_date"),
            summary=_abstract_from_inverted(w.get("abstract_inverted_index"))[:2000],
            payload={"authors": _openalex_authors(w),
                     "venue": source, "type": w.get("type"),
                     "cited_by_count": w.get("cited_by_count"),
                     "pdf_url": pdf_url,
                     # The score OpenAlex itself ranked this page by. No judgement here — the
                     # intake cut lives in `frontier_execute._relevance_cut`, where the whole
                     # page is visible; an adapter sees one work at a time.
                     "relevance_score": w.get("relevance_score")}))
    return out


# ── OpenAlex, by the id of whatever produced the work ───────────────────────────
# Not a Frontier adapter and deliberately not in `adapters()` below: it answers "what has THIS
# AUTHOR OR VENUE published", not "what is new on this topic". It lives in this file anyway
# because `api.openalex.org` transport belongs in one place — the `_get`, the `RateLimited` split
# and, above all, the persisted breaker keyed on that host. Its two callers (`scholar_probe`,
# which writes the untrusted candidate store, and `ingest_scholar_footprint`, which writes
# `atoms`) sit on opposite sides of a trust boundary and must not import each other.

# The fields one work needs. Asked for explicitly because OpenAlex returns ~50 per work by
# default and the inverted abstract index alone is most of the body.
WORKS_SELECT = ("id,doi,title,publication_date,abstract_inverted_index,"
                "primary_location,best_oa_location,type,cited_by_count,authorships")
AUTHOR_SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                 "last_known_institutions")

# The fields ONE work needs when the DOI is already known — `work_by_doi`'s select. Separate from
# `WORKS_SELECT` because that one feeds a corpus walk keyed on an id, where `doi` identifies the
# row; here the caller HOLDS the DOI it asked with and instead needs the year as its own field
# (`ingest_papers._openalex_metadata` fills S2's `year`, which no corpus caller reads).
_OPENALEX_WORK_SELECT = ("title,abstract_inverted_index,primary_location,best_oa_location,"
                         "publication_date,publication_year,cited_by_count,authorships")

# Rows per request when paging a whole corpus. OpenAlex's documented maximum, so a 928-work
# author costs 5 requests rather than 38.
_PAGE_SIZE = 200

# The hard stop on ONE id's corpus walk, in requests. 50 pages × 200 = 10,000 works, which is an
# order of magnitude past the most prolific real researcher (Frances Arnold: 928, measured
# 2026-09-08). It exists because `next_cursor` is server-supplied: a server that keeps handing one
# back would page forever, and this rail's whole allowance resets only at midnight UTC.
#
# A VENUE has far more works than any author — ChemRxiv is 63,565, measured 2026-09-08 — but it
# still does not reach this ceiling, because `sync_scholar_footprint` caps its own pull at
# `MAX_WORKS_PER_PULL` (2,000) first. That cap is where a venue truncates, it is where the
# `capped` flag is raised, and this one stays what it always was: a guard against a server that
# keeps handing back a cursor forever.
MAX_AUTHOR_PAGES = 50

# Which `/works` field filters for a given OpenAlex id, keyed by the letter OpenAlex prefixes
# every id with: `A…` is an author, `S…` a source (a journal, a preprint repository, a venue).
# That one letter is the whole reason an author feed and a venue feed are the SAME code path
# instead of two adapters — the id already says which field it belongs in.
_WORKS_FILTER_FIELD: dict[str, str] = {"A": "author.id", "S": "primary_location.source.id"}


def works_filter(openalex_id: str, *, topics: str | None = None,
                 since: datetime | None = None) -> str:
    """An OpenAlex id, plus optional topic and date bounds → one `/works` filter expression.

    The BASE clause is chosen by the id's own letter prefix, which is what makes
    `author.id:A5043841592` and `primary_location.source.id:S4393918830` interchangeable
    everywhere below. The topic and date clauses are identical on both.

    `topics` is a `|`-joined list of OpenAlex topic ids — OpenAlex's own OR syntax, so the stored
    value IS the filter value. It is validated where it is WRITTEN
    (`oracle_refresh_state.set_topic_filter`), which is the trust boundary; by the time it reaches
    here it came out of our own store.

    Raises on an id whose prefix names no field. The producer is real: `pair_from_member` splits
    an `openalex:{tail}` entity id and returns the tail unchecked, so a malformed entity id
    arrives here. Refusing is the point — `author.id:W123` is a well-formed request that returns
    zero works, which reads as "this person published nothing" and is the worst possible lie for
    a filter to tell.
    """
    field = _WORKS_FILTER_FIELD.get((openalex_id or "")[:1])
    if not field:
        raise ValueError(f"{openalex_id!r} is not an OpenAlex author (A…) or source (S…) id — "
                         f"there is no /works field to filter it on")
    filt = f"{field}:{openalex_id}"
    if topics:
        filt += f",primary_topic.id:{topics}"
    if since:
        filt += f",from_publication_date:{since.strftime('%Y-%m-%d')}"
    return filt


class OpenAlexWorksAdapter(_BreakerBacked):
    """One OpenAlex id → the works it produced, plus the counts that let a caller ask about them
    before pulling. The id is an AUTHOR (`A…`) or a SOURCE (`S…` — a journal, a preprint
    repository, a venue); `works_filter` reads the prefix and everything below is shared.

    Not to be confused with `OpenAlexAdapter` above, which searches TERMS. This one never
    searches: the id already names exactly what produced the work, so there is no match quality
    and nothing to rank on.

    Transport only, no judgement: it never decides which works matter, how far back to go, which
    topics are wanted, or what becomes an atom. `available()` is checked BEFORE any politeness
    delay, the way `frontier_execute` does it — a host already known to be down must not also
    cost a delay per caller to rediscover that.
    """
    breaker_host = "api.openalex.org"
    min_interval_s = 1.0        # OpenAlex documents 10/s; a courtesy margin, not a measurement

    def _call(self, path: str, params: dict) -> dict | None:
        import json as _json

        from pipeline.circuit_breaker import CircuitOpenError

        url = f"{_OPENALEX_HOST}{path}?{urllib.parse.urlencode(params)}"
        try:
            body = self._get_breaker().call(lambda: _get(url))
        except CircuitOpenError as e:
            raise SourceError(f"openalex breaker open ({e}) — backing off") from None
        except RateLimited as e:
            raise SourceError(str(e)) from None
        if body is None:
            return None
        try:
            return _json.loads(body)
        except ValueError as e:
            raise SourceError(f"openalex returned unparseable JSON: {e}") from None

    def works(self, openalex_id: str, *, since: datetime | None = None, limit: int = 0,
              topics: str | None = None, pace_seconds: float = 0.0) -> list[dict]:
        """This id's works, NEWEST FIRST, optionally windowed, topic-filtered and capped.

        Sorted by date, not relevance — the opposite of `OpenAlexAdapter`, and for a reason that
        does not generalize between them. That one searches TERMS, so a work competes on match
        quality and sorting by date would return the same newest page every run. Here the filter
        already names exactly what produced the work, so there is no match quality to rank on and
        recency is the only ordering that means anything.

        `limit=0` walks the whole corpus by cursor, bounded by `MAX_AUTHOR_PAGES`. A single page
        (`limit<=_PAGE_SIZE`) skips the cursor entirely, so the common candidate-probe call is one
        request with no paging state.

        A caller that passed a `limit` sees truncation for itself: a full return means more
        matched. That is how `sync_scholar_footprint` reports `capped`, and it is why this returns
        a plain list rather than growing a flag — the one caller that asks the question already
        holds the number that answers it.
        """
        base = {"filter": works_filter(openalex_id, topics=topics, since=since),
                "sort": "publication_date:desc", "select": WORKS_SELECT}

        if 0 < limit <= _PAGE_SIZE:
            payload = self._call("/works", {**base, "per-page": str(int(limit))})
            return list((payload or {}).get("results") or [])

        out: list[dict] = []
        cursor = "*"
        for page in range(MAX_AUTHOR_PAGES):
            if page and pace_seconds > 0:
                time.sleep(pace_seconds)
            payload = self._call("/works", {**base, "per-page": str(_PAGE_SIZE),
                                            "cursor": cursor})
            results = list((payload or {}).get("results") or [])
            out.extend(results)
            if limit and len(out) >= limit:
                return out[:limit]
            cursor = ((payload or {}).get("meta") or {}).get("next_cursor")
            # A short page is the LAST page. Checked as well as the cursor because a server that
            # keeps handing one back would otherwise page to the request cap on every call.
            if not cursor or len(results) < _PAGE_SIZE:
                break
        return out

    def author(self, author_id: str) -> dict | None:
        """The author's own record — display name, ORCID, institution, counts, h-index."""
        return self._call(f"/authors/{author_id}", {"select": AUTHOR_SELECT})

    def author_by_orcid(self, orcid: str) -> dict | None:
        """An ORCID → the OpenAlex author record, in one free call.

        The ORCID is the only identifier nobody can claim on another person's behalf, and it is
        the one a researcher actually publishes about themselves — so it is what a user reaches
        for when adding one. It is not a handle OPYT can pull anything by, though: only an
        OpenAlex author id has a works feed. This is the bridge between the two.

        Verified 2026-09-08: `/authors/orcid:0000-0002-4027-364X` returns A5043841592 (928 works).
        """
        return self._call(f"/authors/orcid:{orcid}", {"select": AUTHOR_SELECT})

    def year_counts(self, openalex_id: str, *, topics: str | None = None) -> dict[int, int]:
        """`{year: n_works}` over this id's corpus, in one call.

        This is what lets the lookback question carry real numbers — "928 papers; the last 2 years
        is 28" — instead of blind presets, which is something neither the `x` nor the `web`
        selector can do. `group_by` returns every year bucket, not a page of them.

        `topics` narrows it to the same works the pull will take, so once a topic filter is stored
        the window numbers are POST-FILTER. A count-first ask that reported the unfiltered total
        against a filtered pull would misstate the one number it exists to state.
        """
        payload = self._call("/works", {"filter": works_filter(openalex_id, topics=topics),
                                        "group_by": "publication_year"})
        out: dict[int, int] = {}
        for row in (payload or {}).get("group_by") or []:
            try:
                out[int(row["key"])] = int(row["count"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def topic_counts(self, openalex_id: str) -> list[dict]:
        """`[{id, name, count}]` over this id's WHOLE corpus, biggest first, in one call.

        Deliberately UNFILTERED by topic, unlike `year_counts`: this is the list the user picks
        FROM, so narrowing it by the current selection would hide every topic they might add.

        OpenAlex returns at most 200 groups and sorts them by count descending, so an id with a
        longer tail than that is silently short — measured 2026-09-08: Frances Arnold has 174
        topics and fits, ChemRxiv reports exactly 200 and does not. The caller shows a handful and
        states the rest as a number, and everything is selected until the user narrows, so the
        clipped tail costs a display line rather than a paper.

        `key` comes back as a full `https://openalex.org/T…` URL; the bare id is what
        `works_filter` puts in a filter, so the strip happens here rather than at each caller.
        """
        payload = self._call("/works", {"filter": works_filter(openalex_id),
                                        "group_by": "primary_topic.id"})
        out: list[dict] = []
        for row in (payload or {}).get("group_by") or []:
            tid = str(row.get("key") or "").rstrip("/").rsplit("/", 1)[-1]
            name = row.get("key_display_name")
            if not tid.startswith("T") or not name:
                continue                    # `unknown` buckets and anything unrecognisable
            try:
                out.append({"id": tid, "name": name, "count": int(row["count"])})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def source(self, source_id: str) -> dict | None:
        """One OpenAlex source record by id — display name, work count, kind. The venue
        counterpart of `author`, and the exact-match half of the venue lookup: a pasted
        `openalex.org/S…` needs no search at all."""
        return self._call(f"/sources/{source_id}",
                          {"select": "id,display_name,works_count,type"})

    def sources_by_name(self, name: str) -> list[dict]:
        """OpenAlex sources whose display name matches `name` — the venue lookup, transport half.

        A NAME search, which is exactly what the scholar path refuses for PEOPLE ("Frances Arnold"
        returned 16 authors on 2026-09-08). It is admissible here only because the caller applies
        a uniqueness rule to the result rather than taking the top hit: see `oracles._openalex_root`,
        which is where that judgement lives. OpenAlex exposes no filterable homepage field —
        `homepage_url` is not a valid filter and comes back NULL even for ChemRxiv — so there is
        no exact-match alternative to apply.
        """
        payload = self._call("/sources", {"filter": f"display_name.search:{name}",
                                          "select": "id,display_name,works_count,type",
                                          "per-page": "5"})
        return list((payload or {}).get("results") or [])

    def work_by_doi(self, doi: str) -> dict | None:
        """The OpenAlex record for one DOI, or None if it indexes no such work.

        The FILTER form, not `/works/doi:…`: an unknown DOI comes back as an empty result set
        rather than a 404, so "OpenAlex has never heard of this paper" is a value to test and not
        an exception to catch (verified 2026-09-11). Its caller —
        `ingest_papers._openalex_metadata` — runs inside `paper_from_url`, which may not raise, so
        the shape that needs no `except` is the one worth having.

        Transport only, like every method here: it reports what OpenAlex holds and decides nothing
        about identity. The caller keeps the DOI it asked with as the paper's id.
        """
        payload = self._call("/works", {"filter": f"doi:https://doi.org/{doi}",
                                        "select": _OPENALEX_WORK_SELECT, "per-page": "1"})
        works = (payload or {}).get("results") or []
        return works[0] if works else None

    def work_by_openalex_id(self, work_id: str) -> dict | None:
        """One work by its own `W…` id — the exact-match counterpart of `works_by_landing_page`,
        for the case where the user pasted an OpenAlex page itself."""
        return self._call(f"/works/{work_id}", {"select": "doi,title"})

    def works_by_landing_page(self, url: str) -> list[dict]:
        """Every OpenAlex work that lists `url` as one of its locations — transport half.

        An EXACT match on a stored string, not a search: a hit means OpenAlex has recorded that
        this exact page is a copy of that work. That is the difference between this and a title
        lookup, and it is the whole reason this is admissible where bibliographic search is not
        (Crossref title search resolved 1 of 3 on 2026-09-09, returning the wrong paper twice).

        Returns the LIST, deliberately. A page that several works claim is ambiguous, and the
        uniqueness judgement belongs to the caller — the same split as `sources_by_name` and
        `oracles._openalex_root`. Measured 2026-09-11: 30 of 33 recovered pages matched exactly
        one work, 1 matched two, so ambiguity is rare but real.
        """
        payload = self._call("/works", {"filter": f"locations.landing_page_url:{url}",
                                        "select": "doi,title", "per-page": "5"})
        return list((payload or {}).get("results") or [])


# ── Registry ────────────────────────────────────────────────────────────────────
# A `target_sources` value with no entry here is reported as `no_adapter` and COUNTED, never
# silently dropped — so a name the reader may route to but that has no adapter is a deliberate
# second pass, not an oversight. `reader_core.VALID_SOURCES` is the routable vocabulary; the
# difference between it and this registry is what remains unbuilt. Deliberately NOT restated as a
# number here: the previous count sat at "five" while the real figure was seven.
def adapters() -> dict:
    return {a.slug: a for a in (ArxivAdapter(), GitHubAdapter(), OpenAlexAdapter())}
