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
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pipeline.timeparse import utc_now

from pipeline.ingestion.utils import log

_UA = "opyt/1.0 (+https://github.com/opyt)"
_TIMEOUT = 20.0


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
    published: str | None = None       # ISO date; used for the cursor and for display
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

    Persisted rather than in-memory because each run is a fresh detached process — an in-memory
    cooldown never survives to the run it is supposed to stop. That matters more here than the
    word "breaker" suggests: a failed pull deliberately does NOT stamp `last_pulled_at`
    (`frontier_execute.record_pull`), so a failing source is due again on the very next spawn and
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
        """False while the breaker is open, so the loop can skip this source WITHOUT paying the
        politeness delay first. Extended measurement:
"""
        try:
            return bool(self._get_breaker().allow())
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
            payload={"authors": [a.findtext(f"{_ATOM_NS}name")
                                 for a in e.findall(f"{_ATOM_NS}author")][:12],
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
    min_interval_s = 0.0        # token'd search is 30/min; the client's cache and breaker cover it

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
_OPENALEX_API = "https://api.openalex.org/works"
# Asked for explicitly so the response stays small; every field below is read.
_OPENALEX_SELECT = ("id,doi,title,publication_date,authorships,abstract_inverted_index,"
                    "primary_location,best_oa_location,type,cited_by_count,relevance_score")


class OpenAlexAdapter(_BreakerBacked):
    """OpenAlex works search — ~250M records across every discipline, keyless.

    Its coverage is the point: it indexes published, peer-reviewed literature in fields arXiv does
    not touch, so it subsumes the discipline-specific sources (RePEc for economics, INSPIRE for
    physics, ERIC for education) that would otherwise each need an adapter.

    WHY THIS ADAPTER LOOKS BACK FURTHER THAN THE CURSOR, AND SORTS BY RELEVANCE.
    Both filtering and sorting on INDEX date (`from_created_date`, `sort=created_date`) are behind
    OpenAlex's paid plans — verified 2026-08-26, each returns HTTP 429 "Plan upgrade required".
    The free tier can only window on PUBLICATION date, and OpenAlex indexes a work well after it
    is published. Measured over 400 works in two one-week publication windows (2026-02 and
    2026-05, both old enough that indexing had finished): the lag is 1-2 days at the median, but
    8-9% of works are indexed more than 7 days after publication, 4-5% more than 14, and 1-1.5%
    more than 30. A cursor-width window would miss every one of those PERMANENTLY and silently —
    they are published before the window opens and indexed after it closes.

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

    A consequence worth stating plainly: `cursor_ts` is recorded for this source but is NOT
    load-bearing, because the free tier cannot support an incremental pull at all.

    METERED, BUT NOT PAID. Anonymous calls carry `x-ratelimit-limit: 1000` credits against
    `x-ratelimit-credits-required: 10` per request — 100 requests — with
    `x-ratelimit-prepaid-remaining-usd: 0`, so there is no balance and nothing can be charged. At
    30 standing queries on a 48h beat that is well inside the ceiling.

    THE ALLOWANCE RESETS AT MIDNIGHT UTC, NOT ON A ROLLING WINDOW. Measured 2026-08-27 by
    exhausting it: `Retry-After` came back 21,946s, which lands within 22 SECONDS of the next
    midnight UTC, and the body says so outright ("Resets at midnight UTC"). This matters in two
    ways a rolling window would not. A day that burns its credits stays burned until midnight —
    nothing trickles back — which is what the persisted breaker below exists to prevent, since a
    FAILING source does not stamp and is due again on the next spawn. And the allowance is per
    IP, so ANYTHING ELSE on this machine querying OpenAlex anonymously spends the same pool: a
    one-off measurement script starved the live loop for the rest of that day.

    The ceiling is deliberately not wired into the per-rail spend meters: those measure money
    leaving, and this is an allowance.
    """
    slug = "openalex"
    # OpenAlex documents 10 requests/second. One per second is an order of magnitude inside it and
    # costs at most one second per pair per run. Not a measurement — a courtesy margin.
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
            payload={"authors": [(a.get("author") or {}).get("display_name")
                                 for a in (w.get("authorships") or [])[:12]
                                 if (a.get("author") or {}).get("display_name")],
                     "venue": source, "type": w.get("type"),
                     "cited_by_count": w.get("cited_by_count"),
                     "pdf_url": pdf_url,
                     # The score OpenAlex itself ranked this page by. No judgement here — the
                     # intake cut lives in `frontier_execute._relevance_cut`, where the whole
                     # page is visible; an adapter sees one work at a time.
                     "relevance_score": w.get("relevance_score")}))
    return out


# ── Registry ────────────────────────────────────────────────────────────────────
# A `target_sources` value with no entry here is reported as `no_adapter` and COUNTED, never
# silently dropped — so a name the reader may route to but that has no adapter is a deliberate
# second pass, not an oversight. `reader_core.VALID_SOURCES` is the routable vocabulary; the
# difference between it and this registry is what remains unbuilt. Deliberately NOT restated as a
# number here: the previous count sat at "five" while the real figure was seven.
def adapters() -> dict:
    return {a.slug: a for a in (ArxivAdapter(), GitHubAdapter(), OpenAlexAdapter())}
