"""OpenAlex: the term search, the client by author or venue (`OpenAlexClient`), the `/works`
filter, and a work's abstract and authors. Transport and records only.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

from pipeline.credentials import openalex_headers
from pipeline.platforms.papers import registry as paper_registry
from pipeline.platforms.papers.registry import (
    MAX_PAYLOAD_AUTHORS,
    BreakerBacked,
)
from pipeline.platforms.errors import SourceError

# Pacing limits bursts; the shared breaker also honors daily-budget Retry-After responses.
MIN_INTERVAL_S = 1.0
BREAKER_HOST = "api.openalex.org"

_OPENALEX_HOST = "https://api.openalex.org"
_OPENALEX_API = f"{_OPENALEX_HOST}/works"
# Asked for explicitly so the response stays small; every field below is read.
_OPENALEX_SELECT = ("id,doi,title,publication_date,authorships,abstract_inverted_index,"
                    "primary_location,best_oa_location,type,cited_by_count")
# The two orders a works search is asked in; which one a window deserves is the caller's
# decision (`frontier/openalex_candidates.sort_for`).
NEWEST, MOST_CITED = "publication_date:desc", "cited_by_count:desc"


def search_works(query: str, *, since: datetime | None, limit: int, get_breaker,
                 sort: str = NEWEST) -> list[dict]:
    """The works whose title or abstract holds `query` as a phrase, in `sort` order (`NEWEST` or
    `MOST_CITED`), windowed on publication date: one request through the caller's breaker
    (`get_breaker()`, built only once a request is due), the page's works. A phrase filter, not
    OpenAlex's full-text `search`, which always fills its page with near-misses. Raises
    `SourceError` on a query with no searchable terms, an open breaker, a 429, a failed request or
    an unparseable page."""
    phrase = _openalex_phrase(query)
    if not phrase:
        # Same fail-safe as arXiv: a bare filter with no phrase re-pulls the entire window of
        # OpenAlex under one query's name. One error row is the cheaper outcome.
        raise SourceError(f"openalex: no searchable terms in {query!r}")
    found = f'title_and_abstract.search:"{phrase}"'
    if since:
        found += f",from_publication_date:{since.strftime('%Y-%m-%d')}"
    params = {"filter": found, "per-page": str(int(limit)), "select": _OPENALEX_SELECT,
              "sort": sort}
    url = f"{_OPENALEX_API}?{urllib.parse.urlencode(params)}"
    breaker = get_breaker()
    body = paper_registry.call_get(breaker, url, headers=openalex_headers())
    if body is None:
        raise SourceError("openalex request failed")
    return parse_works(body)


def parse_works(body: bytes) -> list[dict]:
    """An OpenAlex works page (JSON) → its works. Raises `SourceError` on a malformed page."""
    return _listed(_json_object(body), "results")


def _json_object(body: bytes) -> dict:
    """An OpenAlex response body → its JSON object. Raises `SourceError` for anything else."""
    try:
        payload = json.loads(body)
    except ValueError as e:
        raise SourceError(f"openalex returned unparseable JSON: {e}") from None
    if not isinstance(payload, dict):
        raise SourceError(f"openalex returned JSON that is not an object: {type(payload).__name__}")
    return payload


def _listed(payload: dict | None, key: str) -> list:
    """The list under `key` in an OpenAlex response; empty when the resource is absent (404).
    Raises `SourceError` when the page carries no such list: a malformed page is not an empty
    one, and reading it as empty is how a failure became "nothing published"."""
    if payload is None:
        return []
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise SourceError(f"openalex response carries no {key!r} list")
    return rows


def _openalex_phrase(name: str) -> str:
    """The name with the characters OpenAlex's filter syntax reads removed: the quote that closes
    the phrase, and the comma, pipe and colon that separate filters and values."""
    return " ".join(re.sub(r'["\\,|:]', " ", name or "").split())


def abstract_from_inverted(index: dict | None) -> str:
    """OpenAlex ships abstracts as a `{word: [positions]}` inverted index (a licensing artifact,
    not a compression one). Rebuilt because abstract length is the substance signal stage 4 ranks
    papers by, and because it is the body an admitted atom gets when no open PDF resolves."""
    if not isinstance(index, dict) or not index:
        return ""
    words: dict[int, str] = {}
    for word, positions in index.items():
        for p in positions or []:
            if isinstance(p, int):
                words[p] = word
    return " ".join(words[i] for i in sorted(words))


@dataclass(frozen=True)
class WorkRecord:
    """The fields every reader of one OpenAlex work takes from it, decoded once.

    `doi` and `published` are as OpenAlex gives them (a `https://doi.org/` url, an ISO date);
    `abstract` is rebuilt and whole, so each caller cuts it to its own length; `pdf_url` is the
    open copy, `best_oa_location` first because the primary location can be the paywalled
    publisher copy; `venue` is the primary location's source name."""
    doi: str | None
    landing_page: str | None
    pdf_url: str | None
    venue: str | None
    title: str
    published: str | None
    abstract: str

    @property
    def url(self) -> str | None:
        """The work's offline route to an atom id: its DOI, else its landing page."""
        return self.doi or self.landing_page


def work_record(work: dict) -> WorkRecord:
    """One OpenAlex work → its `WorkRecord`."""
    loc = work.get("primary_location") or {}
    return WorkRecord(
        doi=work.get("doi"),
        landing_page=loc.get("landing_page_url"),
        pdf_url=(work.get("best_oa_location") or {}).get("pdf_url") or loc.get("pdf_url"),
        venue=(loc.get("source") or {}).get("display_name"),
        title=" ".join((work.get("title") or "").split()),
        published=work.get("publication_date"),
        abstract=abstract_from_inverted(work.get("abstract_inverted_index")),
    )


def openalex_authors(work: dict) -> list[dict]:
    """One work's authorships → `[{"name", "openalex_id", "orcid", "position"}, …]`, capped.

    The id is the only stable handle on a person OpenAlex gives (a name search is ambiguous); the
    ORCID is the only safe key for merging an `openalex:` entity with a `scholar:` one. Both are
    stored bare (`A5043841592`, `0000-0002-4027-364X`), which is what filters and entity ids
    take. `position` is OpenAlex's own `first`/`middle`/`last`. Truncation is plain byline order.
    """
    out = []
    for a in (work.get("authorships") or [])[:MAX_PAYLOAD_AUTHORS]:
        author = a.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if not name:
            continue
        # An absent key rather than a null, so the payload has one shape.
        oid = str(author.get("id") or "").rsplit("/", 1)[-1]
        orcid = str(author.get("orcid") or a.get("raw_orcid") or "").rsplit("/", 1)[-1]
        pos = (a.get("author_position") or "").strip()
        out.append({"name": name,
                    **({"openalex_id": oid} if oid else {}),
                    **({"orcid": orcid} if orcid else {}),
                    **({"position": pos} if pos else {})})
    return out


# ── OpenAlex, by the id of whatever produced the work ───────────────────────────
# Not a Frontier adapter: it answers "what has this author or venue published", not "what is new
# on this topic". It lives here because `api.openalex.org` transport and its persisted breaker
# belong in one place.

# The fields one work needs. Asked for explicitly because OpenAlex returns ~50 per work by
# default and the inverted abstract index alone is most of the body.
WORKS_SELECT = ("id,doi,title,publication_date,abstract_inverted_index,"
                "primary_location,best_oa_location,type,cited_by_count,authorships")
AUTHOR_SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                 "last_known_institutions")

# The fields one work needs when the DOI is already known — `work_by_doi`'s select: the caller
# holds the DOI and instead needs the year (`lookup.openalex_metadata` fills S2's `year`).
_OPENALEX_WORK_SELECT = ("title,abstract_inverted_index,primary_location,best_oa_location,"
                         "publication_date,publication_year,cited_by_count,authorships")

# Which `/works` field filters for a given OpenAlex id, keyed by the letter OpenAlex prefixes
# every id with: `A…` is an author, `S…` a source (a journal, a preprint repository, a venue).
# That letter is why an author feed and a venue feed share one code path.
_WORKS_FILTER_FIELD: dict[str, str] = {"A": "author.id", "S": "primary_location.source.id"}

# A DOI or an ORCID pasted as its URL. Left on, the request asks OpenAlex for
# `https://doi.org/https://doi.org/…` or `orcid:https://orcid.org/…`, and either reads as unknown.
_DOI_URL_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.I)


def _bare_doi(doi: str) -> str:
    return _DOI_URL_PREFIX.sub("", (doi or "").strip())


def _bare_orcid(orcid: str) -> str:
    return (orcid or "").strip().rstrip("/").rsplit("/", 1)[-1]


def works_filter(openalex_id: str, *, topics: str | None = None,
                 since: datetime | None = None) -> str:
    """An OpenAlex id, plus optional topic and date bounds → one `/works` filter expression.

    The base clause is chosen by the id's letter prefix; the topic and date clauses are the same
    on both. `topics` is a `|`-joined list of OpenAlex topic ids (OpenAlex's own OR syntax),
    validated where it is written (`registry.set_topic_filter`).

    Raises on an id whose prefix names no field (`pair_from_member` can pass a malformed tail):
    `author.id:W123` would return zero works and read as "this person published nothing".
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


class OpenAlexClient(BreakerBacked):
    """One OpenAlex id → the works it produced, plus the counts that let a caller ask about them
    before pulling. The id is an author (`A…`) or a source (`S…` — a journal, a preprint
    repository, a venue); `works_filter` reads the prefix and everything below is shared.

    Unlike `search_works`, this never searches terms. Transport only: it never decides which works
    matter or what becomes an atom.
    """
    breaker_host = BREAKER_HOST
    min_interval_s = MIN_INTERVAL_S

    def _call(self, path: str, params: dict) -> dict | None:
        """One GET's JSON object, or None when OpenAlex has no such resource (404). Raises
        `SourceError` on a failed request, an open breaker, a 429, or a body that is not a JSON
        object."""
        url = f"{_OPENALEX_HOST}{path}?{urllib.parse.urlencode(params)}"
        body = paper_registry.call_get(self._get_breaker(), url, headers=openalex_headers())
        if body is None:
            return None
        return _json_object(body)

    def works(self, openalex_id: str, *, limit: int, since: datetime | None = None,
              topics: str | None = None) -> list[dict]:
        """This id's newest `limit` works, newest first, optionally windowed and topic-filtered:
        one request. Sorted by date, because the filter already names what produced the work, so
        there is no match quality to rank on. A full return means more matched.
        """
        payload = self._call("/works", {
            "filter": works_filter(openalex_id, topics=topics, since=since),
            "sort": "publication_date:desc", "select": WORKS_SELECT,
            "per-page": str(int(limit))})
        return _listed(payload, "results")

    def author(self, author_id: str) -> dict | None:
        """The author's own record — display name, ORCID, institution, counts, h-index."""
        return self._call(f"/authors/{author_id}", {"select": AUTHOR_SELECT})

    def author_by_orcid(self, orcid: str) -> dict | None:
        """An ORCID → the OpenAlex author record, in one free call.

        Only an OpenAlex author id has a works feed; this bridges the ORCID a user has to it.
        """
        return self._call(f"/authors/orcid:{_bare_orcid(orcid)}", {"select": AUTHOR_SELECT})

    def year_counts(self, openalex_id: str, *, topics: str | None = None) -> dict[int, int]:
        """`{year: n_works}` over this id's corpus, in one call.

        Lets the lookback question carry real numbers. `topics` narrows it to the same works the
        pull will take, so the counts are post-filter.
        """
        payload = self._call("/works", {"filter": works_filter(openalex_id, topics=topics),
                                        "group_by": "publication_year"})
        out: dict[int, int] = {}
        for row in _listed(payload, "group_by"):
            try:
                out[int(row["key"])] = int(row["count"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def topic_counts(self, openalex_id: str) -> list[dict]:
        """`[{id, name, count}]` over this id's whole corpus, biggest first, in one call.

        Unfiltered by topic, unlike `year_counts`: this is the list the user picks from. OpenAlex
        returns at most 200 groups. `key` arrives as a full URL and is stripped to the bare id.
        """
        payload = self._call("/works", {"filter": works_filter(openalex_id),
                                        "group_by": "primary_topic.id"})
        out: list[dict] = []
        for row in _listed(payload, "group_by"):
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

        A name search, admissible only because the caller (`identify.openalex_root`) applies a
        uniqueness rule rather than taking the top hit; OpenAlex has no filterable homepage field.
        """
        payload = self._call("/sources", {"filter": f"display_name.search:{name}",
                                          "select": "id,display_name,works_count,type",
                                          "per-page": "5"})
        return _listed(payload, "results")

    def work_by_doi(self, doi: str) -> dict | None:
        """The OpenAlex record for one DOI, or None if it indexes no such work.

        The filter form, not `/works/doi:…`: an unknown DOI comes back as an empty result set
        rather than a 404, so its caller inside `paper_from_url` needs no `except`.
        """
        payload = self._call("/works", {"filter": f"doi:https://doi.org/{_bare_doi(doi)}",
                                        "select": _OPENALEX_WORK_SELECT, "per-page": "1"})
        works = _listed(payload, "results")
        return works[0] if works else None

    def work_by_openalex_id(self, work_id: str) -> dict | None:
        """One work by its own `W…` id — the exact-match counterpart of `works_by_landing_page`,
        for the case where the user pasted an OpenAlex page itself."""
        return self._call(f"/works/{work_id}", {"select": "doi,title"})

    def works_by_landing_page(self, url: str) -> list[dict]:
        """Every OpenAlex work that lists `url` as one of its locations — transport half.

        An exact match on a stored string, not a search, which is why it is admissible where
        title search is not. Returns the list: the uniqueness judgement belongs to the caller.
        """
        payload = self._call("/works", {"filter": f"locations.landing_page_url:{url}",
                                        "select": "doi,title", "per-page": "5"})
        return _listed(payload, "results")
