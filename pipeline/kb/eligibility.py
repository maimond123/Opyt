"""
pipeline/kb/eligibility.py — Stage-5 footprint-eligibility: the single-author gate.

The trust graph answers OWNERSHIP ("does this URL belong to this person?"). This module answers
the second, missing question — FOOTPRINT-ELIGIBILITY: "is this site written by them ALONE?" — so
we never attribute an Oracle's employer's multi-author blog to the Oracle under `who_id = the
person` (the one-atom-one-author breach that got YouTube cut).

Two pieces, split along the person-independent / person-specific seam:

  classify_authorship(conn, url)  → a SITE property (single|multi|unknown), CACHED forever. Fetch
                                    the home/about page (reuse the blog fetch), one LLM classify.
  gate(conn, url, expected_author) → the per-run DECISION (ingest|skip|needs-review), a pure
                                    function of the cached verdict + the (optional) known name.

Degrade direction FLIPS vs Stage-4 SCREEN, on purpose: SCREEN degrades OPEN (hiding a real person
is the expensive error there), while here the expensive error is INGESTING (laundering an org onto
a trusted person), so an unsure/failed classify degrades CLOSED → needs-review.

Fail-safe: only DEFINITIVE verdicts (single|multi) are cached; a transient fetch/LLM failure
returns `unknown` and writes NOTHING, so one hiccup never poisons a source permanently
(mirrors `feedback_llm_failure_must_skip`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from . import schema

# Reuse Stage-4 SCREEN's LLM role by default: it ships on every install (Stage-4), is bound to
# OpenRouter Llama + `response_format: json_object`, and needs no config migration. A distinct
# `authorship_classify` role would RAISE on any ~/.opyt/settings.yaml that predates it → every
# source silently to needs-review. Parameterized so a future caller can override without a code
# change.
_DEFAULT_ROLE = "entity_classify"

# Home/about text budget handed to the classifier — enough to judge single-vs-multi authorship
# without paying for a whole archive page.
_TEXT_BUDGET = 4000

_VALID_AUTHORSHIP = ("single", "multi")

# NOTE the "prefer multi when unsure" lean — the OPPOSITE of SCREEN's "prefer person when unsure".
# Leaning multi means SKIP (a cheap miss) rather than ingest (expensive laundering); it is the
# classifier-level expression of the flipped degrade direction.
_CLASSIFY_SYSTEM = (
    "You label a WEBSITE by whether its content is written by a SINGLE author or by MULTIPLE "
    "authors / an organization, for a knowledge base that ingests ONLY single-author writing. "
    "Judge from the site's home/about text: an individual's personal blog or newsletter is "
    "'single'; a company/product blog, team publication, news outlet, lab, or multi-contributor "
    "site is 'multi'. Return STRICT JSON only, no prose:\n"
    '  {"authorship": "single" | "multi", '
    '"author_name": "<the sole author\'s full name, or null if multi / if single but unnamed>", '
    '"confidence": "high" | "medium" | "low"}\n'
    "When genuinely unsure whether it is one person, prefer 'multi' — we would rather miss a real "
    "blog than attribute an organization's words to one trusted person."
)


@dataclass
class AuthorshipVerdict:
    """A SITE-level authorship verdict. `authorship='unknown'` is the transient failure sentinel —
    it is NEVER cached and the gate routes it to needs-review (degrade-closed). `cached` records
    whether this came from the cache (so the CLI can report a no-cost hit); `confidence`/`reason`
    are observability-only (confidence is not persisted — the cache table has no column for it)."""
    authorship: str                    # "single" | "multi" | "unknown"
    author_name: str | None = None
    confidence: str | None = None
    reason: str | None = None
    cached: bool = False


@dataclass
class GateDecision:
    """The per-run ingest decision. `decision` ∈ {ingest, skip, needs-review}. `verdict` is the
    verdict it rode on (None only when `--force` short-circuited before any classify)."""
    decision: str
    reason: str
    verdict: AuthorshipVerdict | None = None


# ── site key ───────────────────────────────────────────────────────────────────

def _site_key(url: str) -> str:
    """The person-independent cache key: the canonical bare host (`carol.substack.com`,
    `simonwillison.net`) via url_canon — so every URL form on one site shares a verdict.
    Falls back to a normalized raw url if canonicalization yields nothing (never empty-key)."""
    from pipeline.ingestion.url_canon import canonical_identity
    return canonical_identity(url) or (url or "").strip().lower().rstrip("/")


# ── name match (person-specific bonus) ───────────────────────────────────────────

def _norm_tokens(s: str | None) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if t}


def _name_matches(expected: str | None, found: str | None) -> bool:
    """Lenient name equality for the mismatch check: exact, one token set ⊆ the other, or a shared
    token of length ≥4. Either side empty → True (nothing to contradict). Leniency errs toward
    'match' (→ ingest) since this is only a secondary corroboration signal on top of an already
    trust-graph-vouched, already single-authored site."""
    ea, fa = _norm_tokens(expected), _norm_tokens(found)
    if not ea or not fa:
        return True
    if ea <= fa or fa <= ea:
        return True
    return any(len(t) >= 4 for t in (ea & fa))


# ── classify (site property, cached) ─────────────────────────────────────────────

def _parse_verdict(text: str) -> dict | None:
    """LLM text → {authorship, author_name, confidence} or None. Tolerant of Llama's fenced/
    prefixed-JSON habit (same shape as screen._parse_verdicts). None when unparseable or the
    authorship value is not single/multi (→ the caller treats it as `unknown`)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1] or t)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    authorship = str(obj.get("authorship", "")).strip().lower()
    if authorship not in _VALID_AUTHORSHIP:
        return None
    author = obj.get("author_name")
    author = str(author).strip() if author not in (None, "", "null") else None
    conf = obj.get("confidence")
    conf = str(conf).strip().lower() if conf else None
    return {"authorship": authorship, "author_name": author, "confidence": conf}


def _fetch_home_text(url: str) -> str | None:
    """The site's home/about text via the note-ingester's blog fetch (one GET → trafilatura),
    imported lazily so one fetch implementation serves the gate AND the adapters (and so tests
    monkeypatch a single symbol). None on a fetch miss / no extractable content."""
    from pipeline.ingestion.sources.blog import _fetch_article
    article = _fetch_article(url)
    if not article:
        return None
    content = (article.get("content") or "").strip()
    return content or None


def classify_authorship(conn: sqlite3.Connection, source_url: str, *,
                        role: str = _DEFAULT_ROLE) -> AuthorshipVerdict:
    """Classify whether `source_url`'s SITE is single- or multi-authored, caching the verdict.

    Cache-first: a stored `single`/`multi` for this site key returns WITHOUT any fetch or LLM
    call. On a miss, fetch the home/about page and run one classify. Anything that prevents a
    definitive answer — no fetch/content, missing key, a call error, an unparseable body — returns
    `authorship='unknown'` and writes NOTHING (the gate degrades that to needs-review). A definitive
    verdict is written to the cache before returning.
    """
    from pipeline.ingestion.utils import log

    key = _site_key(source_url)
    row = schema.get_authorship(conn, key)
    if row is not None:
        return AuthorshipVerdict(authorship=row["authorship"], author_name=row["author_name"],
                                 reason="cache hit", cached=True)

    text = _fetch_home_text(source_url)
    if not text:
        log(f"[eligibility] classify unknown (no fetchable home text): {source_url}")
        return AuthorshipVerdict("unknown", reason="fetch failed / no content")

    try:
        from pipeline import llm_client
    except Exception as e:                                  # pragma: no cover - import guard
        return AuthorshipVerdict("unknown", reason=f"llm_client import failed: {e}")
    # Preflight: a missing role/key degrades CLOSED (→ unknown → needs-review), never raises.
    try:
        reason = llm_client.preflight(role)
    except Exception as e:
        reason = f"role {role!r} unavailable: {e}"
    if reason:
        log(f"[eligibility] classify unknown (degrade-closed): {reason}")
        return AuthorshipVerdict("unknown", reason=reason)

    user = f"Site: {source_url}\n\nHome/about text:\n{text[:_TEXT_BUDGET]}"
    try:
        resp = llm_client.call(role, system=_CLASSIFY_SYSTEM, user=user)
        parsed = _parse_verdict(resp.text)
    except Exception as e:
        log(f"[eligibility] classify call failed (degrade-closed): {type(e).__name__}: {e}")
        return AuthorshipVerdict("unknown", reason=f"{type(e).__name__}: {e}")

    if not parsed:
        log("[eligibility] classify returned no usable verdict (degrade-closed)")
        return AuthorshipVerdict("unknown", reason="no parseable verdict")

    # Definitive → cache under the site key (fail-safe: a cache-write hiccup still returns the
    # verdict; we just re-classify next run rather than crash the ingest).
    try:
        schema.put_authorship(conn, key, parsed["authorship"], parsed["author_name"])
    except Exception as e:
        log(f"[eligibility] cache write failed for {key}: {e}")
    return AuthorshipVerdict(parsed["authorship"], author_name=parsed["author_name"],
                             confidence=parsed["confidence"], reason="classified")


# ── gate (per-run decision) ──────────────────────────────────────────────────────

def gate(conn: sqlite3.Connection, source_url: str, *, expected_author: str | None = None,
         force: bool = False, role: str = _DEFAULT_ROLE) -> GateDecision:
    """The footprint-eligibility decision for one source, dispatched BEFORE its adapter runs.

    - `force`         → INGEST without classifying (operator override for a solo publication the
                        classifier mislabels; no fetch, no LLM spend — we'd ignore the verdict).
    - `multi`         → SKIP (do not attribute an org/team site to one person).
    - `unknown`       → needs-review (degrade-closed: never auto-ingest an unclassifiable source).
    - `single`        → ingest, unless a known `expected_author` disagrees with the site's sole
                        author → needs-review (single-authored, but by someone ELSE — the squatter
                        / mistaken-link case). A missing name on either side is NOT a mismatch.
    """
    if force:
        return GateDecision("ingest", "forced (eligibility gate overridden by operator)")

    verdict = classify_authorship(conn, source_url, role=role)
    hit = " [cache]" if verdict.cached else ""

    if verdict.authorship == "multi":
        return GateDecision("skip", f"multi-author/org site{hit}", verdict)
    if verdict.authorship == "unknown":
        return GateDecision("needs-review", f"could not classify ({verdict.reason}) — "
                            f"degrade-closed", verdict)
    # single
    if expected_author and verdict.author_name and not _name_matches(expected_author,
                                                                     verdict.author_name):
        return GateDecision("needs-review", f"single-authored by {verdict.author_name!r}, not "
                            f"{expected_author!r} (author mismatch){hit}", verdict)
    return GateDecision("ingest", f"single-author, eligible{hit}", verdict)


# ── affiliation (keep the SKIPPED org, don't discard it) ──────────────────────────

def record_affiliation(conn: sqlite3.Connection, person_entity_id: str, org_url: str, *,
                       org_name: str | None = None) -> str | None:
    """Keep a footprint source the gate SKIPPED as `multi` instead of discarding it: upserts an
    `org:{host}` entity for it. Writes no atoms/chunks — the org's words aren't the person's
    opinions. Idempotent. Returns the org id, or None when inputs are too thin to record (no
    person id, or an uncanonicalizable org_url).

    It also wrote an attested `affiliated_with` edge from the person until the `edges` table was
    deleted 2026-08-23 for having no reader, and stamped the org row with a type label until that
    column went the same day for the same reason. So the ORG ENTITY survives — its `org:` id prefix
    is now the whole marker, and it is what keeps a gate-skipped source from being silently
    discarded — while the person→org RELATION and the unread label do not. The `person_entity_id` argument is kept because the caller's
    contract is unchanged and a future relation store would need it again."""
    from . import derive

    pid = (person_entity_id or "").strip()
    org_id = derive.org_entity_id(org_url)
    if not pid or org_id == "org:unknown":
        return None
    schema.upsert_entity(conn, org_id, name=org_name, identity_links=[org_url])
    return org_id
