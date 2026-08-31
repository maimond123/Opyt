"""
pipeline/kb/content_gate.py — EXTRACTIVE content-quality gate for fetched web pages.

ONE (or a few) LLM passes per page, run BEFORE embedding: split the page body into UNITS
(paragraph/blocks) → classify each keep/drop → reassemble the kept original units VERBATIM
(the model never rewrites). A wrong-source page (nav-only landing, /careers page, dead JS
shell) falls out for free: every unit drops → the whole page is rejected (None, no atom).

Extractive rather than abstractive so the KB never indexes an LLM paraphrase of the author.
An LLM rather than regex/length because a URL blocklist rots and length doesn't separate
marketing copy from a one-line aphorism — semantic judgment is required.

FAIL-SAFE (load-bearing): a false DROP is invisible (the raw snapshot survives, but nobody
re-derives the index to notice), so the gate is biased hard toward keeping — degrade to
keep-all on any failure, a conservative "when in doubt, keep" prompt, and per-BATCH isolation
(one bad batch keep-alls only its own units, never the page).

Scope: blogs/websites only, not GitHub READMEs / papers / X (its own structural filter).

Rubric: KEEP any unit with substantive knowledge (analysis, transcript dialogue, reference/
protocol-mechanics content, a short real note, reviews/roundups). DROP nav/chrome, promo CTAs,
ads, careers/pricing boilerplate, dead-render shells, off-topic personal content.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pipeline.concurrency import AdaptiveSemaphore

from .chunk import strip_frontmatter

# The gate's own concurrency budget (ARC-1): batches of one page grade independently (disjoint
# unit indices, missing index defaults to KEEP), so fan-out changes only wall-clock, not verdicts.
# Process-wide singleton so future across-atom parallelism shares one budget with the provider,
# same contract as `embed._EMBED_GATE`. Sizing/measurement history:
_GATE_CONCURRENCY = 8
_GATE_SEM = AdaptiveSemaphore(4, min_permits=2, max_permits=_GATE_CONCURRENCY, increase_after=4)

_ROLE = "content_quality"

# Batch budget: accumulate units until either bound trips, then flush one LLM call. Most pages
# fit in one call; a giant transcript spans several batches, which is correct and still cheap.
_MAX_BATCH_CHARS = 12_000
_MAX_BATCH_UNITS = 40

# Split the body into units on blank lines (markdown paragraph/block boundaries). Verbatim: we
# keep each block's EXACT text so a reassembly of kept blocks is byte-for-byte the author's.
_UNIT_SPLIT = re.compile(r"\n[ \t]*\n")

_SYSTEM = """You are a content-quality filter for a personal knowledge base. The KB indexes \
substantive writing and information from people the user follows (researchers, founders, \
investors, scientists). You are shown the UNITS (paragraphs/blocks) of ONE fetched web page, \
each with a numeric index. For EVERY index, decide "keep" or "drop".

DEFAULT TO KEEP. Only drop a unit that clearly matches a DROP category below. Deleting real \
content is far worse than keeping a little chrome.

KEEP (this is the KB's whole purpose):
- analysis, argument, explanation, opinion, or reporting written in prose
- interview / podcast / talk transcript dialogue
- reference or how-it-works content — product docs, protocol mechanics, technical explanation \
(KEEP even when it is NOT the author's personal opinion)
- a short but real note, claim, or aphorism (one sharp idea is substantive even if brief)
- ARTICLE TITLES and SECTION HEADINGS / subheadings — they signpost the content. Keep them (a \
heading fused with its "link to heading" anchor still counts — keep the whole unit)
- book/product reviews, link or paper roundups with commentary
- an INDEX or PORTFOLIO of the author's OWN work: a list of their articles, essays, projects, \
repos, talks, or posts, each a title + a link (e.g. "AI Trading Agent -> \
github.com/them/ai-trading-agent", "Entering the Era -> exponentialview.co/p/..."). The titles \
and links ARE the reference — KEEP them. This is NOT navigation.

DROP ONLY these clearly NON-knowledge units:
- generic SITE NAVIGATION / menus: single-word site sections (Home, About, Blog, Contact, Menu, \
Login), social-media icon links, header/footer chrome
- subscribe / newsletter / "join our community" calls-to-action
- advertising or sponsor reads ("this episode is sponsored by...", "head to acme.com", promo codes)
- promotional / SALES calls-to-action ("Curate NOW", "Buy the course", "I'M READY TO SCALE") and \
links to PRODUCTS / COURSES / SERVICES for sale (NOT the author's own writing or projects)
- careers / jobs / about-us / pricing / team / press / contact boilerplate
- LEGAL disclaimers, disclosures, regulatory or compliance boilerplate ("this is not investment \
advice", "past performance is not indicative...", terms of use, privacy/copyright notices)
- cookie or consent notices, login walls, "JavaScript is required" / empty dead-render shells
- like/share counts, date-only bylines, and the importer's "Original post" / canonical-link footer
- off-topic PERSONAL-LIFE content unrelated to the author's work (travel, family, hobbies)

CALIBRATE BY THE WHOLE PAGE — this matters most:
- If MOST units are substantive writing (the page is clearly a real article / essay / transcript / \
index of work), be VERY conservative: keep every heading, keep every borderline unit, and cut ONLY \
the unmistakable chrome above (ads, subscribe, legal, dead-render, generic nav). On a substantive \
page, if a unit is not obviously chrome, KEEP it.
- Only when a page is MOSTLY non-content (a marketing / landing / nav / boilerplate page with \
little real writing) should you cut aggressively. On such a page even a bare page title or section \
label (e.g. "# Optimize", "# Pricing", "# Team") is just chrome — drop it too, so a pure marketing \
or navigation page is removed entirely (every unit dropped).

RULES:
- Do NOT rewrite, summarize, or reorder. Only classify existing units.
- One page may mix both: keep the substantive units and drop the promo/nav units on the SAME page.

Return ONLY a JSON object mapping every index shown to "keep" or "drop", e.g. {"0":"keep","1":"drop"}."""


@dataclass
class PageVerdict:
    """The full audit trail for one page — what the adapter's thin `gate()` summarizes, and what
    the validation harness grades against."""
    units: list[str]              # the ORIGINAL units, in order (verbatim)
    keep: list[bool]              # keep[i] == True → units[i] survives
    kept_text: str | None         # frontmatter + kept units re-joined verbatim; None if all dropped
    frontmatter: str              # the leading YAML block (provenance) preserved as-is
    degraded: bool                # True if any batch fell back to keep-all (LLM unavailable/failed)
    n_calls: int                  # LLM calls made (0 when degraded before any call)

    @property
    def n_kept(self) -> int:
        return sum(self.keep)

    @property
    def n_dropped(self) -> int:
        return len(self.keep) - self.n_kept


def _split_units(body: str) -> list[str]:
    """Body → ordered, non-empty units (blank-line-delimited blocks), each kept VERBATIM so a
    reassembly of a subset is exactly the author's original text."""
    return [u.strip("\n") for u in _UNIT_SPLIT.split(body) if u.strip()]


def _batch(units: list[str]) -> list[list[int]]:
    """Group unit indices into LLM-sized batches (char + count budget). A single unit larger than
    the char budget still gets its own batch — never dropped for size."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    for i, u in enumerate(units):
        if cur and (cur_chars + len(u) > _MAX_BATCH_CHARS or len(cur) >= _MAX_BATCH_UNITS):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += len(u)
    if cur:
        batches.append(cur)
    return batches


def _prompt(units: list[str], idxs: list[int]) -> str:
    """Render one batch as index-labeled units for classification."""
    return "\n\n".join(f"[{i}] {units[i]}" for i in idxs)


# Tolerant extraction of `"<idx>": "keep"|"drop"` pairs — json_object mode should already give
# valid JSON, but Llama occasionally leaks a stray token; a regex fallback keeps a whole batch's
# verdicts usable instead of degrading it to keep-all on one bad char.
_PAIR_RE = re.compile(r'["\']?(\d+)["\']?\s*:\s*["\'](keep|drop)["\']', re.I)


def _parse_verdicts(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    out[int(str(k).strip())] = str(v).strip().lower()
                except (ValueError, TypeError):
                    continue
    except (json.JSONDecodeError, TypeError):
        pass
    if not out:                                   # fallback: scrape pairs out of the raw text
        for m in _PAIR_RE.finditer(text or ""):
            out[int(m.group(1))] = m.group(2).lower()
    return out


def classify_page(markdown: str, *, role: str = _ROLE) -> PageVerdict:
    """Grade one page's units keep/drop. Never raises — degrades to keep-all on any failure so a
    gate outage can only ADD junk, never delete writing. Returns the full audit (`PageVerdict`)."""
    from pipeline.ingestion.utils import log

    body, offset = strip_frontmatter(markdown or "")
    frontmatter = (markdown or "")[:offset]
    units = _split_units(body)

    def _keep_all(degraded: bool, n_calls: int) -> PageVerdict:
        kept = markdown if units else (markdown or None)
        return PageVerdict(units=units, keep=[True] * len(units), kept_text=kept,
                           frontmatter=frontmatter, degraded=degraded, n_calls=n_calls)

    if not units:                                 # nothing to grade (empty/frontmatter-only body)
        return _keep_all(degraded=False, n_calls=0)

    try:
        from pipeline import llm_client
    except Exception as e:                        # import failure → keep-all
        log(f"[content_gate] llm_client import failed (keep-all): {e}")
        return _keep_all(degraded=True, n_calls=0)

    try:
        reason = llm_client.preflight(role)
    except Exception as e:
        reason = f"role {role!r} unavailable: {e}"
    if reason:                                    # missing role/key → keep-all (degrade)
        log(f"[content_gate] skipped, keep-all (degrade): {reason}")
        return _keep_all(degraded=True, n_calls=0)

    keep = [True] * len(units)                    # default KEEP; only an explicit "drop" flips it
    degraded = False
    n_calls = 0
    batches = _batch(units)
    # `keep` needs no lock: each batch writes ONLY its own (disjoint) indices, and a single list-item
    # assignment is atomic under the GIL — so the result is identical whatever order batches finish
    # in. The tallies below are read-modify-write across threads and DO need one; a lost `n_calls`
    # would silently under-report spend.
    tally_lock = threading.Lock()

    def _grade(idxs: list[int]) -> None:
        """Grade ONE batch. Never raises — per-batch isolation is the contract: a failed batch keeps
        its units and the page lives (identical to the serial version, just off the calling thread)."""
        nonlocal degraded, n_calls
        try:
            with _GATE_SEM:
                try:
                    resp = llm_client.call(role, system=_SYSTEM, user=_prompt(units, idxs))
                except Exception as e:
                    # Halve only on a real 429 — other failures aren't a "too fast" signal.
                    if getattr(e, "status", None) == 429:
                        _GATE_SEM.decrease()
                    raise
            _GATE_SEM.record_success()
            with tally_lock:
                n_calls += 1                      # counted on a RETURNED call, before parsing —
            verdicts = _parse_verdicts(resp.text)  # a parse failure still cost us the call
        except Exception as e:                    # per-BATCH isolation: this batch keeps all, page lives
            log(f"[content_gate] batch failed, keep-all for {len(idxs)} units (degrade): "
                f"{type(e).__name__}: {e}")
            with tally_lock:
                degraded = True
            return
        if not verdicts:
            log(f"[content_gate] batch returned no usable verdicts, keep-all for {len(idxs)} units")
            with tally_lock:
                degraded = True
            return
        for i in idxs:
            if verdicts.get(i) == "drop":         # missing index → stays KEEP (conservative)
                keep[i] = False

    if len(batches) <= 1:                         # short page: no pool, no thread, no overhead
        for idxs in batches:
            _grade(idxs)
    else:
        with ThreadPoolExecutor(max_workers=min(len(batches), _GATE_CONCURRENCY),
                                thread_name_prefix="content_gate") as ex:
            list(ex.map(_grade, batches))         # _grade swallows its own errors, so map can't raise

    kept_units = [units[i] for i in range(len(units)) if keep[i]]
    if not kept_units:                            # EVERY unit dropped → whole-page reject (wrong source)
        return PageVerdict(units=units, keep=keep, kept_text=None, frontmatter=frontmatter,
                           degraded=degraded, n_calls=n_calls)
    kept_text = frontmatter + "\n\n".join(kept_units) if frontmatter else "\n\n".join(kept_units)
    return PageVerdict(units=units, keep=keep, kept_text=kept_text, frontmatter=frontmatter,
                       degraded=degraded, n_calls=n_calls)


def gate(markdown: str, *, role: str = _ROLE) -> str | None:
    """Adapter entry point. Returns the KEPT original text (verbatim, embed this instead of the
    full page), or None if the page is whole-page-rejected (all units non-knowledge → no atom).
    Never raises; degrades to keep-all (returns `markdown`) on any gate failure."""
    return classify_page(markdown, role=role).kept_text


def reapply_keep(markdown: str, keep: list[bool]) -> str | None:
    """Replay a keep-mask computed on the PRE-enrichment body onto the POST-enrichment body
    (adapters grade before describing images, so the mask must be reapplied after enrichment
    adds text). Index i still means the same block before/after because enrichment injects a
    single newline while units split on blank lines. Returns the kept text (frontmatter
    preserved), or None if the page keeps nothing or the mask no longer aligns (refuse rather
    than guess)."""
    body, offset = strip_frontmatter(markdown or "")
    frontmatter = (markdown or "")[:offset]
    units = _split_units(body)
    if len(units) != len(keep):               # alignment lost → refuse, do not guess
        return None
    kept_units = [units[i] for i in range(len(units)) if keep[i]]
    if not kept_units:
        return None
    return frontmatter + "\n\n".join(kept_units) if frontmatter else "\n\n".join(kept_units)


def _cli(argv: list[str] | None = None) -> int:
    """Eyeball one page: `python -m pipeline.kb.content_gate path/to/snapshot.md` prints the
    per-unit keep/drop verdicts + a one-line summary. Manual spot-check before the gold-set run."""
    import sys
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m pipeline.kb.content_gate <snapshot.md>")
        return 2
    md = open(args[0], encoding="utf-8").read()
    v = classify_page(md)
    for i, (u, k) in enumerate(zip(v.units, v.keep)):
        tag = "KEEP" if k else "DROP"
        print(f"[{tag}] {u[:110].replace(chr(10), ' ')}")
    verdict = "REJECT (no atom)" if v.kept_text is None else f"{v.n_kept}/{len(v.units)} units kept"
    print(f"\n--- {verdict}  |  calls={v.n_calls}  degraded={v.degraded} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
