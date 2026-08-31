"""
pipeline/kb/embed_surface.py — the text an atom's chunk is EMBEDDED from, as opposed to the
text it is STORED and RENDERED as.

OPYT's own renderer output (footer, byline, media headers, thread markers, image markup) is
byte-identical across atoms from the same source, so leaving it in pushes unrelated atoms
together in embedding space. BM25 is immune to this (a term in most documents is discounted
automatically); an averaged embedding is not. Do NOT reach for an attribution-style extractor
here — one that keeps only the author's own words is destructive for retrieval, which wants
everything a person could search for. (The vault rail had exactly such a surface; it went with
that rail, and reintroducing its shape here would be the mistake this paragraph names.)

Strip the MARKUP, keep every IDENTITY token: stripping markup costs nothing on author-name
queries, but stripping names collapses precision near chance. The display name in the byline
survives, the author handle is RECOVERED from the footer URL before the footer is dropped, and a
quoted person's `@handle` survives untouched.

Fail-safe direction (CLAUDE.md invariant): every pattern is LINE-ANCHORED and runs per chunk, so
a scaffolding block cut in half by a chunk boundary is simply left in — the failure mode is
"kept too much", never "cut content". A strip that empties a chunk falls back to the original
text, since an empty string is itself a shared, degenerate vector.

Bump `STRIP_VERSION` whenever a pattern changes: `embed.ensure_kb_meta` refuses to write a new
generation of vectors into a store built by a different one, so an unbumped regex change leaves
the store silently stale.
"""

from __future__ import annotations

import re

# Identity of the strip that produced a stored vector. Guarded in `kb_meta.strip_version`.
# '' (empty) means "never stripped" — every store written before this module existed.
# ONE constant covers BOTH profiles (deliberately, at the cost of over-reporting staleness for
# `full`).
STRIP_VERSION = "2026-08-11.2"

# The renderer's middle dot (U+00B7) and em dash (U+2014). Spelled out because they are load-bearing
# in the anchors below and are invisible in a diff.
_DOT = "·"
_DASH = "—"

# ── whole-line drops ─────────────────────────────────────────────────────────────
# A line that survives one of these carries nothing a person could search for.

# `*Bookmarked · [Original post](url)*` / `*GitHub · [View repo](url)*` / similar labeled footers.
# The label maps to `source_type`, already a filterable column, so the label is dropped. The URL
# is NOT thrown away: `_identity_from_url` lifts the handle out before the footer line is dropped.
_FOOTER = re.compile(
    rf"^\*(?P<label>[A-Za-z][A-Za-z ]{{0,30}}?) {_DOT} "
    rf"\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)\)\*$"
)

# `> **Thread** · [4 posts](url)` — positional scaffolding, one distinct shape.
_THREAD_HEADER = re.compile(rf"^>?\s*\*\*Thread\*\* {_DOT} \[\d+ posts?\]\([^)]*\)\s*$")

# `## Media` / `## Links` — pure template. Only the HEADER line goes: what sits under `## Media`
# is the `*Image:*` VLM description, and taking the section would take that with it.
_SECTION_HEADER = re.compile(r"^#{1,6}\s+(?:Media|Links)\s*$")

# `---` between thread posts and above the footer.
_HRULE = re.compile(r"^-{3,}\s*$")

# ── line rewrites ────────────────────────────────────────────────────────────────

# `# Soren Larson — 2025-06-26` -> `Soren Larson`. The DATE goes: it is stored structurally as
# `when_ts`, which is what the date filter actually reads, and a bare ISO date embeds poorly.
# The NAME stays (see module docstring). `date_str` is `unknown` when unparseable.
_BYLINE = re.compile(rf"^#\s+(?P<name>.+?)\s+{_DASH}\s+(?:\d{{4}}-\d{{2}}-\d{{2}}|unknown)\s*$")

# `**Quoting** [@lecong](url):` -> `@lecong:`. The word is template; the handle behind it is
# frequently the ONLY naming of that person in the atom.
_QUOTING = re.compile(r"^\*\*Quoting\*\*\s+")

# `**[1/2]** …` -> `…`. Positional only.
_THREAD_MARKER = re.compile(r"^\*\*\[\d+/\d+\]\*\*\s*")

# `*Image:* a chart showing …` -> `a chart showing …`. The MARKER only, never what follows it.
_IMAGE_MARKER = re.compile(r"^\*Image:\*\s*")

# `![photo](https://pbs.twimg.com/…)`. TWO different rules wearing one regex — on X both alt and
# url are machine-emitted (media-type field + CDN hash), so `_strip_line` branches on source
# rather than gating this as a unit. Elsewhere the alt is an author-written caption and survives;
# only the url goes, and that half is `full`-only.
_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")

# `[Original post](url)` -> `Original post`. Runs AFTER `_IMAGE`, or the image's `!` would be
# orphaned by the link rule eating the `[alt](url)` half of it.
_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\([^)]*\)")

# Bare `https://t.co/xyz` — an opaque shortcode. The EXPANDED url it points at is rendered
# separately as a link card, so nothing is lost. Runs after `_LINK`, which has already removed the
# t.co links that were inside markdown link syntax.
_TCO = re.compile(r"https?://t\.co/\S+")

_HEADING = re.compile(r"^#{1,6}\s+")     # after `_BYLINE`, which needs its own `#` to match
_QUOTE_MARK = re.compile(r"^>\s?")       # `> ` on quoted lines — the CONTENT of the quote stays
_EMPHASIS = re.compile(r"\*+")           # `**bold**` / `*italic*`; `_` is left alone (`__init__`)
_BLANK_RUN = re.compile(r"\n{3,}")


def _identity_from_url(url: str) -> str:
    """The identity token buried in a footer URL, or `''` when there is none.

    This is why the footer is a rewrite and not a plain drop: it carries a second identity token
    per atom (an author handle), distinct from the display name in the byline, found nowhere else.
    """
    m = re.match(r"https?://(?:www\.)?x\.com/([A-Za-z0-9_]{1,20})/status/", url)
    if m:
        return f"@{m.group(1)}"
    m = re.match(r"https?://(?:www\.)?github\.com/([^/\s?#]+/[^/\s?#]+)", url)
    if m:
        return m.group(1)
    # Anything else (a blog, a Substack): the host is the publication's identity.
    m = re.match(r"https?://(?:www\.)?([^/\s?#]+)", url)
    return m.group(1) if m else ""


# ── profiles: WHICH rules run ────────────────────────────────────────────────────
# Two arms, different blast radii. "full" runs every rule below, including urls a person chose
# to write; "scaffolding" runs the uniform render-template rules only (footer, byline markup,
# section headers, thread markers, the `*Image:*` marker word, X's image markup, emphasis/heading
# marks) and leaves every author-written url alone. The boundary is AUTHOR-WRITTEN vs
# MACHINE-EMITTED, not "touches a url". The strip's inertness has a shelf life: scaffolding is
# atom-level and vectors are chunk-level, so it depends on the corpus's long-form share.
#
# Named for the JUDGEMENT (what a PERSON chose to write), not the syntax — see the doc for the
# naming history.
_AUTHORED_URL_RULES = frozenset({"image", "link", "tco", "blank_collapse"})

_PROFILES = {
    "full": frozenset(),                    # nothing skipped
    "scaffolding": _AUTHORED_URL_RULES,     # every author-written url is left alone
}

# The profile the INGEST path embeds with. One name, read by both the writer (`ingest_common.
# _chunk_snapshot`) and the guard (`embed.assert_strip_version`), so the two cannot disagree about
# what the store holds. Switching which arm ships is this line plus a `restrip --profile <name>
# --apply`.
#
# Ships on the argument, not a measured retrieval win (neither arm was shown to improve it): the
# text `scaffolding` removes is definitionally not the author's, so deleting it cannot be the
# wrong direction. `full` additionally deletes urls the author chose to cite — a content
# judgement, not covered by that argument. See the doc for the full rationale.
DEFAULT_PROFILE = "scaffolding"


def _strip_line(line: str, source_type: str, skip: frozenset = frozenset()) -> str | None:
    """One line -> its embeddable form, or `None` to drop the line entirely.

    `skip` names rules NOT to apply (see `_PROFILES`). Ordering still matters among the rules that
    DO run: `_IMAGE` must precede `_LINK`, or the link rule eats the `[alt](url)` half of an image
    and orphans its `!`."""
    s = line.strip()
    if not s:
        return ""
    if _SECTION_HEADER.match(s) or _HRULE.match(s) or _THREAD_HEADER.match(s):
        return None
    m = _FOOTER.match(s)
    if m:
        ident = _identity_from_url(m.group("url"))
        return ident or None

    out = _BYLINE.sub(r"\g<name>", s)
    out = _QUOTE_MARK.sub("", out)
    out = _THREAD_MARKER.sub("", out)
    out = _QUOTING.sub("", out)
    out = _IMAGE_MARKER.sub("", out)
    # ONE regex, TWO rules, split by who wrote the bytes. On X both halves of `![photo](pbs.twimg…)`
    # are machine-emitted (a media-type field and a CDN hash), so it is scaffolding and runs under
    # BOTH profiles. Everywhere else the alt is a caption the author wrote, so dropping the url
    # around it is a content judgement and stays `full`-only. Written as branches rather than as
    # `"image" not in skip or source_type == "x"` on purpose: that one-liner would make
    # `"image" in _AUTHORED_URL_RULES` stop meaning "this rule is skipped", and a profile table you
    # cannot read behavior off of is worse than an extra line.
    if source_type == "x":
        out = _IMAGE.sub("", out)
    elif "image" not in skip:
        out = _IMAGE.sub(r"\g<alt>", out)
    if "link" not in skip:
        out = _LINK.sub(r"\g<text>", out)
    if "tco" not in skip:
        out = _TCO.sub("", out)
    out = _HEADING.sub("", out)
    out = _EMPHASIS.sub("", out)
    return out.strip()


def strip_version(profile: str = "full") -> str:
    """The identity stamped into `kb_meta.strip_version` for this profile.

    Profile is IN the version because it is part of what produced the vector, exactly as the regex
    set is. Without it, re-embedding a store from `full` to `scaffolding` would leave kb_meta
    reporting an unchanged identity while every vector moved — the precise silent-staleness this
    guard exists to prevent."""
    if profile not in _PROFILES:
        raise ValueError(f"unknown strip profile {profile!r} (known: {sorted(_PROFILES)})")
    return STRIP_VERSION if profile == "full" else f"{STRIP_VERSION}+{profile}"


def _strip_body(text: str, source_type: str | None = None, profile: str = "full") -> str:
    """The strip WITHOUT the empty-result fallback — so `''` genuinely means "this chunk was
    nothing but scaffolding". Separated from `strip_for_embedding` only so a caller that needs to
    COUNT those chunks can tell them apart from a chunk the strip simply had no work to do on.
    Both look identical from the outside once the fallback has fired."""
    if not text:
        return ""
    skip = _PROFILES.get(profile)
    if skip is None:
        raise ValueError(f"unknown strip profile {profile!r} (known: {sorted(_PROFILES)})")
    st = (source_type or "").strip().lower()
    kept: list[str] = []
    for line in text.split("\n"):
        out = _strip_line(line, st, skip)
        if out is not None:
            kept.append(out)
    joined = "\n".join(kept)
    # The blank-line collapse is itself a profile-gated rule: it is whitespace, not scaffolding.
    if "blank_collapse" not in skip:
        joined = _BLANK_RUN.sub("\n\n", joined)
    return joined.strip()


def strip_for_embedding(text: str, source_type: str | None = None,
                        profile: str = "full") -> str:
    """`chunks.text` -> the string that chunk is embedded from. Pure: no I/O, no DB, no clock.

    Rules are keyed on `source_type` because a source-blind strip is what makes this dangerous:
    X alt text is a template slot, blog alt text is a caption the author wrote. `profile` selects
    WHICH rules run — see `_PROFILES` for the two arms and why the near-inert one has a point.

    A strip that empties the chunk returns the ORIGINAL text. An empty string is not a free vector
    — it is one vector shared by every fully-scaffolded chunk, which is the clustering failure this
    module exists to remove, reintroduced at the bottom of the corpus.
    """
    if not text:
        return text or ""
    return _strip_body(text, source_type, profile) or text
