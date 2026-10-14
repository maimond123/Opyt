"""
pipeline/retrieval/who.py — a handle, URL or id → the author clusters it names, by local SQL only.
"""

from __future__ import annotations

from pipeline.kb import derive, schema


# ── handle → who_id (the read-side lookup) ────────────────────────────────────
# Not shared with `identify.resolve_handle`: that is a write path (pays twitterapi.io, mints
# entity rows). This is read-only local SQL — a search must never cost money or invent an entity.

# Entity-id namespaces (`derive.py` mints all of them); an id passes straight through.
_ID_PREFIXES = ("x:user:", "github:", "substack:", "blog:", "org:", "scholar:", "paper-authors:")


def resolve_who(conn, who: str) -> list[dict]:
    """A handle / URL / id → the cluster(s) of entities it names, each with the `who_ids` to
    hand back as a `who_id=` filter. Local SQL only. Unknown input → [] (never everything).

    A handle is stored in three shapes depending on adapter: GitHub/Substack key the id suffix
    directly (`github:karpathy`, matched case-insensitively); X ids are numeric so the handle
    lives in `entities.profile.handle`; Substack/blog homes key on the host and resolve via
    `derive.*_entity_id`, tried only when the input looks like a URL.

    Each match expands to its whole resolved cluster: an atom carries the per-platform `who_id`
    of its author, so one person's work spans several ids. `who_ids` is the full cluster; an
    unlinked platform stays its own cluster, which under-returns but never returns the wrong
    person's atoms.

    Returns [{canonical_id, name, handle, who_ids, matched_on}], one entry per cluster."""
    raw = (who or "").strip()
    if not raw:
        return []

    seeds: dict[str, str] = {}          # lower(entity_id) → why it was tried
    if raw.startswith(_ID_PREFIXES):
        seeds[raw.lower()] = "entity_id"

    handle = raw.lstrip("@").lower()
    if handle and "/" not in handle and not raw.startswith(_ID_PREFIXES):
        seeds.setdefault(f"github:{handle}", "handle")
        seeds.setdefault(f"substack:{handle}", "handle")

    if raw.startswith("http") or "." in raw:
        for eid in (derive.blog_entity_id(raw), derive.substack_entity_id(None, raw)):
            if not eid.endswith(":unknown"):
                seeds.setdefault(eid.lower(), "home_url")

    # One query for all three shapes; the `profile` arm covers X and is simply false on a
    # handle-less profile.
    ph = ",".join("?" for _ in seeds) or "NULL"
    rows = conn.execute(
        f"SELECT entity_id, canonical_id FROM entities "
        f"WHERE lower(entity_id) IN ({ph}) "
        f"   OR lower(json_extract(profile, '$.handle')) = ?",
        [*seeds.keys(), handle],
    ).fetchall()
    if not rows:
        return []

    out, seen_heads = [], set()
    for r in rows:
        head = schema.current_canonical(conn, r["canonical_id"] or r["entity_id"])
        if head in seen_heads:            # two seeds landing in one cluster = one person
            continue
        seen_heads.add(head)
        members = [{"entity_id": m["entity_id"], "name": m["name"], "profile": m["profile"]}
                   for m in schema.entities_for_canonical(conn, head)]
        name, x_handle = derive.best_name(members)
        out.append({
            "canonical_id": head,
            "name": name,
            "handle": x_handle,
            "who_ids": sorted(m["entity_id"] for m in members),
            "matched_on": seeds.get(r["entity_id"].lower(), "x_profile_handle"),
        })
    return sorted(out, key=lambda c: c["canonical_id"])


def author_ids(conn, who: str | None, who_id: str | list[str] | None):
    """Fold a `who` handle and any explicit `who_id`s into one author-id list (a union).
    Returns `(ids, resolved)`; `resolved` is None when no `who` was passed. A `who` that resolves
    to nobody yields `[]` ("match nothing"), never None."""
    ids = ([who_id] if isinstance(who_id, str) else list(who_id or [])) if who_id else []
    if not who:
        return who_id, None
    resolved = resolve_who(conn, who)
    for cand in resolved:
        ids.extend(cand["who_ids"])
    return (sorted(set(i for i in ids if i)) if ids else []), resolved
