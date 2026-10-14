"""Read one X handle's profile identity (UserByScreenName)."""

import pipeline.platforms.x.records as x_records
from pipeline.platforms.x.request import XClient

# ── User profile (UserByScreenName; free cookie-scrape) ───────────────────────
# One handle -> the identity fields two callers need: `identify.fetch_x_identity` (which keys
# `x:user:{id}` entities on `rest_id`, the load-bearing field) and
# `discover_profile._probe_twitter_bio` (which mines the bio for a person's other homes).
USERBYSCREENNAME_OP = "UserByScreenName"
# Empty: `XClient.resolve_query_id` returns a non-empty seed without validating it, so a baked id
# survives rotation as a permanent 404.
DEFAULT_USERBYSCREENNAME_QID = ""


def _expand_tco(url: str, url_entities: dict) -> str:
    """A t.co short link -> the real destination, via the `urls` list X ships beside it. Returns
    `url` unchanged when nothing matches, so a caller never gets an empty string for a link that
    exists. X hands back the shortened form in `website.url` and the expansion only in the
    entities block, so this is a join, not a fallback."""
    for u in (url_entities.get("urls") or []):
        if u.get("url") == url and u.get("expanded_url"):
            return u["expanded_url"]
    return url


def fetch_user_profile(client: XClient, screen_name: str) -> dict | None:
    """One X handle -> `{user_id, handle, display_name, bio, website, bio_urls, verified,
    followers}`, or None when the account cannot be read.

    The response has no `legacy` block: the live paths are `core.*`, `profile_bio.*`,
    `relationship_counts.*`, `verification.*` and `website.url`. Returns None rather than raising
    `XUserUnavailable`, because neither caller distinguishes a suspended account from a failure.
    `website` and `bio_urls` are expanded past t.co; self-links to x.com are left for the caller
    to filter.
    """
    handle = (screen_name or "").strip().lstrip("@")
    if not handle:
        return None
    qid = client.resolve_query_id(USERBYSCREENNAME_OP,
                                  default_seed=DEFAULT_USERBYSCREENNAME_QID,
                                  env_var="X_USERBYSCREENNAME_QUERY_ID",
                                  page_url="https://x.com/home")
    data = client.graphql_get(USERBYSCREENNAME_OP, qid,
                              {"screen_name": handle, "withSafetyModeUserFields": True},
                              x_records.TIMELINE_FEATURES,
                              field_toggles={"withAuxiliaryUserLabels": True}, tolerate_errors=True)
    user = ((data.get("data") or {}).get("user") or {}).get("result") or {}
    # A handle that does not exist comes back as `{"data": {}}` — no `user` key at all — and a
    # suspended one as `__typename: "UserUnavailable"`. Both are "cannot read this account".
    if not user or user.get("__typename") == "UserUnavailable":
        return None
    uid = str(user.get("rest_id") or "").strip()
    if not uid:
        return None

    core_ = user.get("core") or {}
    bio = user.get("profile_bio") or {}
    entities = bio.get("entities") or {}
    counts = user.get("relationship_counts") or {}
    site = (user.get("website") or {}).get("url") or ""
    return {
        "user_id": uid,
        "handle": core_.get("screen_name") or handle,
        "display_name": core_.get("name") or "",
        "bio": bio.get("description") or "",
        "website": _expand_tco(site, entities.get("url") or {}),
        # The other homes a person lists (a Substack, a personal site); the caller classifies them.
        "bio_urls": [u.get("expanded_url") or u.get("url") or ""
                     for u in ((entities.get("description") or {}).get("urls") or [])
                     if u.get("expanded_url") or u.get("url")],
        # A blue check: `verification.verified` (legacy blue) or `is_blue_verified` (subscription).
        "verified": bool(user.get("is_blue_verified")
                         or (user.get("verification") or {}).get("verified")),
        "followers": counts.get("followers"),
    }
