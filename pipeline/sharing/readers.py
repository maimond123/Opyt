"""
pipeline/sharing/readers.py — which reader a person means, out of the roster the service lists,
and what cutting that one reader off says.

Plain values in, plain values out: the token rows come from `GET /v1/tokens` on the Mac and from
`tokens.list_tokens` on the connector, and nothing here calls the service. Both `unshare`s, the
Mac's and the connector's, pick and word a cut-off here, so they cannot drift apart.

An agent's key is in the roster with the readers' and is cut off the same way, by its name. It is
marked (`agent`) wherever it is listed, and its sentences say what an agent loses: it acted as
the owner, where a reader only read.
"""
from __future__ import annotations

from typing import NamedTuple


class Picked(NamedTuple):
    """The one reader a name matched."""
    row: dict


def readers_of(tokens: list[dict]) -> list[dict]:
    """Everyone the owner gave a key to, in a `GET /v1/tokens` list: readers and agents. The
    owner's own token is neither."""
    return [t for t in tokens if t["role"] in ("reader", "agent")]


def is_agent(reader: dict) -> bool:
    """Whether this key is an agent's, which acts as the owner, not a reader's."""
    return reader["role"] == "agent"


def short_id(reader: dict) -> str:
    """The 12-character hash prefix a reader can be named by. It exists because `for_whom` is
    optional: an unlabelled reader has no other handle, and without one it could be seen in a
    preview and never revoked."""
    return reader["token_sha256"][:12]


def name(reader: dict) -> str:
    """What to call a reader in a sentence: their label, or their short id."""
    return reader["label"] or short_id(reader)


def roster(readers: list[dict]) -> list[dict]:
    """Every reader, as the pair a person can actually name one by: the label their owner gave
    them at `share` time, and a short prefix of the hash. An agent also carries `agent: True`."""
    return [{"label": t["label"], "id": short_id(t), "since": t["created_at"],
             **({"agent": True} if is_agent(t) else {})} for t in readers]


def match(readers: list[dict], wanted: str) -> list[dict]:
    """The readers `wanted` names: a case-insensitive label, or a hash prefix of 8 characters or
    more. Several means ambiguous, and the caller refuses rather than picks."""
    want = wanted.strip()
    return [t for t in readers
            if want and ((t["label"] or "").strip().lower() == want.lower()
                         or (len(want) >= 8 and t["token_sha256"].startswith(want.lower())))]


def pick(held: list[dict], wanted: str) -> Picked | dict:
    """The one reader `wanted` names, or the answer that refuses: nobody by that name, or several.
    Two readers matching one name refuse rather than pick."""
    hits = match(held, wanted)
    if not hits:
        return _no_such_reader(held, wanted)
    if len(hits) > 1:
        return _ambiguous(hits, wanted)
    return Picked(hits[0])


def preview(hit: dict, held: list[dict], service: str) -> dict:
    """What cutting off `hit` does, before it is done. The served copy stays and everybody else
    keeps their access."""
    label, others = name(hit), len(held) - 1
    out = {"status": "preview",
           "scope": "one_reader",
           "reader_label": label,
           "reader_id": short_id(hit),
           "readers_remaining": others,
           "service": service}
    if is_agent(hit):
        return {**out, "agent": True, "consent": [
            f"This cuts off {label}, an agent that acts as the user, and nobody else. "
            f"{others} other(s) keep their access.",
            f"{label} can no longer search this knowledge base or change it. What it already "
            f"saved or removed stays as it is.",
            "The served copy of this knowledge base stays up.",
            f"The link {label} redeemed was already spent, so nothing it holds keeps working. "
            f"Letting it back in means a new link from `share` with `to='agent'`.",
        ]}
    return {**out, "consent": [
        f"This cuts off {label} and nobody else. {others} other(s) keep their access.",
        "The served copy of this knowledge base stays up.",
        f"The invite {label} redeemed was already spent, so nothing they hold keeps "
        f"working. Letting them back in means a new `share` link.",
    ]}


def revoked(hit: dict, held: list[dict], was_there: bool) -> dict:
    """The answer once the service has deleted `hit`'s token; `was_there` is whether a row went."""
    label = name(hit)
    if not was_there:
        # The row went between the read and the delete: another session revoked them, or the
        # owner unpublished. The end state is the one asked for, so this is not a failure.
        return {"status": "already_gone", "reader_label": label,
                "message": f"{label} no longer had access, so nothing changed."}
    out = {"status": "reader_revoked", "reader_label": label, "readers_remaining": len(held) - 1}
    if is_agent(hit):
        return {**out, "agent": True,
                "message": (f"{label} loses access on its very next request: it can no longer "
                            f"search the knowledge base or change it. Everyone else keeps their "
                            f"access.")}
    return {**out,
            "message": (f"{label} loses access on their very next request. The knowledge base is "
                        f"still shared with everyone else.")}


def _no_such_reader(held: list[dict], wanted: str) -> dict:
    return {"status": "no_such_reader",
            "readers": roster(held),
            "message": (f"nobody holding access to this knowledge base is called "
                        f"{wanted!r}. The current readers are listed in `readers` — pass a "
                        f"`label` or an `id` from that list. An unlabelled reader is one "
                        f"who was invited without a `for_whom`.")
            if held else
            "nobody holds access to this knowledge base, so there is no one to cut off."}


def _ambiguous(hits: list[dict], wanted: str) -> dict:
    return {"status": "ambiguous_reader",
            "readers": roster(hits),
            "message": (f"{wanted!r} names {len(hits)} of those who hold access, so this will "
                        f"not guess which one to cut off. Ask the user which, and pass the `id` "
                        f"of that one from `readers`.")}
