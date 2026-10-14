"""
pipeline/oracles/sources/threads/tail_state.py — a thread atom's tail state: which threads are
owed, what a page's group does about its stored tail, and the one writer of `thread_tail`.

The walk writes a thread from what Originals returned and marks its tail owed; the tail queue
(`x_threads.tails`) completes it. `decide` is the pure rule the timeline's writer follows for each
group. The writer (`ingest_x_footprint._GroupWriter`, library's) is handed this module as its
`tail_rules`, so library never imports oracles.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.platforms.x.conversation import is_self_reply

# A thread atom's tail state, `payload.thread_tail`. The walk writes a
# thread from what Originals returned — its root and one self-reply — as OWED; the tail queue
# (`x_threads.tails`) rewrites it FETCHED, or marks it GONE when the conversation no longer
# exists (final). Not `body_state = pending`: that means "stored WITHOUT a body", and
# `schema.load_body_pending` and `ingest_curation` re-fetch whatever carries it. A thread root has
# a body; it is short of its tail.
TAIL_OWED, TAIL_FETCHED, TAIL_GONE = "owed", "fetched", "gone"


def threads_owed(tweets: list[dict], author_id: str, *, pinned: dict | None = None) -> set[str]:
    """conversationIds whose tails the thread lane is owed:
    every conversation holding at least one self-reply by the author, plus the PINNED tweet's
    when it has replies.

    Originals ships a self-thread as its root plus ONE of the author's own replies, so a
    self-reply in the walk is the sign there is a tail to fetch. Against fetching every root with any replies,
    measured over 257 roots on three accounts, this takes half the requests (160 vs 328) and has one miss — the pin. Originals ships the pinned
    tweet only in its `TimelinePinEntry`, never in its chronological slot, so the module that
    would carry its self-reply never appears. `pinned` is that tweet, passed only by the walk whose
    range holds its date (`walk_page`'s `pin_here`); owing it costs at most one thread per range."""

    owed = {str(t["conversationId"]) for t in tweets
            if t.get("conversationId") and is_self_reply(t, str(author_id))}
    if pinned and (pinned.get("replyCount") or 0) > 0:
        owed.add(str(pinned.get("conversationId") or pinned["id"]))
    return owed


@dataclass(frozen=True)
class TailDecision:
    """What one page's group does about its thread's tail.

    `skip`        — the stored atom already holds these replies, or is owed: nothing to write.
    `mark_owed`   — replies to a stored thread without its root: keep the body, mark it owed.
    `keep_stored` — the stored tail holds more than this group: don't rewrite the body.
    `tail`        — the tail state the atom is written with (None: not a thread owing a tail)."""
    skip: bool = False
    mark_owed: bool = False
    keep_stored: bool = False
    tail: str | None = None

    @property
    def owes(self) -> bool:
        return self.tail == TAIL_OWED


# What the tail queue writes: a whole thread, its tail fetched.
FETCHED = TailDecision(tail=TAIL_FETCHED)


def decide(prior: dict | None, *, conv: str, ids: list[str], has_root: bool, stored: bool,
           owed_convs: set[str] | frozenset) -> TailDecision:
    """A page's group against what the store holds for its thread (`prior`: the stored
    `thread_tail` and `thread_ids`, or None).

    | case                                                              | result      |
    |-------------------------------------------------------------------|-------------|
    | no root, stored, prior owed                                       | skip        |
    | no root, stored, prior fetched/gone and holds these ids           | skip        |
    | no root, stored, otherwise                                        | mark_owed   |
    | prior fetched/gone and holds these ids                            | keep_stored |
    | a self-reply here, no root, prior owed, or prior holds more ids   | tail owed   |
    | otherwise                                                         | no tail     |
    """
    prior_ids = {str(i) for i in prior["thread_ids"]} if prior is not None else set()
    holds_these = (prior is not None and prior["thread_tail"] in (TAIL_FETCHED, TAIL_GONE)
                   and set(ids) <= prior_ids)
    if not has_root and stored:
        # The thread is stored and this page holds only replies to it. Rewriting from the
        # replies alone would replace the thread with its tail, so the body stays as it is
        # and the thread is marked owed, unless what is stored already holds these replies.
        # The tail queue (`x_threads.tails`) then writes it whole.
        if holds_these or (prior is not None and prior["thread_tail"] == TAIL_OWED):
            return TailDecision(skip=True)
        return TailDecision(mark_owed=True, tail=TAIL_OWED)
    # A stored thread whose tail was already fetched (or found gone) holds MORE than the walk
    # can see. Rewriting it from Originals would shorten it, flip it back to owed and rewrite
    # it — on every pass that re-reads its range, the forward refresh's overshoot page
    # included. Only a self-reply the stored atom does not hold reopens it.
    if holds_these:
        return TailDecision(keep_stored=True)
    # Owed when this page shows a self-reply, when the group lacks its root (its replies
    # were on another page), or when the stored atom holds posts this group does not —
    # the replies a newer page wrote, which a rewrite from here would drop until the tail
    # queue puts them back.
    owes = (conv in owed_convs or not has_root
            or (prior is not None and (prior["thread_tail"] == TAIL_OWED
                                       or not prior_ids <= set(ids))))
    return TailDecision(tail=TAIL_OWED if owes else None)


def set_tail_state(conn, atom_id: str, state: str, *, body: str | None = None,
                   root: str | None = None, author: str | None = None) -> None:
    """Move one atom's `thread_tail` without rewriting its body — the one writer of the key
    outside a whole-atom write: GONE or FETCHED when the queue finishes a thread
    (`x_threads.state.finish`), and OWED when a page shows self-replies to a stored post whose
    page it is not (`root` and `author` are what the queue fetches by). The caller commits."""
    sets = ["'$.thread_tail', ?"]
    args: list = [state]
    if root is not None:
        sets.append("'$.thread_root', ?")
        args.append(root)
    if author is not None:
        sets.append("'$.thread_author', ?")
        args.append(author)
    if body is not None:
        sets.append("'$.body_state', ?")
        args.append(body)
    conn.execute(f"UPDATE atoms SET payload = json_set(payload, {', '.join(sets)}) "
                 f"WHERE atom_id = ?", (*args, atom_id))
