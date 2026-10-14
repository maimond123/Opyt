"""
mcp_server/offers.py — at most one offer per answer: one thing the user's own store can do for
them that they are not doing now, backed by a measurement and said in one line after the answer.

OPYT's features are found by asking, and a user who does not know them never asks. They have
to be shown what they can do, and not just once. Which answers may carry
an offer, and which already ask something, is `rider_rules`' decision; this module picks the
offer.

  • gather — the history (`offers.json`) and the measurements this tool's offers need;
  • decide — `pick`: of this call's offers whose measurement clears its bar (twice the bar for a
    feature the user has used), and that do not lead to what the call is already doing, the one
    shown least recently, never the one shown last (reader view excepted, `STANDING`). Plain
    values in, one offer or None out;
  • act — stamp what was shown. `note_use` stamps the feature a call used.

Until the user has made a reader view, its offer is pushed (`push`): ahead of the rotation, and
said on the first turns of a conversation, not once.

No clocks: within a conversation the host remembers what it offered and what was declined;
the server rotates by offer, whatever the subject. Keyed on the subject too, the circle offer won
all ten searches of a conversation, and the host would have dropped it after the first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from mcp_server.rider_rules import Call
from opyt_core.paths import opyt_path
from pipeline.kb import derive, schema
from pipeline.oracles import mentions, people_map, probe_store, ranking, roster
from pipeline.retrieval import corpus_census
from pipeline.retrieval.rank import content_words, fts_phrase
from pipeline.topics import store as topics_store

_HISTORY = "offers.json"

# What each offer's measurement must reach; starting values, to be tuned from measurement. A feature the user has used needs twice its bar: it returns only on a strong fit.
BARS = {
    "circle": 5,                        # sampled people whose posts hold the subject's phrase
    "topic": 10,                        # items on a subject no followed topic covers yet
    "people": 5,                        # sampled people on the subject, or waiting to be screened
    "engages": 5,                       # people one Oracle's posts name
    "oracle": 2,                        # their posts the user saved
    "map": people_map.MIN_PEOPLE,       # sampled people: below this the map has nothing to group
    "reader_view": 3,                   # hits a search answered with: sources enough for one
}

# The feature each offer leads to. No offer is made on a call that already uses its feature, and
# once the user has used it, it needs a strong fit.
FEATURE = {"circle": "circle", "topic": "topic", "people": "people",
           "engages": "engages", "oracle": "people", "map": "map",
           "reader_view": "reader_view"}

# Offers about the answer itself, not its subject. A host runs several searches for one answer
# and the rotation spends an offer on the first, so one of these rides again when no other offer
# is due: the searches behind an answer all carry it, and the host says it once.
STANDING = frozenset({"reader_view"})

# A standing offer the user has never taken is pushed: it leads the menu, and the host says it on
# its first turns of a conversation, not once, because said once it can be missed.
# The host counts the turns (`PUSH_NOTE`): the server sees searches, not turns.

# The cap is in the text: a list handed to a host becomes paragraphs.
HOST_NOTE = ("Put `say` to the user as one line, last, after your answer, in your own words: no "
             "tool names, no explanation, no list. At most once in this conversation: if you "
             "already made this offer here, or the user said no to it, leave it out. If they say "
             "yes, do `if_yes`.")
PUSH_NOTE = ("Put `say` to the user as one line, last, after your answer, in your own words: no "
             "tool names, no explanation, no list. Say it on each of the first three turns in "
             "this conversation whose answer carries it, even though you said it the turn "
             "before: said once, it can be missed. After the third, or once the user has said "
             "no to it, leave it out. If they say yes, do `if_yes`.")


@dataclass(frozen=True)
class Offer:
    name: str
    subject: str | None
    measure: int
    bar: int
    say: str
    if_yes: str


@dataclass(frozen=True)
class Author:
    """The person behind an opened or saved post."""

    name: str
    reference: str | None   # what `add_oracle` takes for them
    oracle: str | None      # their Oracle label, when the post is an Oracle's X post
    is_oracle: bool
    saved: int              # their posts the user saved
    named: int              # people their posts name, when an Oracle


@dataclass(frozen=True)
class Facts:
    """What `menu` measures against, gathered for one call; each tool fills what its offers need."""

    subject: str | None = None
    hits: int = 0                               # the hits a search answered with
    items: int = 0                              # atoms holding the subject, or the whole store
    people: int = 0                             # sampled people holding it, or everyone sampled
    followed: frozenset[str] = frozenset()      # followed topics' names and words, lowercased
    waiting: int = 0                            # people waiting to be screened
    author: Author | None = None
    top_oracle: tuple[str, int] | None = None   # the Oracle whose posts name the most people


@dataclass(frozen=True)
class History:
    shown: tuple[str, ...] = ()                 # offers by when last shown, most recent last
    used: frozenset[str] = frozenset()          # every feature the user has used


# ── decide ──────────────────────────────────────────────────────────────────────

def feature_of(call: Call) -> str | None:
    """The catalog feature a call uses, or None."""
    args = call.args
    match call.tool, args.get("action"):
        case "watch", _:
            return "topic"
        case "oracle", "candidates":
            return "engages" if args.get("oracle") else ("circle" if args.get("query") else None)
        case "oracle", "confirm":
            return "people"
        case "oracle", "map":
            return "map"
        case "add_oracle", _:
            return "people" if args.get("confirm") else None
        case "reader_view", _:
            return "reader_view"
    return None


def pick(offers: list[Offer], doing: str | None, history: History) -> Offer | None:
    """Of the offers that clear their bar and do not lead to what the call is `doing`, the one
    never shown, else the one shown least recently, in `offers`' order; never the one shown
    last, unless it is a standing offer and no other is due."""
    fit = _fits(offers, doing, history)
    last = history.shown[-1] if history.shown else None
    due = [o for o in fit if o.name != last]
    recency = {name: i for i, name in enumerate(history.shown)}
    return min(due, key=lambda o: recency.get(o.name, -1),
               default=next((o for o in fit if o.name in STANDING), None))


def _fits(offers: list[Offer], doing: str | None, history: History) -> list[Offer]:
    return [o for o in offers
            if FEATURE[o.name] != doing
            and o.measure >= o.bar * (2 if FEATURE[o.name] in history.used else 1)]


def push(offers: list[Offer], doing: str | None, history: History) -> Offer | None:
    """The standing offer to push on this answer, ahead of `pick`'s rotation; None when none
    fits or the user has used its feature."""
    return next((o for o in _fits(offers, doing, history)
                 if o.name in STANDING and FEATURE[o.name] not in history.used), None)


def menu(call: Call, facts: Facts) -> list[Offer]:
    """This call's offers, the most fitting first, each with its measurement."""
    match call.tool:
        case "search":
            subject = _subject_offers(facts, ("circle", "topic", "people")) if facts.subject else []
            return [*subject, _reader_view_offer(facts)]
        case "aggregate":
            return store_offers(facts)
        case "open" | "hopper":
            return _author_offers(facts.author)
        case "oracle":
            return _people_offers(facts)
    return []


def _subject_offers(f: Facts, names: tuple[str, ...]) -> list[Offer]:
    s = f.subject
    made = {
        "circle": Offer(
            "circle", s, f.people, BARS["circle"],
            f"{f.people} people you read have posted about {s}. Want to hear where they agree "
            f"and where they split?",
            f"oracle(action='candidates', query={s!r}), then sum up where those people agree "
            f"and split, naming each"),
        "topic": Offer(
            "topic", s, 0 if s.strip().lower() in f.followed else f.items, BARS["topic"],
            f"You have {f.items} items on {s}. Want to be told when something new on it "
            f"appears, from new papers and repos and from your own people?",
            f"write a topic on {s} as `watch` describes, show the user watch(action='preview') "
            f"and create it once they agree"),
        "people": Offer(
            "people", s, f.people, BARS["people"],
            f"{f.people} people you read write about {s}. Want to see which are worth making "
            f"Oracles?",
            f"oracle(action='candidates', query={s!r}); the user picks; "
            f"oracle(action='confirm', canonical_ids=[…])"),
    }
    return [made[name] for name in names]


def _reader_view_offer(f: Facts) -> Offer:
    return Offer("reader_view", None, f.hits, BARS["reader_view"],
                 "Want this answer and every source behind it on one page to read?",
                 "reader_view(atom_ids=[the atom_id of each item you cited, in the order you "
                 "cited them], title=<a short plain title>, answer=<your answer, in markdown>)")


def _map_offer(f: Facts) -> Offer:
    return Offer("map", None, f.people, BARS["map"],
                 f"The {f.people} people you read fall into a handful of circles by what they "
                 f"write. Want the map, and where your Oracles sit in it?",
                 "oracle(action='map')")


def store_offers(f: Facts) -> list[Offer]:
    """The whole store's offers, for `aggregate` and the new-material tour: a topic to follow
    while none is followed, then pick people, then the map."""
    return [
        Offer("topic", None, 0 if f.followed else 1, 1,
              "You don't follow any topic yet. Name one and you're told when something new on "
              "it appears: new papers and repos, and your own people's posts on it.",
              "ask which subject, write a topic on it as `watch` describes, show the user "
              "watch(action='preview') and create it once they agree"),
        Offer("people", None, f.waiting, BARS["people"],
              f"{f.waiting} people you follow are waiting to be picked as Oracles. Want to see "
              f"them?",
              "oracle(action='screen')"),
        _map_offer(f),
    ]


def _author_offers(a: Author | None) -> list[Offer]:
    if a is None:
        return []
    if a.oracle:
        return [Offer("engages", a.oracle, a.named, BARS["engages"],
                      f"{a.oracle}'s posts name {a.named} people. Want to see who they engage "
                      f"with most?",
                      f"oracle(action='candidates', oracle={a.oracle!r})")]
    if a.is_oracle:
        return []
    return [Offer("oracle", a.name, a.saved, BARS["oracle"],
                  f"You've saved {a.saved} posts by {a.name}. Want everything they write?",
                  (f"add_oracle(reference={a.reference!r}), preview it with the user, then "
                   f"confirm" if a.reference else
                   "add_oracle(reference=<their handle or home page>), preview it with the "
                   "user, then confirm"))]


def _people_offers(f: Facts) -> list[Offer]:
    """The screen's and candidates' offers, in order: the map, who an Oracle engages with,
    then asking the circle about a subject."""
    out = [_map_offer(f)]
    if f.top_oracle:
        label, n = f.top_oracle
        out.append(Offer("engages", label, n, BARS["engages"],
                         f"{label}'s posts name {n} people. Want to see who they engage with "
                         f"most?",
                         f"oracle(action='candidates', oracle={label!r})"))
    out.append(Offer("circle", None, f.people, BARS["circle"],
                     f"You can ask what the {f.people} people you read think about any subject. "
                     f"Want to try one?",
                     "ask which subject, then oracle(action='candidates', query=<it>) and sum up "
                     "where they agree and split, naming each"))
    return out


# ── gather ──────────────────────────────────────────────────────────────────────

def _gather(conn, call: Call, answer: dict) -> Facts:
    match call.tool:
        case "search":
            subject = call.args["query"]
            hits = len(answer.get("hits") or [])
            # Only common words name no subject: a host lists one person's posts with "the".
            if not content_words(subject):
                return Facts(hits=hits)
            return Facts(subject=subject, hits=hits,
                         items=corpus_census.subject_count(conn, subject),
                         people=probe_store.count_people_matching(conn, fts_phrase(subject)),
                         followed=_followed(conn))
        case "aggregate":
            return _store_facts(conn)
        case "open":
            return Facts(author=_author(conn, answer["who_id"]))
        case "hopper":
            row = conn.execute("SELECT who_id FROM atoms WHERE atom_id = ?",
                               (answer["atom_id"],)).fetchone()
            return Facts(author=_author(conn, row[0]) if row else None)
        case "oracle":
            counts = mentions.named_counts(mentions.named_by_oracles(conn))
            return Facts(people=len(probe_store.probed_who_ids(conn)),
                         top_oracle=max(counts.items(), key=lambda kv: kv[1], default=None))
    return Facts()


def _store_facts(conn) -> Facts:
    return Facts(items=schema.count_atoms(conn), waiting=ranking.waiting_count(conn),
                 followed=_followed(conn), people=len(probe_store.probed_who_ids(conn)))


def _followed(conn) -> frozenset[str]:
    return frozenset(t.strip().lower() for topic in topics_store.topics(conn)
                     for t in (topic["name"], *topic["words"]))


def _author(conn, who_id: str | None) -> Author | None:
    if not who_id:
        return None
    canonical = schema.current_canonical(conn, who_id)
    rows = [dict(r) for r in schema.entities_for_canonical(conn, canonical)]
    name, handle = derive.best_name(rows)
    label = mentions.oracle_authors(conn)["authors"].get(who_id)
    return Author(
        name=name or handle or who_id,
        reference=f"@{handle}" if handle else None,
        oracle=label,
        is_oracle=canonical in {o["canonical_id"] for o in roster.confirmed_oracles(conn)},
        saved=corpus_census.saved_count(conn, [r["entity_id"] for r in rows]),
        named=(mentions.named_counts(mentions.named_by_oracles(conn)).get(label, 0)
               if label else 0))


def _read_history() -> History:
    """An unreadable history reads as empty: a repeated offer beats a broken call."""
    try:
        raw = json.loads(opyt_path(_HISTORY).read_text())
        return History(shown=tuple(raw.get("shown") or ()), used=frozenset(raw.get("used") or ()))
    except Exception:
        return History()


# ── act ─────────────────────────────────────────────────────────────────────────

def offer(conn, call: Call, answer: dict) -> dict | None:
    """This answer's offer, or None. Stamps what it hands over."""
    history = _read_history()
    offers, doing = menu(call, _gather(conn, call, answer)), feature_of(call)
    pushed = push(offers, doing, history)
    if pushed is not None:
        return _hand_over(pushed, history, PUSH_NOTE)
    return _hand_over(pick(offers, doing, history), history)


def tour(conn) -> dict:
    """The whole-store offer the new-material tour carries, or `{}`."""
    history = _read_history()
    return _hand_over(pick(store_offers(_store_facts(conn)), None, history), history) or {}


def note_use(call: Call) -> None:
    """Stamp the feature this call used, so its offers need a strong fit from now on."""
    feature = feature_of(call)
    if feature is None:
        return
    history = _read_history()
    if feature not in history.used:
        _write_history(replace(history, used=history.used | {feature}))


def _hand_over(chosen: Offer | None, history: History,
               host_note: str = HOST_NOTE) -> dict | None:
    if chosen is None:
        return None
    shown = tuple(name for name in history.shown if name != chosen.name) + (chosen.name,)
    _write_history(replace(history, shown=shown))
    return {"say": chosen.say, "if_yes": chosen.if_yes, "host_note": host_note}


def _write_history(history: History) -> None:
    """A failed write costs a repeated offer, never the call."""
    try:
        path = opyt_path(_HISTORY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"shown": list(history.shown),
                                    "used": sorted(history.used)}))
    except Exception:
        pass
