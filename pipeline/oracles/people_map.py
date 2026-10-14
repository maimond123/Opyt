"""
pipeline/oracles/people_map.py — the people map, `oracle(action='map')`: the people the user reads,
grouped by what they write, and which of the user's Oracles sit in each group.

Words only, computed here: TF-IDF over each sampled person's posts (`probe_store`), reduced
through the people-by-people similarity, then grouped by spherical k-means. Words gave crisper groups than vectors in
measurement and need no embedding, so both tiers take this one path. The scan never samples a
confirmed Oracle, so each Oracle is placed from their
own posts, projected onto the same words.

Nothing here names or judges a group. The host names each from its `words` and `central` people
and leaves out a group with no clear subject: no model call of OPYT's own decides it,
the rule the screen keeps for organisations.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter

import numpy as np

from pipeline.oracles import probe_store, roster

# Below this many sampled people a map is a list, not groups.
MIN_PEOPLE = 50
# At most this many groups, and about this many people per group below that.
MAX_GROUPS = 10
PEOPLE_PER_GROUP = 25
# A word must be used by at least this many people, and by at most this share of them: rarer is
# noise, commoner names no group.
MIN_USERS = 5
MAX_SHARE = 0.3
# The words kept, most widely used first. Bounds the people-by-words matrix (float32) on a large
# store: 1,142 people by 10,000 words is 46 MB.
MAX_WORDS = 10_000
# Dimensions kept from the people-by-people similarity, k-means rounds, and seeded runs.
DIMS = 200
ROUNDS = 50
RESTARTS = 10
# How many of an Oracle's most recent items place them.
ORACLE_ITEMS = 400
# Shown per group.
WORDS_SHOWN = 8
CENTRAL_SHOWN = 5

_STRIP = re.compile(r"https?://\S+|\[[^\]]*\]\([^)]*\)|@\w+")
_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = frozenset(
    "the a an and or of to in on for is it this that with as at by be are was i you we they my "
    "our your from not but have has just so if what all can will more about like do no one get "
    "how its it's im i'm don't me he she his her their them out up new now than then there here "
    "when who which would been were also into only very much any some these those amp rt".split())

NOTE = ("Name each group in a few plain words from its `words` and `central` people. Leave out any "
        "group with no clear subject (chatter, politics, one language): do not mention it. For "
        "each group you keep, give its name, the people central to it, and which of the user's "
        "Oracles sit in it, saying plainly when none do: that is where a new Oracle would add "
        "the most. Nothing about how the map was made.")


def people_map(conn: sqlite3.Connection) -> dict:
    """action='map' — gather the texts, group them, and describe each group. See `oracle()`."""
    texts = probe_store.texts_by_person(conn)
    if len(texts) < MIN_PEOPLE:
        return {"people": len(texts), "groups": [],
                "note": (f"Only {len(texts)} people have been sampled so far, too few to group. "
                         f"Say so in one line; the background scan samples more over the "
                         f"following days.")}
    oracles = _oracle_texts(conn)
    people = sorted(texts)
    groups, placed = group([texts[p] for p in people], list(oracles.values()))
    names = _names(conn, people)
    labels = list(oracles)
    out = []
    for g in groups:
        out.append({"people": len(g["members"]),
                    "words": g["words"],
                    "central": [names.get(people[i]) or people[i] for i in g["central"]],
                    "oracles": [labels[o] for o, home in enumerate(placed) if home == g["id"]]})
    answer = {"people": len(people), "groups": out, "note": NOTE}
    unplaced = [labels[o] for o, home in enumerate(placed) if home is None]
    if unplaced:
        answer["oracles_not_placed"] = unplaced
    return answer


# ── decide ──────────────────────────────────────────────────────────────────────

def tokens(texts: list[str]) -> list[str]:
    """One person's words: lowercased, links, mentions and stop words dropped."""
    return [w for t in texts for w in _WORD.findall(_STRIP.sub(" ", t).lower()) if w not in _STOP]


def group(people: list[list[str]], oracles: list[list[str]]) -> tuple[list[dict], list[int | None]]:
    """Each person's texts → groups, largest first, each `{id, members, central, words}` with
    member and central indexes into `people`; and each Oracle's group id, or None for an Oracle
    with no words in common with anyone. Deterministic: the same texts give the same groups."""
    docs = [Counter(tokens(t)) for t in people]
    n = len(docs)
    used_by = Counter(w for d in docs for w in d)
    vocab = [w for w, c in used_by.most_common() if MIN_USERS <= c <= MAX_SHARE * n][:MAX_WORDS]
    idf = {w: math.log(n / used_by[w]) for w in vocab}
    index = {w: i for i, w in enumerate(vocab)}
    X = _weigh(docs, index, idf)

    # The people-by-people similarity is n×n, far smaller than people-by-words: its leading
    # eigenvectors give the same reduction an SVD of X would.
    values, vectors = np.linalg.eigh(X @ X.T)
    top = np.argsort(-values)[:min(DIMS, n - 1)]
    scale = np.sqrt(np.maximum(values[top], 1e-12))
    P = _unit(vectors[:, top] * scale)

    k = max(2, min(MAX_GROUPS, n // PEOPLE_PER_GROUP))
    labels, centres = _kmeans(P, k)

    # An Oracle projects onto the same words and dimensions: (x · Xᵀ) U / S.
    O = _weigh([Counter(tokens(t)) for t in oracles], index, idf)
    placed = []
    for row in O:
        if not row.any():
            placed.append(None)
            continue
        placed.append(int((_unit((row @ X.T) @ vectors[:, top] / scale) @ centres.T).argmax()))

    groups = []
    for j in np.argsort(-np.bincount(labels, minlength=k), kind="stable"):
        members = np.where(labels == j)[0]
        if not len(members):
            continue
        central = members[np.argsort(-(P[members] @ centres[j]), kind="stable")[:CENTRAL_SHOWN]]
        words = np.argsort(-X[members].mean(0), kind="stable")[:WORDS_SHOWN]
        groups.append({"id": int(j), "members": members.tolist(), "central": central.tolist(),
                       "words": [vocab[w] for w in words]})
    return groups, placed


def _weigh(docs: list[Counter], index: dict[str, int], idf: dict[str, float]) -> np.ndarray:
    """TF-IDF rows, length-normalised: (1 + log tf) · idf."""
    X = np.zeros((len(docs), len(index)), np.float32)
    for r, d in enumerate(docs):
        for w, c in d.items():
            if w in index:
                X[r, index[w]] = (1 + math.log(c)) * idf[w]
    return _unit(X)


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


def _kmeans(P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means, the tightest of `RESTARTS` seeded runs, so a repeated map gives the same
    groups. One run can settle with two centres in one subject and two subjects sharing a centre,
    and nothing moves it out."""
    runs = [_kmeans_once(P, k, np.random.default_rng(seed)) for seed in range(RESTARTS)]
    return max(runs, key=lambda run: float((P * run[1][run[0]]).sum()))


def _kmeans_once(P: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One run from k-means++ starts: each next start drawn away from the ones already picked."""
    picked = [P[rng.integers(len(P))]]
    for _ in range(1, k):
        far = np.maximum(1 - (P @ np.stack(picked).T).max(1), 0) ** 2
        picked.append(P[rng.choice(len(P), p=far / far.sum())])
    centres = np.stack(picked)
    labels = np.zeros(len(P), int)
    for _ in range(ROUNDS):
        labels = (P @ centres.T).argmax(1)
        centres = _unit(np.stack([P[labels == j].mean(0) if (labels == j).any() else centres[j]
                                  for j in range(k)]))
    return labels, centres


# ── gather ──────────────────────────────────────────────────────────────────────

def _oracle_texts(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Each confirmed Oracle's most recent item texts, by display name."""
    out = {}
    for o in roster.confirmed_oracles(conn):
        ids = [m["entity_id"] for m in o["members"]]
        name = o["name"] or roster.name_for(conn, o["canonical_id"]) or o["canonical_id"]
        out[name] = [r[0] or "" for r in conn.execute(
            "SELECT b.text FROM bodies b JOIN atoms a ON a.atom_id = b.atom_id "
            "WHERE a.who_id IN (%s) ORDER BY a.when_ts DESC LIMIT ?" % ",".join("?" * len(ids)),
            [*ids, ORACLE_ITEMS])]
    return out


def _names(conn: sqlite3.Connection, who_ids: list[str]) -> dict[str, str]:
    wanted = set(who_ids)
    return {r[0]: r[1] for r in conn.execute("SELECT entity_id, name FROM entities "
                                             "WHERE name IS NOT NULL") if r[0] in wanted}
