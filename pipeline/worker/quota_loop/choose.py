"""
pipeline/worker/quota_loop/choose.py — which unit a quota loop runs next.

The rules:
  • three classes, strict: YOURS › NEW_POSTS › SCOUTING;
  • sources in one class take turns, one unit each;
  • except that new posts take every 10th unit while any are owed, so a big ask cannot hold back
    every other Oracle's new posts for the hours it takes;
  • a source with a `floor` (scouting's 50) runs only while the quota's meter shows more than that
    left, or has never been read. That is the only reserve: your work and new posts may spend a
    window to zero.
Inside a source, which of its units is most urgent is the source's own business.

A loop holds one `Rotation`, which remembers whose turn it is. Every executor counts a turn when
it admits a unit (`turn`), so a unit that ends in a wait has had its turn, and the pool's
completion order cannot distort the classes' shares.
"""

from __future__ import annotations

from typing import Callable, Iterable, NamedTuple

from .source import NEW_POSTS, SCOUTING, YOURS, Source, Unit

NEW_POSTS_EVERY = 10            # one unit in ten goes to new posts while any are owed


class Picked(NamedTuple):
    """An admitted unit and its source, which travel together through fetch and settlement."""
    source: Source
    unit: Unit

    @property
    def key(self) -> tuple[str, str]:
        """`(source, unit key)`: the backstop's strikes and the pool's running units key on it."""
        return (self.source.name, self.unit.key)


def skipped(floor: int, meter: tuple[int, float] | None,
            strike_until: float | None, now: float) -> bool:
    """Keep a source's reserve, and leave a struck unit until its retry time."""
    return bool((floor and meter is not None and meter[0] <= floor)
                or (strike_until is not None and now < strike_until))


class Rotation:
    """Whose turn it is in each class, and how many of yours since new posts last ran."""

    def __init__(self) -> None:
        self._last_in_class: dict[int, str] = {}
        self._since_new_posts = 0

    def pick(self, sources: list[Source], units: Callable[[Source], Iterable[Unit]],
             meter: tuple[int, float] | None, strikes: dict[tuple[str, str], tuple[int, float]],
             now: float) -> Picked | None:
        """The next unit by class. `units` asks a source for its ready units; sources are asked
        lazily, so scouting's candidate list is not computed while your own work is owed.
        `strikes` is the loop's backstop: `(source, unit key)` → `(strikes, until)`."""
        def first(klass: int) -> Picked | None:
            return self._first_in_class(sources, units, meter, strikes, now, klass)

        yours = first(YOURS)
        if yours is not None:
            if self._since_new_posts >= NEW_POSTS_EVERY - 1:
                return first(NEW_POSTS) or yours
            return yours
        return first(NEW_POSTS) or first(SCOUTING)

    def turn(self, source: Source) -> None:
        """One scheduling turn: rotate the class, count the new-posts share."""
        self._last_in_class[source.klass] = source.name
        if source.klass == YOURS:
            self._since_new_posts += 1
        elif source.klass == NEW_POSTS:
            self._since_new_posts = 0

    def _first_in_class(self, sources: list[Source], units: Callable[[Source], Iterable[Unit]],
                        meter: tuple[int, float] | None,
                        strikes: dict[tuple[str, str], tuple[int, float]], now: float,
                        klass: int) -> Picked | None:
        """The first source in `klass` with a unit, starting after the one that ran last: that
        rotation is what makes the sources of one class take turns."""
        members = [s for s in sources if s.klass == klass]
        names = [s.name for s in members]
        last = self._last_in_class.get(klass)
        if last in names:
            i = names.index(last) + 1
            members = members[i:] + members[:i]
        for s in members:
            if skipped(s.floor, meter, None, now):
                continue
            for unit in units(s):
                picked = Picked(s, unit)
                strike = strikes.get(picked.key)
                if skipped(s.floor, meter, strike[1] if strike else None, now):
                    continue
                return picked
        return None
