"""opyt_core/paths.py — the single source of truth for the OPYT data home.

Every file that needs a path under ``~/.opyt`` resolves it through here, so one
``export OPYT_HOME=/tmp/opyt-sim`` relocates the ENTIRE data home in lockstep:
the database, embeddings, ``.env``, settings, caches — everything. This is the
sandbox seam that lets the new-user setup flow run end-to-end without touching
the real corpus.

WHY one knob (``OPYT_HOME``) and not a per-file override like the old ``OPYT_DB``:
``opyt.db`` is only one file among many under the home (``signals.db``,
``settings.yaml``, ``.env``, ``run/``, embed caches, …). A file-level override
can't move the rest, so a half-overridden run writes some state to the sandbox
and some to the real home — silently. Deriving every path from a single home
directory makes partial sandboxing impossible: either all of it moves or none.

WHY a function and not a module constant: an env var read at import time is fine
for our use (OPYT_HOME is set in the shell before the process starts), but a
function also lets tests set the var between calls. Module-level constants in
callers that do ``X = opyt_path("...")`` still read the env at import — also fine.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".opyt"


def opyt_home() -> Path:
    """The OPYT data home. Honors ``$OPYT_HOME``; defaults to ``~/.opyt``.

    Pure: it resolves a path, it does NOT create the directory. Callers mkdir
    where they actually write (as they already do), so resolving a path has no
    side effects.
    """
    env = os.environ.get("OPYT_HOME")
    return Path(env).expanduser() if env else _DEFAULT_HOME


def opyt_path(*parts: str) -> Path:
    """A path under the OPYT home. ``opyt_path("opyt.db")`` -> ``<home>/opyt.db``."""
    return opyt_home().joinpath(*parts)


def opyt_db() -> Path:
    """The one SQLite DB (``<home>/opyt.db``) — the most-repeated, most-dangerous path."""
    return opyt_path("opyt.db")
