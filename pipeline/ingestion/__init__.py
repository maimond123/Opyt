"""
OPYT Ingestion — source adapters, one module per source.

Import the submodule you need (``from pipeline.ingestion import x_graphql_core``), or the
Layer-1 fetch/render helpers under ``pipeline.ingestion.sources``.

This package deliberately re-exports NOTHING. The old ``sync_*`` / ``ingest_profile`` block had
no importers — every ``from pipeline.ingestion import X`` in the tree names a SUBMODULE — and an
eager re-export made importing one adapter pull in all of them.

═══════════════════════════════════════════════════════════════════════════════
This package is not the vault producer, and it is no longer a ``raw/`` producer
either. It survives as the atom rail's Layer-1 fetch/render library.
═══════════════════════════════════════════════════════════════════════════════

An earlier version of this header said the ``sync_*`` walks write "vault notes". They did not,
and had not since ``d1bd9e27`` — a belief that survived in four docstrings and cost a wrong
deletion plan on 2026-08-07.

The ``raw/`` rail is gone (2026-08-14), and the sequence is worth keeping because the failure
shape repeats. ``raw/`` had exactly one reader, ``pipeline/radar/annotate.py`` via
``pipeline/parse_raw.py::parse_all``. Both were deleted 2026-08-13 with the radar. For one day
after that, every ``sync_*`` here still ran, still reported a count, and wrote markdown into a
directory nothing read — a dead drop. The count is what let it survive: a status line reports
that a pull HAPPENED, never where it went.

So the writers went with the reader:

  • ``x_bookmarks_graphql.py``  — DELETED. Superseded by ``pipeline/kb/bookmark_catchup.py``,
    which lands the same free cookie-scraped bookmarks as ``entry_mode='user-saved'`` ATOMS.
  • ``ingest_substack.py``     — DELETED. Superseded by ``pipeline/kb/ingest_curation.py``
    (``sync_substack_saved`` + the Following list), which lands full-body content atoms.
  • ``x_twitterapi.py``        — RENAMED to ``x_render.py`` (2026-08-30) after twitterapi.io was
    removed and only the renderer survived. Its network layer, its credential and its
    ``sync_profile`` are all gone; the X transport is ``x_graphql_core.py``, and pulling an
    Oracle's timeline INTO the store is ``pipeline/kb/ingest_x_footprint.py``'s job.

``pipeline/config.py`` no longer has ``raw_path`` or ``raw_dir()``. A ``raw_path`` key left in an
old settings.yaml is inert. See
``docs/plans/2026-08-13-retire-the-sync-tool-and-the-raw-rail.md`` and the ``retired-raw-rail``
guard. ``opyt_home()/kb_raw/`` is UNRELATED and stays — it is the atom rail's snapshot store and
has live readers (``pipeline/kb/raw_store.py``).

Who writes vault notes. **Nothing. There is no vault producer left in this package.**

That is the milestone worth recording here, and it took three deletions across two days to reach.
This block once listed three writers: ``save_paper → {vault}/papers/`` and ``save_repo →
{vault}/repos/`` went 2026-08-13 (see the ``retired-vault-artifact-writers`` guard and
``docs/plans/2026-08-13-delete-save-paper-and-save-repo.md``), along with
``pipeline/processing/process_academic_papers.py``. Papers and repos become ATOMS now, via
``pipeline/kb/ingest_papers.py`` and the Frontier rail.

``claude_chats → {vault}/claude/`` was the last one standing, and it went the same day. It had
been deprecated and paused since 2026-08-07 — its crontab entry commented out — but paused is not
deleted, and a paused writer with a live code path is a writer. See the
``retired-claude-chat-ingestion`` guard for why it went and what the precondition for rebuilding
it is.

The vault directory is still not safe to delete, and no-writers-left is NOT the condition that
makes it safe — CONTENT is. ``{vault}/papers/`` holds ~187 semantic-scholar notes written by the
ORACLE-path ingester (a different writer, never save_paper), against 3 paper atoms in the live
store; plus 42 self-authored notes with no upstream to re-pull, and 25 orphaned synthesis notes.
Those need their own backfill decision.

What the atom rail uses this for. ``pipeline/kb/`` imports from this package 71 times across 21
of its 27 modules — Layer-1 fetch/render (``sources/``), ``utils.log``, ``url_canon``, and seven
X primitives from ``x_render`` (``tweet_to_markdown``, ``_parse_twitter_date``,
``_stitch_threads``, ``_article_shape``, ``_render_article``, ``_dedupe_tweets``,
``_article_tweet_id``) — all pure renderers now that the transport moved out. The
atom rail is built ON this package, not as a replacement for it, and it stores its own snapshots
in ``opyt_home()/kb_raw/`` — deliberately NOT ``{vault}/raw/`` (see ``pipeline/kb/raw_store.py``).

MEASURED 2026-08-07, and now HISTORICAL — the two live directories in that measurement are both
gone. Last write per vault directory then: ``raw/`` 8,818 files and ``claude/`` 374 files, both
live that day; every other note directory dormant (``x/`` 10,284 files last written 2026-07-09,
``github/`` / ``papers/`` / ``blog/`` 2026-07-05, ``substack/`` 2026-05-13). ``claude/``'s
producer went 2026-08-13 and ``raw/`` went 2026-08-14, so NOTHING in this package writes to disk
on its own account any more. It is a library the atom rail calls.

Before proposing a deletion here, check the vault directory's mtimes and this package's actual
write targets. A docstring is not evidence.
"""
