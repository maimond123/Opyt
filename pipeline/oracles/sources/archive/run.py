"""Wire an archive's durable tasks to its adapter; shared quota policy owns retries. A document is
fetched first, then written with its task's completion in one transaction."""
import json
from dataclasses import dataclass, replace

from pipeline.kb import derive, ingest_github, schema
from pipeline.oracles.sources.inputs import image_cache, save_images
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop.retries import UNSAVED, may_run
from pipeline.worker.quota_loop.source import Changed, Item, Outcome, Unit
from pipeline.oracles import roster

from pipeline.oracles.sources.archive.adapter import Adapter
from pipeline.oracles.sources.archive.blog import Blog
from pipeline.oracles.sources.archive.github import Github
from pipeline.oracles.sources.archive.openalex import OpenAlex
from pipeline.oracles.sources.archive import state
from pipeline.oracles.sources.archive.substack import Substack

ADAPTERS: dict[str, Adapter] = {"blog": Blog(), "substack": Substack(), "github": Github(), "openalex": OpenAlex()}


@dataclass(frozen=True)
class Work:
    parent: Unit
    walk: state.Walk
    task: dict | None


def begin(conn, parent, source_type, url, *, author_name=None, since=None, handle=None, topics=None, **kwargs):
    """Called only for a source discovery was sure of or the user added. Sends no request: the
    walk's reads do. A GitHub account gets no entity here; its repos write it, under the login
    GitHub spells (`derive.derive_github`), and `resolve` joins it to the Oracle that added it."""
    oracle = parent.footprint.oracle
    if source_type == "github" and not ingest_github._github_owner(url):
        return None
    entity = (derive.blog_entity_id(url) if source_type == "blog" else
              derive.substack_entity_id(handle, url) if source_type == "substack" else None)
    referenced_repo = source_type == "github" and bool(ingest_github.github_owner_repo(url))
    # Identity writers commit internally, so they run before the guarded transaction that
    # creates owed work, never inside write_tx.
    if entity and roster.on_roster(conn, oracle):
        schema.upsert_entity(conn, entity, name=author_name, identity_links=[url])
    with write_tx(conn, site="archive.begin"):
        if not roster.on_roster(conn, oracle):
            return {"archive_started": True}
        spec = {"type": source_type, "url": url, "author": author_name, "handle": handle,
                "entity": entity, "since": since.isoformat() if since else None, "topics": topics,
                "entry_mode": "author_referenced" if referenced_repo else "oracle-footprint"}
        state.start(conn, parent.key, oracle, spec, ADAPTERS[source_type].initial(spec), utc_now())
    return {"archive_started": True, "added": 0}


def offer(conn, parent, now):
    walk = state.active(conn, parent.key)
    if walk is None:
        yield parent
        return
    tasks = state.pending(conn, walk)
    if not tasks:
        yield replace(parent, key=parent.key + "|finish", payload=Work(parent, walk, None))
    adapter = ADAPTERS[walk.spec["type"]]
    for task in tasks:
        if not may_run(task["retry_at"], task["failures"], now):
            continue
        data = json.loads(task["data"])
        item = Item(state.ROWS, (task["id"],), state.ROWS.read(task), parent.item.platform)
        yield replace(parent, key=f"{parent.key}|{task['id']}", payload=Work(parent, walk, task),
                      item=item, host=adapter.host(walk.spec, task["kind"], data))


def _write(conn, atom, text, *, work):
    work.parent.footprint.authors.add(atom["who_id"])
    old = conn.execute("SELECT entry_mode FROM atoms WHERE atom_id=?", (atom["atom_id"],)).fetchone()
    if old and atom["source_type"] != "github":
        return
    if old and old[0] == "user-saved":
        atom = {**atom, "entry_mode": "user-saved"}
    store_atom(conn, atom, text)


def _document_outcome(conn, result) -> Outcome:
    """What a document task came to once its atom is written."""
    if result in ("present", "skipped", "unchanged"):
        return Outcome(result)
    if conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (result,)).fetchone():
        return Outcome("saved")
    return Outcome(UNSAVED, failure=UNSAVED, detail="archive document did not save")


def fetch(conn, work, ctx):
    if not state.current(conn, work.walk) or not roster.on_roster(conn, work.walk.canonical_id):
        return None
    if work.task is None:
        return {"added": conn.execute("SELECT count(*) FROM archive_units WHERE walk=? AND generation=? "
                                     "AND kind='document' AND status='saved'",
                                     (work.walk.key, work.walk.generation)).fetchone()[0]}
    adapter, spec = ADAPTERS[work.walk.spec["type"]], work.walk.spec
    data = json.loads(work.task["data"])
    if work.task["kind"] == "page":
        return adapter.page(spec, data, ctx)
    images = image_cache(ctx)
    held: list[tuple[dict, str]] = []
    try:
        result = adapter.document(conn, spec, data, ctx, lambda atom, text: held.append((atom, text)))
    finally:
        save_images(images)
    with write_tx(conn, site="archive.document"):
        if ctx.lease_lost() or not state.current(conn, work.walk) or not roster.on_roster(conn, work.walk.canonical_id):
            raise Changed("archive changed or its Oracle was forgotten")
        for atom, text in held:
            _write(conn, atom, text, work=work)
        outcome = _document_outcome(conn, result)
        if outcome.failure is None:
            state.record_document(conn, work.walk, work.task, outcome.status)
    return outcome


def record(conn, work, fetched, finish):
    if fetched is None or not state.current(conn, work.walk) or not roster.on_roster(conn, work.walk.canonical_id):
        return Outcome("changed")
    if work.task is None:
        if state.pending(conn, work.walk):
            return Outcome("changed")
        result = finish(work, fetched)
        if result.failure is None:
            state.complete(conn, work.walk)
        return result
    if work.task["kind"] == "page":
        state.record_page(conn, work.walk, work.task, fetched)
        return Outcome("listed")
    return fetched
