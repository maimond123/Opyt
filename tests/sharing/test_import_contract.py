"""The sharing unit's import-direction contract (S3, S5) — the "Allowed imports" table of
docs/plans/2026-10-04-restructure-sharing.md, checked against the code.

Three things a reader of the unit relies on, each of which an edit can break without failing any
behaviour test:
  • every module imports only what its row allows, so every arrow points down and a job never
    reaches up into a flow;
  • nothing outside the unit imports it — and nothing in the install's half imports `service`,
    which the client wheel does not carry;
  • imports sit at the top of the file, and the two packages' `__init__.py` stay empty, so the
    import line names the file to open.

Relative imports are resolved, which is why this is a test and not a `.guards.py` rule: the guard
engine's `ast_import` does not see `from . import x`. Standard-library and third-party imports are
not policed here; only this repository's packages are.

Changing the table is changing the design: do it with the plan, not to make a red test green.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ("opyt_core", "pipeline", "service", "mcp_server", "gateway", "config")

# Code every unit may use that knows no domain (the system map's foundations).
FOUNDATIONS = {"opyt_core.paths", "opyt_core.config", "opyt_core.keys", "pipeline.credentials",
               "pipeline.ingestion.utils"}

S = "pipeline.sharing."
ALLOWED: dict[str, set[str]] = {
    "mcp_server/share_tools.py": {S + "share", S + "accept", S + "unshare"},
    "mcp_server/page_tools.py": {S + "page"},
    # The publishing pass, a source on the scrapers process (lanes slice 7). Each pass enrolls
    # the home in the worker's `line` job (docs/plans/2026-10-13-connector-writes.md §14).
    "pipeline/sharing/publishing.py": {S + "account", S + "client", S + "publish", S + "push_gate",
                                       "pipeline.worker.quota_loop.source",
                                       "pipeline.worker.job_queue", "pipeline.worker.registry"},
    # One `share` for every way in (docs/plans/2026-10-13-one-share.md, part 1): a person, the
    # user's agent, or their own claude.ai or ChatGPT, which is `link`'s Google sign-in.
    "pipeline/sharing/share.py": {S + "account", S + "client", S + "invite", S + "push_gate",
                                  S + "agent_invite", S + "link",
                                  "pipeline.retrieval.reads", "pipeline.worker.job_queue",
                                  "pipeline.worker.registry", "pipeline.retrieval.filters",
                                  "pipeline.kb.schema"},
    # The Google connection (docs/plans/2026-10-10-handoff-muse-connector.md, 2026-10-12).
    "pipeline/sharing/link.py": {S + "account", S + "client", S + "push_gate",
                                 S + "desktop_only",
                                 "pipeline.worker.job_queue", "pipeline.worker.registry"},
    "pipeline/sharing/accept.py": {S + "account", S + "client", S + "invite", "pipeline.retrieval.peers"},
    # A page of sources (docs/plans/2026-10-13-source-pages.md): items read as the store keeps them.
    "pipeline/sharing/page.py": {S + "account", S + "client", S + "page_words",
                                 "pipeline.kb.schema", "pipeline.retrieval.engine"},
    # What a reader view tells the host, worded once for the Mac and the connector
    # (docs/plans/2026-10-14-reader-view-answer.md).
    "pipeline/sharing/page_words.py": set(),
    # It also disconnects the Google account, which `link` words (part 1 of that plan).
    "pipeline/sharing/unshare.py": {S + "account", S + "client", S + "link", S + "push_gate",
                                    S + "readers"},
    "pipeline/sharing/account.py": set(),
    "pipeline/sharing/client.py": {S + "account", "opyt_core.service_http"},
    "pipeline/sharing/invite.py": {S + "account"},
    "pipeline/sharing/readers.py": set(),
    # What an agent invite means, worded once for the Mac's `share` and the connector's
    # (docs/plans/2026-10-13-connector-writes.md §14.7).
    "pipeline/sharing/agent_invite.py": set(),
    # What only the desktop app does, in one sentence, for `link` and the connector (part 3).
    "pipeline/sharing/desktop_only.py": set(),
    "pipeline/sharing/share_back.py": set(),
    "pipeline/sharing/push_gate.py": {"pipeline.kb.schema", "pipeline.timeparse",
                                      S + "fingerprint"},
    "pipeline/sharing/publish.py": {S + "account", S + "client", S + "export", S + "fingerprint",
                                    S + "delta", S + "push_gate"},
    "pipeline/sharing/export.py": {"pipeline.kb.schema", "pipeline.retrieval.engine"},
    # Changes-only uploads (docs/plans/2026-10-13-shared-copy-deltas.md): a copy's rows as
    # hashes, and the delta cut from an export and applied to the served copy.
    "pipeline/sharing/fingerprint.py": {S + "export"},
    "pipeline/sharing/delta.py": {S + "export", S + "fingerprint"},
    "service/app.py": {"service.access", "service.serving", "service.tokens", "service.grants",
                       "service.publications", "service.stats", "service.uploads",
                       "service.accounts", "service.connector", "service.deletion",
                       "service.google_device", "service.line", "service.library_writes",
                       "service.owner_tools",
                       "service.reading", "service.x_signin", "service.x_signin_page",
                       "service.pages.comments", "service.pages.make", "service.pages.render",
                       "service.pages.show", "service.pages.store",
                       "pipeline.hosted_signin.display",
                       "pipeline.retrieval.peers", S + "delta"},
    # X sign-in for an install with no browser of its own: the slot, its desktop, its page.
    "service/x_signin.py": {"service.x_signin_desktop"},
    "service/x_signin_desktop.py": {"pipeline.hosted_signin.display",
                                    "pipeline.platforms.browser.cdp",
                                    "pipeline.platforms.browser.profile"},
    "service/x_signin_page.py": {"opyt_core.web_panel"},
    # Named by log_config.json, not imported by the service.
    "service/log_filters.py": set(),
    # Every served read, from either way in: the caps and the meter.
    "service/reading.py": {"pipeline.retrieval.reads", "service.usage"},
    # The connector: an MCP entry over the caller's own library. Its `share` and `unshare` mint
    # and revoke on this service, worded like the Mac's (docs/plans/2026-10-13-connector-writes.md).
    # Its four library writes are entries to `library_writes`; `watch`'s description carries the
    # local tool's own rules for writing a topic's lists.
    # It also reads a library shared with its caller (docs/plans/2026-10-13-one-share.md).
    "service/connector.py": {"service.accounts", "service.account_peers", "service.reading",
                             "service.serving",
                             "service.tokens", "service.google_signin", "service.library_writes",
                             "service.owner_tools", "pipeline.topics.lists",
                             S + "desktop_only", S + "page_words"},
    # The libraries shared with a Google account: an invite redeemed for it, and its rows.
    "service/account_peers.py": {"service.db", "service.grants", "service.tokens", S + "invite",
                                 "pipeline.retrieval.peers"},
    # What this service answers by itself for a library's owner: status, a reader view, an
    # invite, cutting one reader off. Worded like the Mac's `share` and `unshare`.
    "service/owner_tools.py": {"service.grants", "service.library_writes", "service.publications",
                               "service.reading", "service.tokens", "service.pages.make",
                               "service.pages.store", "pipeline.retrieval.reads", S + "invite",
                               S + "readers", S + "agent_invite", S + "page_words"},
    # The line (§14 of that plan): calls the web leaves for the owner's Mac, and its answers.
    "service/line.py": {"service.accounts", "service.db", "service.grants", "service.tokens"},
    # The four library writes as the web offers them. `forget`'s preview reads the served copy:
    # an Oracle by the local preview itself, items with the local consent sentences.
    "service/library_writes.py": {"service.line", "pipeline.forget.atoms",
                                  "pipeline.forget.oracle", "pipeline.retrieval.peers"},
    # The connector's Google sign-in, with revocation.
    "service/google_signin.py": set(),
    # Everything held for one library, deleted: the route's flow and the operator's command.
    "service/deletion.py": {"service.accounts", "service.db", "service.tokens",
                            "service.uploads", "service.pages.store"},
    "service/accounts.py": {"service.db", "service.google_device"},
    "service/google_device.py": set(),
    "service/access.py": {"service.tokens"},
    "service/serving.py": {"service.access", "service.publications", "service.uploads",
                           S + "delta"},
    "service/uploads.py": {"pipeline.retrieval.peers"},
    "service/tokens.py": {"service.db"},
    "service/grants.py": {"service.db", "service.tokens", S + "invite"},
    "service/publications.py": {"service.db", "service.usage"},
    "service/usage.py": {"service.db"},
    "service/stats.py": {"service.db", "service.publications", "pipeline.retrieval.peers"},
    "service/db.py": set(),
    # Pages of sources: the flows (`make`, `show`), the store, the PDF fetch, the drawing.
    "service/pages/make.py": {"service.pages.pdfs", "service.pages.store"},
    "service/pages/comments.py": {"service.pages.store"},
    "service/pages/show.py": {"service.pages.render", "service.pages.store"},
    "service/pages/store.py": {"service.db", "service.uploads"},
    "service/pages/pdfs.py": {"pipeline.platforms.papers.pdf", "pipeline.robots"},
    "service/pages/render.py": {"service.pages.cards", "service.pages.prose"},
    "service/pages/cards.py": {"service.pages.prose"},
    "service/pages/prose.py": set(),
    "opyt_core/service_http.py": set(),       # foundations: one HTTP exchange, `requests` only
}

# Outside the unit, the only production file that may import it besides the tool's entry: the
# scrapers process's composition, which runs the publishing passes (lanes slice 7).
COMPOSITION = "pipeline/background.py"
# Files outside the unit that may import exactly these of its modules: the server's rider flow
# asks for the share-back offer (server restructure, ruled 2026-10-05).
# Onboarding's X sign-in off macOS calls the service through `client` (2026-10-12).
OUTSIDE_IMPORTERS: dict[str, set[str]] = {"mcp_server/riders.py": {S + "share_back",
                                                                   # reader view's standing line
                                                                   S + "page_words"},
                                          # where the service is, and the X sign-in it runs
                                          "pipeline/onboarding/service_signin.py":
                                              {S + "account", S + "client"},
                                          # the end of setup offers the user's own claude.ai
                                          # while this install has no service token
                                          "pipeline/onboarding/handoff.py": {S + "account"},
                                          # the line: it asks the service for the web's tool
                                          # calls as this install, and owes a push after one
                                          # that changed the library (2026-10-13)
                                          "mcp_server/line.py":
                                              {S + "account", S + "client", S + "push_gate"}}


_INITS = ("pipeline/sharing/__init__.py", "service/__init__.py", "service/pages/__init__.py")


def _module_of(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def _imported(rel: str, tree: ast.AST) -> list[tuple[int, str]]:
    """Every repository module `rel` imports, as `(line, dotted name)`, relative ones resolved."""
    package = _module_of(rel).rsplit(".", 1)[0]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                anchor = ".".join(parts[:len(parts) - node.level + 1])
                base = f"{anchor}.{base}" if base else anchor
            for a in node.names:
                full = f"{base}.{a.name}"
                is_module = (ROOT / (full.replace(".", "/") + ".py")).exists()
                out.append((node.lineno, full if is_module else base))
    return [(ln, m) for ln, m in out if m.split(".")[0] in INTERNAL]


def _parse(rel: str) -> ast.AST:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)


def _unit_files() -> list[str]:
    files = sorted(str(p.relative_to(ROOT)) for d in ("pipeline/sharing", "service", "service/pages")
                   for p in (ROOT / d).glob("*.py"))
    return files + ["mcp_server/share_tools.py", "mcp_server/page_tools.py"]


def violations_of_the_table() -> list[str]:
    bad = []
    for rel, allowed in ALLOWED.items():
        for line, mod in _imported(rel, _parse(rel)):
            if mod not in allowed and mod not in FOUNDATIONS:
                bad.append(f"{rel}:{line} imports {mod}, which its row does not allow")
    return bad


def reaches_into_the_unit() -> list[str]:
    """Production files outside the unit that import it, or the install's half importing the
    service."""
    unit = set(_unit_files())
    bad = []
    for top in INTERNAL:
        for p in (ROOT / top).rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            for line, mod in _imported(rel, _parse(rel)):
                into_unit = mod == "service" or mod.startswith(("service.", S))
                if (rel not in unit and rel != COMPOSITION and into_unit
                        and mod not in OUTSIDE_IMPORTERS.get(rel, ())):
                    bad.append(f"{rel}:{line} imports {mod} from outside the unit")
                if rel.startswith("pipeline/sharing/") and mod.startswith("service"):
                    bad.append(f"{rel}:{line} imports {mod}: the client wheel has no service")
    importers_of_http = {"pipeline/sharing/client.py", "pipeline/retrieval/served.py"}
    for top in INTERNAL:
        for p in (ROOT / top).rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            if rel in importers_of_http:
                continue
            for line, mod in _imported(rel, _parse(rel)):
                if mod == "opyt_core.service_http":
                    bad.append(f"{rel}:{line} imports service_http; the service's clients are "
                               f"sharing's client and retrieval's served")
    return bad


def imports_inside_functions() -> list[str]:
    bad = []
    for rel in _unit_files():
        for fn in ast.walk(_parse(rel)):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bad.append(f"{rel}:{node.lineno} imports inside {fn.name}()")
    return bad


def test_every_module_imports_only_what_its_row_allows():
    assert violations_of_the_table() == []


def test_every_module_of_the_unit_has_a_row():
    assert sorted(set(_unit_files()) - set(_INITS) - set(ALLOWED)) == []


def test_nothing_outside_the_unit_reaches_in():
    assert reaches_into_the_unit() == []


def test_imports_sit_at_the_top():
    assert imports_inside_functions() == []


def test_the_package_inits_stay_empty():
    for rel in _INITS:
        assert (ROOT / rel).read_text(encoding="utf-8").strip() == "", f"{rel} must stay empty"
