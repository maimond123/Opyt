# What Opyt records

> **The local Opyt server reports nothing about you to anybody. The hosted sharing service counts
> what it serves, and publishes the totals.**

Opyt has two pieces. The one you install is a local MCP server that reads and writes files on
your own disk; it has no telemetry, no analytics endpoint, and no reporting call of any kind.
The second piece is the **hosted sharing service**, reached only if you shared your own knowledge
base, accepted an invitation to somebody else's, or opened an answer's sources in reader view. If
you have done none of these, nothing below applies to you.

This document is checked against the code: `tests/service/test_telemetry_doc.py` reads the real
schema out of `service/db.py` and fails if any table or column here is missing.

---

## What your install sends by itself

If you shared your knowledge base, your install does two things without being asked:

- **It keeps the served copy current.** About hourly while your computer is awake, when your
  store has changed since its last upload, it uploads what changed (the whole export when a
  patch will not do), with the token saying which knowledge base it belongs to. An unchanged
  store uploads nothing.
- **It asks the service whether a change is waiting for it.** Only when something can ask your
  library for one from the web: a Google account you connected, or an agent you gave a key.
  Then your install keeps one request open to the service while your computer is awake. When
  you ask an assistant on the web to save a link, remove something, follow a topic or add a
  person, that request comes down, your own Opyt makes the change, and its answer goes back.
  With nothing connected, it asks once an hour whether that has changed, and nothing more.

Nothing else, and nothing about you. An install that has never shared holds no token, so
neither runs at all.

---

## What the service records

Twelve tables in `service.db`, and nothing else.

**`tokens` — who may call the service.** `token_sha256` (the SHA-256 of a bearer token — **never
the token**; the database holds no usable credential), `owner` (the one knowledge base this token
can reach — the entire permission model), `role` (`owner`, `reader`, or `agent`: a key the owner
made for their own agent, which acts as them), `label` (how the owner names this reader, for
revoking), `install_id` (a random id generated once per installation — no account behind it,
never linked to a person, an email, or a machine), `created_at`.

**`owner_claims` — which token may publish under a routing key.** `owner` (a published routing
key, claimed until its knowledge base is deleted — releasing it earlier would let a stranger serve
the previous owner's readers), `token_sha256`, `claimed_at`. Nothing reads it for counting; it
exists so two people can never publish under one key.

**`owner_uploads` — what each knowledge base costs to store.** `owner` (the routing key —
assigned at random, not a name anybody chose), `bytes` (size of the export currently served),
`reads_at_upload` (read-count watermark, so "has anyone read since the last
push" is a comparison of two counters), `first_published_at`, `last_published_at`. Publishing is
self-service with no identity check, so this is how an operator sees a knowledge base eating the
disk and removes it — the whole of the abuse response. The per-knowledge-base rows are published
at `/v1/stats`.

**`grant_codes` — the one-time invitations.** `code_sha256` (the hash, never the code), `owner`,
`label`, `created_at`, `redeemed_at` (non-NULL means dead: a code buys exactly one token).

**`agent_grants` — which invitations are for an agent.** `code_sha256` (the same hash as the
invitation's row above) and `owner`. A row means that code buys an agent token; without one it
buys a reader token.

**`usage_daily` — how often a knowledge base was read.** `day` (`YYYY-MM-DD` — **deliberately a
day, not a timestamp**), `owner`, `reader` (the reader's token hash), `tool` (`search`, `open`,
`aggregate`, or `feed`), `n` (how many times, that day), `zero_results` (searches returning nothing).
An earlier version logged one row per request; every column was innocent and the *shape* was not —
a full-resolution record of who read from whom and when is also the thing that gets breached or
subpoenaed. Daily counts keep every metric and drop the trace.

**`accounts` — which library a Google account reads, for those who connect one.** Only when you
connect your library to Google (`share(to="connector")`) so the Opyt connector can read it
from Claude, ChatGPT or another assistant. `google_sub` (Google's permanent id for the account — **not your email or
name**, which are never stored), `owner` (the routing key of your library), `linked_at`. The row
needs both your install's owner token and your approval on Google's own page. The connector's
reads go into `usage_daily` under a hash of the account id, never the id itself.

**`account_peers` — the libraries someone shared with a Google account.** Only when you paste an
invite into the Opyt connector in Claude or ChatGPT, where there is no install of yours to hold
it. `google_sub` (as above: not your email or name), `name` (what your assistant calls that
library: its owner's own label), `owner` (its routing key), `token_sha256` (the hash of the
reader token your invite bought, the one its owner sees and can revoke) and `added_at`. Your reads
of it go into `usage_daily` under that token's hash, like any reader's. The row goes when the
owner deletes their library, and when they cut you off: the next read tells your assistant the
share has ended and removes it. To have it removed sooner, email hello@useopyt.com.

**`account_links` — a Google sign-in in progress.** `owner`, `device_code` and `user_code`
(Google's codes for that one sign-in), `verification_url` (Google's page for entering the code),
`expires_at`. Deleted when the sign-in completes or fails, and replaced if it expires; it holds
nothing once you are connected.

**`pages` — reader views of sources, each kept until 90 days after it was last opened.** Only when you ask your assistant to open the
sources an answer used in reader view. `id` (128 random bits: the page's link, and its only lock), `owner` (the
routing key of the library that made it; only that library's token can take it down), `title` and
`note` (what your assistant wrote at the top), `items` (the sources themselves, in full, as your
library keeps them), `made`, `expires` (90 days after the page was last opened, so
opening it pushes this out). A paper's PDF is fetched from where it is openly published and kept
beside the page. Anyone holding the link can read the page. Nothing records who did: the time in
`expires` is all a visit leaves. The page is deleted when it expires, or sooner when you take it
down.

**`page_comments` — what people write on a reader view.** Only when someone holding a reader
view's link comments on it. `id`, `page_id` (the reader view it is on), `thread` (the comment it
answers; its own id when it starts a thread), `item` (which source on the page), `quote`, `prefix`
and `suffix` (the passage commented on and a few words either side of it, copied from the source
so the page can find the passage again), `name` (whatever the commenter typed: nobody signs in and
nothing checks it), `body` (the comment), `made`, `key_sha256` (the hash of a key the commenter's
browser holds, which is all that can delete the comment before its page goes). Anyone holding the
link reads every comment on the page. Nothing else is kept about a commenter: no account, no
address, nothing about their browser. A comment is deleted with its page, or sooner by the
browser that posted it.

**`line_calls` — a change you asked for from the web, until your computer has made it.** Only
when you ask an assistant on the web to save a link, remove something, follow a topic or add a
person. Your library lives on your computer, so the service holds the request until your own Opyt
takes it, and then its outcome, so your assistant can tell you how it ended. `id` (128 random
bits), `owner` (the routing key of your library), `tool` (which of the four), `args` (what you
asked for and nothing else: the link, the topic's lists, the item's id or the person's handle),
`state` (`waiting`, `running`, `done` or `failed`), `answer` (what your Opyt answered: what was
saved, removed or added, or why not), `asked`, `waited_until` and `taken` (when you asked, until
when your assistant waited for the answer, and when your Opyt took it), `told` (whether the
outcome has been passed on to you). Unlike a read, this is one request with its time and its
content, so it is deleted a week after it was asked, whatever became of it.

**`line_seen` — when your Opyt last asked for such requests.** `owner`, `seen` (the time of its
latest ask: one row per library, overwritten each time). It tells the service whether your
computer is on, so a change is either made while your assistant waits or left for when it is
next on.

**Stated honestly: this is not fully anonymous.** `tokens.label` means a reader hash resolves to
a name the owner chose — kept on purpose, so owners can revoke. "This reader searched 12 times on
Tuesday" is ordinary analytics about someone deliberately granted access; "this reader read these
four documents" is a different object, and the service records no such thing.

**Deleting it all.** `unshare` deletes everything this service holds for your knowledge base:
the served copy, every reader's token and invitation, the Google accounts linked to it and the
ones it was shared with, the
changes asked for from the web that it still holds, its row in `owner_uploads`, its read counts
in `usage_daily`, its routing key's claim, and last your install's own token. The public totals are sums over these rows, so they drop by what is deleted.
The nightly backup of `service.db` keeps a deleted row for up to seven days. If you no longer
have the computer your library was built on, write to hello@useopyt.com: the operator runs the
same deletion for the Google account or routing key you name.

**Outside the database.** The process log records HTTP method, path, and status code — not the
caller's address (`service/log_config.json` removes the field the web server writes by default),
not request bodies, and no query string or sign-in link (`service/log_filters.py`). Raw logs are
deleted after 30 days.

**The connector's sign-in.** The connector signs you in with Google, asking only for `openid`:
your account's permanent id, no email address and no name. It keeps, encrypted on the service's
disk: each assistant that registered to connect (its name and return address), each sign-in in
progress for a few minutes, and for each connected assistant Google's answer to the sign-in (an
access token, a refresh token, and an id token naming the account id) for up to a year. FastMCP,
the sign-in library, stops using an expired record but leaves its encrypted file on disk. When an
assistant disconnects and revokes its token, that answer is deleted and Google is asked to revoke
the grant; removing Opyt from your Google account ends it at Google, which the connector checks
on every call.

**An X sign-in, while it runs.** An install with no browser of its own (an assistant's computer,
such as Muse's) connects X through a sign-in page this service runs. That page shows a browser
running on the service, and like any browser it sees what you type into X's page. When X shows
its home page, the service reads three of X's cookies (`auth_token`, `ct0` and `twid`, which
together are your X session) and holds them **in memory only, never on disk**, until your install
collects them (at most ten minutes), then forgets them. The browser's profile is deleted when the
sign-in ends. The log records only that a sign-in ended and how (connected, failed or expired);
the token your install collects with travels in a request body, and the page's own link and
session are blanked in the log, so no log holds any of them.

---

## What is never collected

- **Query text.** No column holds a search string and nothing logs a request body — queries are
  the sharpest re-identification surface there is. This is a retention commitment, not a
  structural one: keyword search tokenizes the query in memory for the length of the request; it
  is never written down.
- **IP addresses.** Never stored, deliberately not logged; `tests/service/test_logging.py` fails
  if the field comes back. TLS terminates on the service's own machine, so no third-party proxy
  sees a reader's address either.
- **Which atoms a reader read.** A `(reader, atom_id)` pair is a reading history the reader did
  not choose to disclose and cannot retract. Per-atom read counts were designed and rejected.
- **Per-request timestamps of reads.** `usage_daily` is daily by construction; the day is the
  finest resolution of time the service keeps about a read. A change you ask for from the web is
  not a read: `line_calls` holds it with its time, for a week.
- **Client-side telemetry, and cross-owner content analysis.** The local server phones nobody —
  install counts come from PyPI and GitHub statistics, which are public and need no code on your
  machine. And owners uploaded so *their* readers could query; mining what everyone uploaded is a
  purpose nobody agreed to, and it does not exist.

---

## The honest limit: aggregation buys nothing at one reader

If exactly one person holds a grant, then "this knowledge base served 40 reads" **is** that
person's activity, whatever the columns say. There is no technical fix, so the service does the
only thing that works: the response to redeeming a grant code states exactly what gets counted,
before you have made a single query.

---

## Where the totals are published

- `https://api.useopyt.com/stats` — a page.
- `https://api.useopyt.com/v1/stats` — the same numbers as JSON.

Both public, no credential, no reader name and no token hash. The JSON carries one
per-knowledge-base list (`stored_bytes_by_kb`: random routing key, bytes, first/last published
dates — no label, no traffic), published so that whoever runs the service holds no more
information than anyone else does.
