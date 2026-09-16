# What Opyt records

> **The local Opyt server reports nothing about you to anybody. The hosted sharing service counts
> what it serves, and publishes the totals.**

Opyt has two pieces. The one you install is a local MCP server that reads and writes files on
your own disk; it has no telemetry, no analytics endpoint, and no reporting call of any kind.
The second piece is the **hosted sharing service**, reached only if you shared your own knowledge
base or accepted an invitation to somebody else's. If you have done neither, nothing below
applies to you.

This document is checked against the code: `tests/service/test_telemetry_doc.py` reads the real
schema out of `service/store.py` and fails if any table or column here is missing.

---

## The one thing your install sends by itself

If you shared your knowledge base, your install re-uploads it when somebody has read the served
copy since your last upload **and** your store has changed since then — so an unchanged store
never re-uploads. What goes up is the export itself and the token saying which knowledge base it
replaces. Nothing else, and nothing about you. An install that has never shared holds no token,
so this never runs at all.

---

## What the service records

Five tables in `service.db`, and nothing else.

**`tokens` — who may call the service.** `token_sha256` (the SHA-256 of a bearer token — **never
the token**; the database holds no usable credential), `owner` (the one knowledge base this token
can reach — the entire permission model), `role` (`owner` or `reader`), `label` (how the owner
names this reader, for revoking), `install_id` (a random id generated once per installation — no
account behind it, never linked to a person, an email, or a machine), `created_at`.

**`owner_claims` — which token may publish under a routing key.** `owner` (a published routing
key, claimed forever — releasing one would let a stranger serve the previous owner's readers),
`token_sha256`, `claimed_at`. Nothing reads it for counting; it exists so two people can never
publish under one key.

**`owner_uploads` — what each knowledge base costs to store.** `owner` (the routing key —
assigned at random, not a name anybody chose), `bytes` (size of the export currently served; `0`
means unpublished), `reads_at_upload` (read-count watermark, so "has anyone read since the last
push" is a comparison of two counters), `first_published_at`, `last_published_at`. Publishing is
self-service with no identity check, so this is how an operator sees a knowledge base eating the
disk and removes it — the whole of the abuse response. The per-knowledge-base rows are published
at `/v1/stats`.

**`grant_codes` — the one-time invitations.** `code_sha256` (the hash, never the code), `owner`,
`label`, `created_at`, `redeemed_at` (non-NULL means dead: a code buys exactly one reader token).

**`usage_daily` — how often a knowledge base was read.** `day` (`YYYY-MM-DD` — **deliberately a
day, not a timestamp**), `owner`, `reader` (the reader's token hash), `tool` (`search`, `open`,
or `aggregate`), `n` (how many times, that day), `zero_results` (searches returning nothing).
An earlier version logged one row per request; every column was innocent and the *shape* was not —
a full-resolution record of who read from whom and when is also the thing that gets breached or
subpoenaed. Daily counts keep every metric and drop the trace.

**Stated honestly: this is not fully anonymous.** `tokens.label` means a reader hash resolves to
a name the owner chose — kept on purpose, so owners can revoke. "This reader searched 12 times on
Tuesday" is ordinary analytics about someone deliberately granted access; "this reader read these
four documents" is a different object, and the service records no such thing.

**Outside the database.** The process log records HTTP method, path, and status code — not the
caller's address (`service/log_config.json` removes the field the web server writes by default)
and not request bodies. Raw logs are deleted after 30 days.

---

## What is never collected

- **Query text.** No column holds a search string and nothing logs a request body — queries are
  the sharpest re-identification surface there is. This is a retention commitment, not a
  structural one: keyword search tokenizes the query in memory for the length of the request; it
  is never written down. Semantic search is structurally blind — the query is embedded on the
  reader's own machine and only numbers cross the wire.
- **IP addresses.** Never stored, deliberately not logged; `tests/service/test_logging.py` fails
  if the field comes back. TLS terminates on the service's own machine, so no third-party proxy
  sees a reader's address either.
- **Which atoms a reader read.** A `(reader, atom_id)` pair is a reading history the reader did
  not choose to disclose and cannot retract. Per-atom read counts were designed and rejected.
- **Per-request timestamps.** `usage_daily` is daily by construction; the day is the finest
  resolution of time the service keeps.
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
