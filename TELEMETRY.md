# What Opyt records

> **The local Opyt server makes no outbound call of its own. The hosted sharing service counts
> what it serves, and publishes the totals.**

That sentence is the whole design. Everything below is the detail behind it.

Opyt has two pieces. The one you install is a local MCP server that reads and writes files on
your own disk; it has no telemetry, no analytics endpoint, and no reporting call of any kind.
The second piece is the **hosted sharing service** — it exists so somebody can publish a
knowledge base and grant other people read access to it. You reach it only by redeeming a grant
code somebody sent you. If you never do that, nothing in this document applies to you.

This document is checked against the code. `tests/service/test_telemetry_doc.py` reads the real
schema out of `service/store.py` and fails if any table or column here is missing, so this page
cannot quietly fall behind what the service actually stores.

---

## What the service records

Four tables in `service.db`, and nothing else.

### `tokens` — who may call the service

| column | what it holds |
|---|---|
| `token_sha256` | the SHA-256 of a bearer token. **Never the token.** The database holds no usable credential. |
| `owner` | the one knowledge base this token can reach. It is the entire permission model. |
| `role` | `owner` (upload, grant, revoke) or `reader` (query). |
| `label` | how the owner names this reader, so they can tell one from another when revoking. |
| `install_id` | a random id the client generates once per installation. There is no account behind it, and it is never linked to a person, an email, or a machine. |
| `created_at` | when the token was issued. |

Gives: how many knowledge bases are published, how many readers each has, how many distinct
installations ever redeemed a code, and how long it takes a reader to go from redeeming to
reading.

### `owner_claims` — which token may publish under a name

| column | what it holds |
|---|---|
| `owner` | a published name, claimed forever. Revoking every token for it does not release it — a released name would let a stranger re-claim it and be served to the previous owner's readers under the name they saved. |
| `token_sha256` | the one owner token allowed to upload under this name. |
| `claimed_at` | when the name was first claimed. |

Gives: no metric — nothing reads it for counting. It exists so two people can never publish
under one name.

### `grant_codes` — the one-time invitations

| column | what it holds |
|---|---|
| `code_sha256` | the SHA-256 of a grant code. Same rule as a token: the hash, never the code. |
| `owner` | whose knowledge base the code grants access to. |
| `label` | the name the owner attaches to whoever they are inviting. |
| `created_at` | when the code was minted. |
| `redeemed_at` | when it was spent, or NULL. Non-NULL means dead — a code buys exactly one reader token. |

Gives: codes minted, codes redeemed, the conversion rate between them, and time-to-redeem.

### `usage_daily` — how often a knowledge base was read

| column | what it holds |
|---|---|
| `day` | `YYYY-MM-DD`. **Deliberately a day, not a timestamp.** |
| `owner` | whose knowledge base was read. |
| `reader` | the reader's token hash. |
| `tool` | which of the three read operations ran: `search`, `open`, or `aggregate`. |
| `n` | how many times, that day. |
| `zero_results` | how many of those searches returned nothing. Always 0 for the other tools. |

Gives: reads per day, opens per search, the zero-result rate, days-active, retention, and the
counter a rate limit would read.

**Why counts and not a log.** An earlier version of this table wrote one row per request, with a
timestamp. Every column in it was innocent and the *shape* was not: one row per request is a
full-resolution record of who read from whom and when, and that record is also the thing that
gets breached or subpoenaed. Daily counts keep every number listed above and drop the trace.
What is given up: exact timing, burst patterns, and the order of reads within a session — none of
which appears in any metric anyone has asked to read.

**Stated honestly: this is not fully anonymous.** `tokens.label` is how an owner names their
readers so they can revoke access, so a reader hash resolves to a name the owner chose. That is
kept on purpose. "This reader searched 12 times on Tuesday" is ordinary product analytics about
someone who was deliberately granted access; "this reader read these four documents" is a
different object, and the service records no such thing.

### Outside the database

The service's process log records the HTTP method, the path and the status code of each request,
which is what makes a broken deploy diagnosable. It does **not** record the caller's address —
`service/log_config.json` exists specifically to remove the field the web server writes by
default — and it does not record request bodies. Raw process logs are kept for 30 days and then
deleted.

---

## What is never collected

Five refusals, each for its own reason. They are not one rule restated.

**Query text.** No column in `service.db` holds a search string, and nothing in the service logs
a request body. Search queries are the sharpest re-identification surface there is — AOL's
"anonymized" 2006 search logs were unpicked to named individuals from the queries alone — and no
number this service publishes needs them. Note the one thing this promise is: a retention
commitment, not a structural guarantee. Keyword search tokenizes the query string, so on a
keyword or hybrid search the text does reach the service's memory for the length of the request.
It is never written down. Semantic search is structurally blind — the reader embeds the query on
their own machine and only the resulting numbers cross the wire.

**IP addresses.** Never stored, and deliberately not logged. The web server writes the caller's
address into its access log by default; `service/log_config.json` replaces that default with the
same format minus the address, and `tests/service/test_logging.py` fails if it comes back. TLS
terminates on the service's own machine, so no third-party proxy sees a reader's address either.

**Which atoms a reader read.** The service holds the documents, so a document id resolves to the
document — a `(reader, atom_id)` pair is a reading history, it reveals what a person was trying
to learn, they did not choose to disclose it, and they cannot retract it. Per-atom read counts
were designed and then rejected for a second reason as well: `tokens` already lists every reader
of a knowledge base, so at one or two readers a missing reader column is *implied* rather than
absent, and shipping both halves of a pair while refusing the pair is not a policy.

**Per-request timestamps.** `usage_daily` is daily by construction, so there is nothing in the
database to expire. The day is the finest resolution of time the service keeps.

**Client-side telemetry, and cross-owner content analysis.** The local server phones nobody —
install counts come from PyPI download statistics and GitHub clone counts, which are public,
collected by third parties, and require no code on your machine. And an owner uploaded their
knowledge base so that the readers *they granted* could query it; mining what everyone uploaded
is a different purpose than the one they agreed to. It would need each owner's explicit consent,
and it does not exist.

---

## The honest limit: aggregation buys nothing at one reader

If exactly one person holds a grant to a knowledge base, then "this knowledge base served 40
reads" **is** that person's activity, whatever the columns say. Every scheme on this page
protects a reader at twenty and none of them protects a reader at one.

There is no technical fix for that, so the service does the only thing that works: it tells you.
The response to redeeming a grant code carries a plain-language notice of exactly what gets
counted, before you have made a single query.

---

## Where the totals are published

The service publishes its own aggregates, so that whoever runs it holds no more information than
anyone else does:

- `https://api.useopyt.com/stats` — a page.
- `https://api.useopyt.com/v1/stats` — the same numbers as JSON.

Both are public and need no credential. Neither carries an owner name, a reader name, or a token
hash — only totals.
