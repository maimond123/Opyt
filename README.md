<div align="center">

# Opyt

**Your attention, made queryable.**

[![License](https://img.shields.io/github/license/maimond123/Opyt)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/opyt?color=%2334D058&label=pypi)](https://pypi.org/project/opyt/)
[![Python](https://img.shields.io/pypi/pyversions/opyt)](https://pypi.org/project/opyt/)
[![Stars](https://img.shields.io/github/stars/maimond123/Opyt?style=flat)](https://github.com/maimond123/Opyt/stargazers)

[**Website**](https://useopyt.com) · [**Docs**](https://useopyt.com/docs.html) · [**Use cases**](https://useopyt.com/use-cases.html) · [**Compare**](https://useopyt.com/compare.html)

</div>

---

**A local-first knowledge base built from the people you already read, served to your AI client over MCP.**

Every bookmark, follow and subscription was you deciding whose thinking is worth your time. Opyt pulls the full public archive of those people from X, Substack, GitHub, their own blogs and arXiv, and turns it into one SQLite file your assistant can search, read and count over. There is no new app and no chat UI. Your client calls the tools; your client's model does the reasoning, on the subscription you already pay for.

*If it is useful to you, star it. That is how other people find it.*

<a href="https://github.com/maimond123/Opyt"><img src="https://img.shields.io/github/stars/maimond123/Opyt.svg?style=social&label=Star&maxAge=2592000" alt="GitHub stars"></a>

---

## Why Opyt

- **Nobody you did not choose.** Candidates are ranked off your own follows, Lists, subscriptions, bookmarks and likes. There is no recommendation model and no trending list. Zero hits is an answer: nobody you trust has touched the claim.
- **Your backlog is searchable on the first run.** Everything you saved before today comes in during setup. You are not starting from an empty store.
- **It grows on the days you never open it.** Reading a topic end to end emits standing questions. Those questions keep running against arXiv, GitHub and OpenAlex, and stage what they find for you to review when you feel like it.
- **Full archives, not the three posts you bookmarked.** Confirm one person and Opyt finds their other platforms, verifies them, and pulls years of posts, repos and essays in full text.
- **Free of new subscriptions.** Reading and reasoning run on the AI client you already have. One metered key covers classification and embeddings, on cheap open models.
- **One local SQLite file.** Everything lives in `~/.opyt/opyt.db`. No vault of markdown, no dashboard, no daemon, no account.
- **Any MCP client.** Claude Code, Claude Desktop, Cursor, Windsurf, Codex, or anything else that speaks MCP over stdio.
- **MIT licensed.** [Read the source](https://github.com/maimond123/Opyt).

---

## Quick start

### 1. Install

<details open>
<summary><b>Claude Code</b></summary>

```bash
# once per machine, if you do not already have uv
curl -LsSf https://astral.sh/uv/install.sh | sh

claude mcp add Opyt -- uvx --from opyt==0.1.0a3 opyt-mcp
```
</details>

<details>
<summary><b>Claude Desktop</b></summary>

Without a terminal: download [`opyt-0.1.0a3.mcpb`](https://github.com/maimond123/Opyt/releases/latest), double-click it, and review the install screen Desktop shows you. The bundle carries its own Python and every package it needs, so it wants no `uv`, makes no network call at launch, and starts in about eleven seconds. That is why it is a 198 MB download.

Or from a terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# absolute path: uv edited your shell profile, not this shell
~/.local/bin/uvx --from opyt==0.1.0a3 opyt-install-client --claude-desktop
```
</details>

<details>
<summary><b>Cursor</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uvx --from opyt==0.1.0a3 opyt-install-client --cursor
```

This merges Opyt into `~/.cursor/mcp.json` beside whatever servers are already there, and copies the old file aside first. Running it twice changes nothing, and `--uninstall` removes the entry and leaves the rest.
</details>

<details>
<summary><b>Windsurf</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uvx --from opyt==0.1.0a3 opyt-install-client --windsurf
```

Merges into `~/.codeium/windsurf/mcp_config.json`, backing up the old file first.
</details>

<details>
<summary><b>Anything else that speaks MCP</b></summary>

Install `uv`, run `which uvx` to get its absolute path, and add this to whatever the client calls its MCP config:

```json
{
  "mcpServers": {
    "Opyt": {
      "command": "/Users/you/.local/bin/uvx",
      "args": ["--from", "opyt==0.1.0a3", "opyt-mcp"]
    }
  }
}
```

Use the absolute path, not a bare `uvx`. A desktop app spawns its servers with a minimal `PATH` that never sources your shell profile, so the directory uv's installer added to that profile does not exist as far as the process is concerned.
</details>

<details>
<summary><b>Let an agent do it</b></summary>

Paste this into the client you want Opyt in:

> Install the Opyt MCP server for this client, from useopyt.com/docs.html.
</details>

> `0.1.0a3` is a pre-release, so the version is pinned explicitly. Drop the pin once a stable release exists.
>
> Opyt does not run at claude.ai. Your client starts Opyt as a process on your own machine and pipes to it, and a browser tab cannot start a process on your machine. Claude Desktop is the same account and the same models, and takes one double-click.

### 2. Say `onboard`

Restart your client. Opyt appears as nine tools. Then:

```
you › onboard

⚙ onboard()

   x bookmarks    691 saved posts, full text
   x likes         99 liked authors
   x follows      492 accounts
   x lists          6 lists
   substack        19 subscriptions · 12 saved posts, full text
   read from your logged-in browser · no API key, no password

claude › Everything you saved is in. 703 posts are now searchable,
         and Opyt knows the 1,193 accounts they came from.
```

A browser tab opens for the one key it needs and you click Approve. Nothing is ever pasted into chat. It then looks for a browser already signed into x.com, which is how Opyt reads X, so there is no X key to get.

### 3. Say `oracle`

```
you › oracle

⚙ oracle()

   1  @jasonfurman   you follow · you subscribe · saved 12×
   2  @drvolts       you follow · saved 8×
   3  @karpathy      liked 4×
```

Confirm the ones you want and Opyt pulls each person's whole archive: their X posts, their GitHub repos and READMEs, their essays on their own site, going back years.

---

## The tools

Nine tools. Every argument, its type and default, what each call returns and what it costs are in the [full reference](https://useopyt.com/docs.html). The docstring on each tool in `mcp_server/` is the authoritative version of the same thing.

| Tool | Effect | What it does |
|---|---|---|
| [`onboard`](#2-say-onboard) | writes · re-entrant | Acquires the one key, reads your browser session, takes consent, runs the first import. |
| [`oracle`](#3-say-oracle) | writes · spends | Screens the people your own signals already point at, confirms them, ingests them. |
| `add_oracle` | two-phase · spends | Adds one person by handle or URL, finds everywhere they publish, pulls the archive. |
| [`search`](#find-something) | read-only | Ranked retrieval over the store, filterable by tag, kind, source, author and date. |
| [`open`](#find-something) | read-only · free | The full stored text of one atom, plus its live source URL. |
| [`aggregate`](#count-what-you-actually-read) | read-only · free | Counts across the whole store rather than a sample: by topic, author, kind, date. |
| [`sitting`](#read-a-topic-end-to-end) | writes · spends | Reads every atom on one topic end to end, in publication order, through a lens. |
| [`frontier`](#catch-up-on-what-arrived) | read-only · free | The ranked review queue of what your standing questions found. |
| [`hopper`](#keep-something) | two-phase · spends | Takes any URL, works out what it is, and saves it as an atom. |

The server owns only the last segment of each name. A client that namespaces adds its own prefix, so `search` reaches Claude Code as `mcp__Opyt__search`. All nine return a JSON object.

`search`, `open` and `aggregate` are the LLM-free retrieval core: they return data and your host model does the reasoning. They are also the only three that take `kb=`, which reads [a knowledge base someone shared with you](#share-a-knowledge-base).

---

## Examples

### Find something

```
you › Are tariff price effects one-off or recurring?

⚙ search("are tariff price effects one-off or recurring")

claude › Three of your own sources speak to this. The closest is a post
         you bookmarked yourself in June:

   0.81  @jasonfurman     x post   2026-06-11   you bookmarked this
   0.74  @m_c_klein       essay    2026-05-04   from an author you confirmed
   0.71  @nicktimiraos    x post   2026-04-22
```

`search` is a router, not an answer. Each hit is a card plus a pointer, and `open(atom_id)` returns the full stored text before you assert what a source says. Ranking is pure relevance: a BM25 arm and a semantic arm fused by rank, with no trust or popularity re-rank on top.

### Read a topic end to end

```
you › Read everything I've got on GLP-1 drugs and muscle loss,
      and lay out what it actually claims.

⚙ sitting(action="read", lens="claims")

claude › Eleven claims came back, each paired with what would break it
         and each citing the documents it came from. The first three:

   claim                                    what would disprove it
   The headline number hides what came      Body-composition scans published
   off: fat and muscle go together.         beside the headline figure.

   A 28% loss is not comparable to a 15%    One measurement standard adopted
   one unless both measured lean mass       across the trials.
   the same way.

   At population scale the binding          Oral dosing flattening the
   constraint is people staying on the      discontinuation curve.
   drug, not how well it works.
```

A sitting is the complement to a search. Search returns the best few; a sitting walks every document on the topic in publication order, so a position that changed over time reads as a change instead of a contradiction. Five lenses read the same material differently:

| Lens | What it returns |
|---|---|
| `claims` | Every claim, each paired with what would disprove it. |
| `briefing` | What holds, what is still open, and where your authors converge. |
| `trajectory` | The phases the thinking moved through, in publication order. |
| `disconfirmation` | The case against a claim you name, from your own sources only. |
| `gaps` | Whether your material answers a question, and the nearest misses if not. |

Every sitting also emits standing questions, which is what feeds the next section.

### Catch up on what arrived

```
Tuesday
you  › Read everything I've saved on data-center grid costs.
codex › Three questions came out of that read and they are still
        running. You typed none of them.

Friday · you ran nothing in between
you  › Anything new?

⚙ frontier()

codex › Six things came in since Tuesday, ranked by how many of your
        own questions reached each one:

   1  paper  Who Pays for the Load? Data-Center Demand and Retail Rates
             all 3 questions · 2 of your topics
   2  repo   PyPSA/pypsa-eur                   1 question · 1.2k stars
   3  paper  Interconnection Queue Times by Region, 2019 to 2026
             1 question
   +  3 more staged, ranked the same way
```

The queue sits on your machine and waits. Opyt says at the start of a session that something is there, and never pushes the list itself into context uninvited.

### Count what you actually read

```
you › What have I been reading most since March?

⚙ aggregate(date_from="2026-03")

cursor › Tariffs and inflation is your biggest topic since March,
         with grid and data centers close behind:

   tariffs & inflation      96
   grid & data centers      84
   interconnection queues   53
   GLP-1 trials             41
```

This is a count over everything you saved, in one pass, not an impression from what you remember saving.

### Keep something

```
you › Save this: arxiv.org/abs/2608.01144

⚙ hopper("arxiv.org/abs/2608.01144")
   routed → paper ingester
   preview: "Marginal Cost Allocation for Large Flexible Loads" · 22 pages · confirm?

you › yes

claude › Saved in full text, searchable beside everything else.
```

One sentence. No folders, no tags, no deciding where it belongs.

---

## What it reads

| Source | What comes in | Access |
|---|---|---|
| **X** | bookmarks · likes · follows · Lists | your browser session |
| **Substack** | subscriptions · saved posts, full text | your browser session |
| **GitHub** | a tracked person's repos and READMEs | public |
| **Blogs** | a tracked person's whole archive, found by feed or sitemap | public |
| **arXiv · OpenAlex** | papers matching your standing questions | public |
| **Any URL** | a paper, post, article or repo you hand it, saved in full | you |
| **LinkedIn** | posts and articles from the people you track | *coming soon* |
| **Aggregators** | Hacker News · Hugging Face · npm · crates.io · Stack Overflow · Product Hunt · Homebrew | *coming soon* |
| **Funding** | company and round data, asked by name | *coming soon* |

X and Substack are read from the browser session you are already signed into, on this machine only. There is no API key and no password.

---

## How it works

Content enters the store as **atoms**, Opyt's unit of stored content, one per post, repo or paper. Two independent paths put them there.

**Track a person.** `add_oracle` (or `oracle` to browse candidates first) takes a handle or URL. Opyt auto-detects which of X, Substack, a personal blog and GitHub that person actually publishes to, then pulls their full archive from each. Candidate ranking is lexicographic: person-level acts such as a follow, a subscription or a List outrank content-level ones such as a bookmark or a like. A background rail keeps every tracked person's sources current without you asking again.

**Let Frontier watch your standing topics.** Frontier is a passive discovery rail. It runs the standing questions your sittings emitted against arXiv, GitHub and OpenAlex on a schedule, judges what it finds against your topics, and stages the good ones. You read the queue with `frontier` when you want to.

Everything lands in one SQLite database at `~/.opyt/opyt.db`: atoms, their chunks, a full-text index, and the entity and trust edges between tracked people. The MCP server is the only interface, and it creates its own store on first use.

Several background rails also spawn per session, refreshing tracked people's sources, running Frontier's stages, pulling new X bookmarks. Each is independently gated, so a stalled or failing rail never blocks the others or the server itself.

---

## What it costs

Opyt is free and MIT-licensed. The expensive part is the reading and reasoning, and that runs on the AI subscription you already pay for.

| | For | Cost |
|---|---|---|
| **Your AI client** | every question, every answer, all the reasoning | your existing subscription |
| **OpenRouter** | sorting and extracting what comes in, on a cheap open model, plus the embeddings search ranks with | pay-as-you-go credits |
| **X** | posts, timelines, profiles, threads | free, through your own browser |
| **Everything else** | papers, blogs, GitHub, Substack, any URL you hand it | free |

`opyt-keys --list` shows which credentials are set, never their values. `.env.example` documents each one.

---

## Share a knowledge base

`search`, `open` and `aggregate` take a `kb=` argument that reads someone else's store instead of your own.

```
you › I need to pick an eval setup for retrieval.
      Has anyone already been through this?

⚙ search("evaluating retrieval quality", kb="ana")
⚙ search("evaluating retrieval quality", kb="ravi")

claude › Two of them have. Ana read the framework and its repo in March,
         and Ravi has the benchmark it is usually compared against.
         Nothing on this in yours.
```

It only goes one way. Their store opens read-only, and what gets recorded is that you searched, never what you searched for.

An owner publishes with `opyt-push`, which is a full replace: an export is a projection of a store, not a log of changes to one, so the newest upload wins. A reader runs `opyt-redeem <url> <code>` once and sets nothing else.

---

## Design principles

Three constraints hold across the whole codebase, enforced by an AST-based guard (`scripts/guard.py`) that runs in pre-commit and CI.

- **Distributable.** Nothing may assume a specific machine. Paths are derived at runtime from `sys.executable` and `Path(__file__)`, never hardcoded.
- **Client-agnostic.** The core and every knowledge tool run on any MCP client, not just Claude Code. Claude-Code-specific behavior is opt-in and never load-bearing.
- **Fail-safe.** A missing optional input degrades to an empty result, not a crash. A failed external call skips cleanly. It never writes partial state and never marks unfinished work done.

---

## Contributing

The install path above is for using Opyt. To work on it:

```bash
git clone https://github.com/maimond123/Opyt.git
cd Opyt
bash scripts/setup.sh   # venv, editable install, git hooks
pytest tests/
```

`setup.sh` assumes a `python3` that already satisfies `requires-python >= 3.10` and does not check. A stock macOS `python3` is 3.9.6, as is the one `xcode-select --install` delivers. Read the header of that script before running it.

Issues and pull requests are welcome at [github.com/maimond123/Opyt](https://github.com/maimond123/Opyt/issues).

---

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

Built with ❤️ in New Jersey

[**useopyt.com**](https://useopyt.com)

</div>
