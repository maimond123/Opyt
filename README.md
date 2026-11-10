<div align="center">

# Opyt

**Your attention is the context your AI is missing.**

[![License](https://img.shields.io/github/license/maimond123/Opyt)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/opyt?color=%2334D058&label=pypi)](https://pypi.org/project/opyt/)
[![Python](https://img.shields.io/pypi/pyversions/opyt)](https://pypi.org/project/opyt/)
[![Stars](https://img.shields.io/github/stars/maimond123/Opyt?style=flat)](https://github.com/maimond123/Opyt/stargazers)

[**Website**](https://useopyt.com) · [**Docs**](https://useopyt.com/docs.html) · [**Use cases**](https://useopyt.com/use-cases.html) · [**Compare**](https://useopyt.com/compare.html)

<a href="https://useopyt.com/opyt-demo.mp4"><img src="https://useopyt.com/opyt-demo-play.jpg" alt="Watch the 90-second Opyt demo" width="720"></a>

</div>

---

**A knowledge base that grows from your attention across the platforms you already use, served to your AI client over MCP.**

Every bookmark, follow and subscription was you deciding whose thinking is worth your time. Opyt pulls the full public archive of those people from X, Substack, GitHub, their own blogs and arXiv, and turns it into one SQLite file your assistant can search, read and count over. There is no new app and no chat UI. Your client calls the tools; your client's model does the reasoning, on the subscription you already pay for.

*If it is useful to you, star it. That is how other people find it.*

<a href="https://github.com/maimond123/Opyt"><img src="https://img.shields.io/github/stars/maimond123/Opyt.svg?style=social&label=Star&maxAge=2592000" alt="GitHub stars"></a>

---

## Why Opyt

I kept bookmarking posts on X, subscribing to Substacks, and saving papers I meant to read. Most of them stayed unread. I wanted that material where I actually work and learn, so Opyt turns the sources you already follow and trust into a knowledge base, keeps adding to it as they publish, and answers from it in the client you already use.

- **Nobody you did not choose.** Candidates are ranked off your own follows, Lists, subscriptions, bookmarks and likes. There is no recommendation model and no trending list. Zero hits is an answer: nobody you trust has touched the claim.
- **Your backlog is searchable on the first run.** Everything you saved before today comes in during setup. You are not starting from an empty store.
- **It grows on the days you never open it.** Name a topic and Opyt follows it: the specific things you watch are searched every day on arXiv, GitHub and OpenAlex, and your own people's posts on it are picked out as they arrive. Your assistant checks what came in before it shows you anything. Catching up becomes one question, answered from people you trust plus what arrived while you were away.
- **Full archives, not the three posts you bookmarked.** Confirm one person and Opyt finds their other platforms, verifies them, and pulls years of posts, repos and essays in full text.
- **Check before you build.** Search repos, papers and posts together: find out whether someone already built your idea, and what the people you follow learned trying it.
- **Share with a friend.** Grant each other access to your knowledge bases, search both with one question, see whose collection each result came from, and revoke whenever you want.
- **Free of new subscriptions.** Reading and reasoning run on the AI client you already pay for, and Opyt calls no model at all: search matches your words.
- **One local SQLite file.** Your whole knowledge base lives in `~/.opyt/opyt.db`. No new app to open, no vault of markdown, no dashboard, no account. One background process, `opyt-worker`, runs the ingest jobs on a schedule; it opens no port and keeps only its own small job queue beside the store.
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

claude mcp add -s user Opyt -- uvx --from opyt@latest opyt-mcp
```
</details>

<details>
<summary><b>Claude Desktop</b></summary>

Without typing anything: download [`opyt-install.command.zip`](https://useopyt.com/opyt-install.command.zip) and open it — it unzips to `opyt-install.command`. **Right-click that and choose Open**, not a double-click: macOS blocks a downloaded file until you approve it once that way. Terminal opens, asks which app to install into, and runs the two commands below for you. (It ships zipped because a browser download drops the execute bit, and macOS will not run a `.command` without it.)

Or run them yourself:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# absolute path: uv edited your shell profile, not this shell
~/.local/bin/uvx --from opyt@latest opyt-install-client --claude-desktop
```
</details>

<details>
<summary><b>Cursor</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uvx --from opyt@latest opyt-install-client --cursor
```

This merges Opyt into `~/.cursor/mcp.json` beside whatever servers are already there, and copies the old file aside first. Running it twice changes nothing, and `--uninstall` removes the entry and leaves the rest; once no app still has Opyt set up, it removes the background worker too.
</details>

<details>
<summary><b>Windsurf</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uvx --from opyt@latest opyt-install-client --windsurf
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
      "args": ["--from", "opyt@latest", "opyt-mcp"]
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

> `opyt@latest` and not a version, so a published fix reaches you on your next client restart with nothing to re-run. It is also not a bare `opyt`: `uvx` reuses its cached environment for a bare requirement and would never look for a newer build. Offline, `@latest` falls back to the copy you already have rather than failing to start.
>
> **A local install is macOS-only.** `onboard` installs the background worker, `opyt-worker`, as a LaunchAgent; there is nothing to run yourself. Every pull and import runs in it, so it is what makes anything arrive, including while your client is closed. It opens no port and starts at login, and if it is switched off or removed, the next `onboard` puts it back. Windows and Linux are not supported yet.
>
> claude.ai and ChatGPT can read the library your Mac builds, from any computer or phone: during `onboard`, connect it to your Google account, then add `https://api.useopyt.com/mcp` as a custom connector and sign in with the same account. They read a published copy, by keyword, and cannot change it; everything else stays in your local client.

### 2. Say `onboard`

Restart your client. Opyt appears as thirteen tools. Then:

```
you › onboard

⚙ onboard()

   x bookmarks    691 saved posts, full text
   x likes         99 liked authors
   x follows      492 accounts
   x lists          6 lists
   substack        19 subscriptions · 12 saved posts, full text
   read in Opyt's own browser window · no API key, no password

claude › Everything you saved is in. 703 posts are now searchable,
         and Opyt knows the 1,193 accounts they came from.
```

There is no key to get: Opyt runs free. It opens its own browser window at x.com or substack.com for you to sign in once, which is how it reads them, and nothing is ever pasted into chat.

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

Fifteen tools. Every argument, its type and default, what each call returns and what it costs are in the [full reference](https://useopyt.com/docs.html). The docstring on each tool in `mcp_server/` is the authoritative version of the same thing.

| Tool | Effect | What it does |
|---|---|---|
| [`onboard`](#2-say-onboard) | writes · re-entrant | Connects your sources and starts importing what you saved. |
| [`oracle`](#3-say-oracle) | writes · spends | Screens the people your own signals already point at, confirms them, ingests them. |
| `add_oracle` | two-phase · spends | Adds one person by handle or URL, finds everywhere they publish, pulls the archive. |
| [`search`](#find-something) | read-only | Ranked retrieval over the store, filterable by tag, kind, source, author and date. |
| [`open`](#find-something) | read-only · free | The full stored text of one atom, plus its live source URL. |
| [`aggregate`](#count-what-you-actually-read) | read-only · free | Counts across the whole store rather than a sample: by source, kind and author, over any date range. |
| [`latest`](#catch-up-on-what-arrived) | read-only · free | What is new since you last looked: per topic you follow, the finds to check and your people's posts, then everything else, with why each item is here. |
| [`hopper`](#keep-something) | two-phase · spends | Takes a paper, repo, Substack post, article or X post link, works out which it is, and saves it as an atom. |
| [`watch`](#follow-a-topic) | writes · free | Follows a topic: names to watch on arXiv, GitHub and OpenAlex, plus the words and people that pick out your own people's posts on it. |
| `look` | read-only · free | Searches X, Substack, arXiv, OpenAlex and GitHub now for a few named things, and saves nothing. |
| [`share`](#share-a-knowledge-base) | two-phase · publishes | Shows you what is in your store, then hands you a link that lets someone search it. |
| [`accept`](#share-a-knowledge-base) | single-phase · writes | Registers a knowledge base somebody shared with you, from the link they sent. |
| [`unshare`](#share-a-knowledge-base) | two-phase · deletes | Cuts off one reader, or every reader plus the served copy. |
| `reader_view` | publishes · expires | Opens the sources behind an answer in reader view: one private page with each in full, that you can read or pass on. It is deleted 90 days after it was last opened. |
| `forget` | two-phase · removes | Removes one atom and its references, or ends one Oracle subscription, keeping their atoms unless `delete_atoms` asks to delete what their own sources brought in. |

The server owns only the last segment of each name. A client that namespaces adds its own prefix, so `search` reaches Claude Code as `mcp__Opyt__search`. All fifteen return a JSON object.

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

`search` is a router, not an answer. Each hit is a card plus a pointer, and `open(atom_id)` returns the full stored text before you assert what a source says. Ranking is pure relevance: BM25 keyword ranking, with no trust or popularity re-rank on top.

### Follow a topic

```
you › Keep me up to date on AI drug discovery.

⚙ watch(action="preview", name="AI drug discovery",
        watches=[{"name": "Boltz-2", "spellings": ["Boltz2"]}, "rentosertib",
                 "Virtual Cell Challenge"],
        words=["drug discovery", "clinical trial", "wet lab", …],
        people=["@slavov_n", "@FredaDuan"])

   last month, it would have found:
   Boltz-2                  7 papers · 23 repos
   rentosertib              2 papers
   Virtual Cell Challenge   20 repos
   your people's posts      about 4 a day

claude › Three things to watch, and two of your people who mostly write
         about this. Boltz-2 is busy, mostly small wrapper repos, so I'll
         only bring you the ones that matter. Follow it, and check it for
         you every morning?
```

A topic watches names, not categories. Each name is searched every day on arXiv, GitHub and OpenAlex, and a find counts only when the name is in its title or abstract: measured on 403 labelled results, that cut off-topic finds from 44% to 13%. The words and people pick out your own people's posts on the subject, which arrive anyway.

### Catch up on what arrived

```
Friday · you ran nothing since Tuesday
you  › Anything new?

⚙ latest()

   AI drug discovery    14 finds to check · 6 posts
   grid costs            5 finds to check · 2 posts
   outside your topics  18 posts

codex › Two things worth your time on grid costs: a paper on who pays for
        data-center load in retail rates, and a regional study of
        interconnection queue times that Jesse Jenkins linked. On drug
        discovery, the rentosertib Phase 2a write-up is out.
```

New finds wait on your machine, outside your library, until your assistant checks them: it shows the ones worth your time, keeps those, and drops the rest without a word. A search of your own store mentions what is new once per conversation, and the list itself never enters context uninvited.

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
| **arXiv · GitHub · OpenAlex** | new papers and repos naming what your topics watch, kept once your assistant checks them | public |
| **A link you hand it** | a paper, repo, Substack post, article or X post, saved in full | you |
| **LinkedIn** | posts and articles from the people you track | *coming soon* |
| **Aggregators** | Hacker News · Hugging Face · npm · crates.io · Stack Overflow · Product Hunt · Homebrew | *coming soon* |
| **Funding** | company and round data, asked by name | *coming soon* |

X and Substack are read through a browser window Opyt opens on this machine, which you sign into once. Opyt never reads your normal browser's sessions, and there is no API key and no password in chat.

---

## How it works

Content enters the store as **atoms**, Opyt's unit of stored content, one per post, repo or paper. Two independent paths put them there.

**Track a person.** `add_oracle` (or `oracle` to browse candidates first) takes a handle or URL. Opyt auto-detects which of X, Substack, a personal blog and GitHub that person actually publishes to, then pulls their full archive from each. Candidate ranking is lexicographic: person-level acts such as a follow, a subscription or a List outrank content-level ones such as a bookmark or a like. A background rail keeps every tracked person's sources current without you asking again.

**Follow a topic.** `watch` takes the specific things to watch, and the words and people that pick out your own people's posts on the subject. The worker searches each name on arXiv, GitHub and OpenAlex every day and keeps a find only when the name appears in it. Finds wait outside the library until your assistant checks them in `latest`; the ones it keeps are added, the rest never come back.

Everything lands in one SQLite database at `~/.opyt/opyt.db`: atoms, their text, a full-text index over each item whole, and the entity and trust edges between tracked people. The MCP server is the only interface, and it creates its own store on first use.

A resident process, `opyt-worker`, does the unattended work — refreshing tracked people's sources, searching your topics' watches, pulling new X bookmarks. `onboard` installs it on macOS, and macOS starts it at login. It runs that work in one process with a loop per rate limit, each waiting on its own limit, so a source that stalls or is throttled never holds up the others or the server itself. The worker restarts itself every six hours, so a published fix reaches the background work too, not just the tools. Nothing is collected without it: pulls and imports run only in the worker, never inside the MCP server.

---

## How calls run

Opyt is free and MIT-licensed, and it calls no model: search matches words rather than meaning, and images are not read.

| | For | Access |
|---|---|---|
| **Your AI client** | every question, every answer, all the reasoning | your configured client |
| **X** | posts, timelines, profiles, threads | your OPYT-managed browser session |
| **Everything else** | papers, blogs, GitHub, public Substack posts, a link you hand it | public source access |

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

Getting there is two sentences and no terminal. On your side:

```
you › share my knowledge base with Leo.

⚙ share()

claude › 2,876 atoms, mostly papers and X posts, from March 2024 to yesterday.
         The top authors are @karpathy, @hypersoren and @swyx. Sharing sends
         all of it, it lasts until you revoke it, and it stays current on its
         own. Go ahead?

you › yes

⚙ share(confirm=True, as_name="David", for_whom="Leo")

claude › Send Leo this: https://useopyt.com/invite#Kq7f...
```

On theirs:

```
you › someone sent me this: https://useopyt.com/invite#Kq7f...

⚙ accept("https://useopyt.com/invite#Kq7f...")

claude › Registered. Ask me anything and I can search David's reading too,
         with kb="David".
```

Publishing is self-service: no invitation and no account. The served copy updates itself every hour while your Mac is awake, whenever your library changed (every five minutes for the hour after you share or connect Google), and `share(refresh=True)` updates it at once. `unshare(reader="Leo")` cuts off one person and leaves the copy serving; `unshare()` cuts off everyone and deletes it.

Under it, a push sends only the rows that changed since the copy that is being served, and the server applies it only if its copy is the one the change was cut against; otherwise the same push sends the whole copy instead. Both acts run from your assistant — `share` publishes and mints the invite, `accept` redeems one — and neither has a terminal command any more.

---

## Design principles

Three constraints hold across the whole codebase.

- **Distributable.** Nothing may assume a specific machine. Paths are derived at runtime from `sys.executable` and `Path(__file__)`, never hardcoded.
- **Client-agnostic.** The core and every knowledge tool run on any MCP client, not just Claude Code. Claude-Code-specific behavior is opt-in and never load-bearing.
- **Fail-safe.** A missing optional input degrades to an empty result, not a crash. A failed external call skips cleanly. It never writes partial state and never marks unfinished work done.

---

## Contributing

The install path above is for using Opyt. To work on it:

```bash
git clone https://github.com/maimond123/Opyt.git
cd Opyt
bash scripts/setup.sh   # venv, editable install
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
