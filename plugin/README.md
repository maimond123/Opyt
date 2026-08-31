# OPYT — Claude Code plugin

Bundles the OPYT Claude Code integration into one installable unit: the `Opyt` MCP server and its
nine knowledge tools.

| | tool | what it does |
|---|---|---|
| **read** | `search` | route to the most relevant atoms — returns POINTERS, not content |
| | `open` | follow a pointer and return the stored raw text (call this before citing anything) |
| | `aggregate` | counts and coverage over a slice of the store |
| | `sitting` | read one whole topic end to end, in publication order |
| **grow** | `hopper` | save any URL as an atom — the one manual "keep this" path |
| | `frontier` | queue of recent papers/repos your standing queries pulled in |
| | `oracle` | choose who to trust, from people you already curate |
| | `add_oracle` | add one person as a trusted source |
| **setup** | `onboard` | set up OPYT on this machine — call this first on a fresh install |

Tools appear as `mcp__Opyt__*`.

`search` and `open` are a pair on purpose. `search` tells you *where* something is; `open` tells
you what it *says*, along with whether the stored copy is complete or a paywall teaser. Quoting a
`search` snippet as if it were the source is the one misuse worth knowing about.

## Why a plugin (vs hand-wiring)

Registering this by hand meant editing `~/.claude.json` with **an absolute path to a specific
venv** — which only works on the machine it was written on. The plugin removes that: the MCP
server runs via **`uvx --from opyt==0.1.0a3 opyt-mcp`** (no pre-install — uvx fetches OPYT from
PyPI into an ephemeral env).

## Install

**You need [`uv`](https://docs.astral.sh/uv/) on your PATH** — nothing works without it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`.mcp.json` runs `uvx --from opyt==0.1.0a3 opyt-mcp`, so `uv` fetches OPYT from PyPI into an
ephemeral environment. Nothing to pre-install, no GitHub account, and no SSH key.

The version is pinned because `0.1.0a3` is a pre-release. Drop the pin to a bare `opyt` when a
stable release exists.

First launch downloads ~96 packages, which takes roughly 40 seconds. It looks like a hang. It
isn't.

**1. Add the plugin.** This repo is a one-plugin marketplace:

```bash
claude plugin marketplace add /path/to/opyt        # repo root holds .claude-plugin/marketplace.json
claude plugin install opyt@opyt                    # add --scope user|project|local; default is user
# or, for local dev without a marketplace:
claude --plugin-dir /path/to/opyt/plugin
```

A GitHub shorthand works too: `claude plugin marketplace add owner/repo`.

> **Add the marketplace from a git repo or a local path — not from a bare `marketplace.json`
> URL.** This marketplace declares its plugin with a relative `"source": "./plugin"`, and relative
> paths only resolve for git-based and local-path marketplaces. Pointing Claude Code at a raw
> `marketplace.json` over HTTP leaves `./plugin` unresolvable.

**2. Restart Claude Code**, or run `/reload-plugins` if the install summary asks for it.

## First run

The server writes **`~/.opyt/settings.yaml`** from the shipped template, once, and never
overwrites an existing one. That file is the install receipt — check it first to answer "did my
install work?". A fresh user starts with no tracked people; you don't inherit the author's list.

`~/.opyt/opyt.db` is created on demand the first time anything opens the store for writing.

Growing a corpus needs **one** account: OpenRouter, which `onboard` walks you through — a browser
tab, one click, nothing to paste. It pays for the classifier, the image reads, and the embedder.
The key lands in `~/.opyt/.env`. Never paste one into the chat; `onboard` opens a local page for
that.

**X needs no key and there is no way to supply one.** Every X read — your bookmarks, an Oracle's
timeline, a single post, a profile — runs on your own logged-in x.com session. If no local browser
has one, `onboard` offers to open a window for you to log in.

Search needs no key on an empty store. Once atoms exist, the default hybrid mode embeds your query
through OpenRouter, so it wants that one key too; `mode="bm25"` stays keyless.

**(Optional) Install the package** to put the `opyt-keys` credential CLI on your PATH:

```bash
uv tool install opyt==0.1.0a3     # adds opyt-keys to your PATH (pipx works too)
```

The MCP tools work without this — it only lights up the `opyt-keys` CLI.

## What runs in the background

Opening a session starts the server, and the server forks detached catch-up rails — keeping your
trusted sources current, running standing research queries, pulling new bookmarks. They coalesce
(mostly hourly), so most session opens do nothing at all.

Some of those rails spend metered credit. Others read your logged-in browser session. **Each one
carries its own consent marker, and a fresh install has none**, so on first launch every one of
them declines and reports what it would need. Consent is per-rail on purpose: opting into one
loop never opts you into another with a different cost shape.

`onboard` is where you grant them. The browser step — which is what triggers the macOS Keychain
prompt, and which warns you before it does — is what opens the rail that reads your X and
Substack sessions. Measured on a cold install: seven rails fire, all seven decline, total spend
`$0.00`.

## Notes / known edges

- **Installed from PyPI, pinned to a pre-release.** `opyt==0.1.0a3`, published 2026-08-30. The
  pin is what makes the install reproducible: `uvx` resolves the exact version and caches it, so
  a later release never changes what an existing install runs.

- **A pre-release does not gate itself here.** pip and uv skip pre-releases only when a stable
  version also exists. While every published version is a pre-release, a bare `uvx --from opyt opyt-mcp`
  resolves to it anyway. The pin is therefore documentation of what you get, not a lock against
  a stable version you would otherwise receive.

- **Client paths are macOS-only.** `opyt_core/install_client.py` knows Cursor, Claude Desktop and
  Windsurf config locations for macOS only; Linux and Windows paths are not yet handled. The
  browser step in `onboard` also reads cookies via the macOS Keychain.
