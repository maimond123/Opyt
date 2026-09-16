#!/usr/bin/env python3
"""
scripts/mcp_trace.py
A stdio pass-through that records the JSON-RPC traffic between an MCP client and an MCP server.

WHY THIS EXISTS. Claude Desktop's own MCP log records that a `tools/call` happened and nothing
about it: every entry reads `params { metadata: undefined }`. So a live onboarding test could be
read only from screenshots, which is slow and loses the arguments — and the arguments are the
thing under test. Which tool the host picked, and what it passed, is how you tell a bad tool
description from a bad model turn.

It sits in the client's config in place of the server, spawns the real server, and copies bytes
both ways unchanged. Zero product code knows it is there — the alternative was a logging hook
inside the server, which is production code that exists only to be observed.

    "opyt-dev": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/scripts/mcp_trace.py",
               "--out", "/tmp/opyt-trace.jsonl",
               "--", "/path/to/venv/bin/python", "-m", "mcp_server.server"],
      "env": {...unchanged...}
    }

STDOUT IS THE PROTOCOL. Nothing but the server's own bytes may reach it — a stray print corrupts
the stream and the client reports a disconnect with no cause. Every trace line goes to `--out`,
and this file's own failures go to stderr, which the client already captures.

Read a trace with:  python scripts/mcp_trace.py --read /tmp/opyt-trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def _pump(src, dst, out_path: Path, direction: str, lock: threading.Lock) -> None:
    """Copy `src` to `dst` line by line, appending a trace record per line.

    Line-oriented because MCP stdio frames are newline-delimited JSON. The copy is of the RAW
    bytes: the trace is a side effect, and a parse failure must never alter what the other side
    receives. That is why the write and flush happen before any decoding, and why a malformed
    line is recorded as raw text rather than dropped.
    """
    for raw in iter(src.readline, b""):
        dst.write(raw)
        dst.flush()

        rec: dict = {"t": round(time.time(), 3), "dir": direction}
        try:
            msg = json.loads(raw)
        except Exception:
            rec["raw"] = raw.decode("utf-8", "replace").rstrip("\n")
        else:
            rec["msg"] = msg
        with lock:                      # two pumps, one file
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
    dst.close()


def trace(out: Path, argv: list[str]) -> int:
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    lock = threading.Lock()
    threads = [
        # `c2s` is what the host sent — the tool name and its arguments, the thing screenshots
        # cannot show. `s2c` is what the server answered.
        threading.Thread(target=_pump, args=(sys.stdin.buffer, proc.stdin, out, "c2s", lock)),
        threading.Thread(target=_pump, args=(proc.stdout, sys.stdout.buffer, out, "s2c", lock)),
    ]
    for t in threads:
        t.daemon = True
        t.start()
    return proc.wait()


# ── reading a trace ─────────────────────────────────────────────────────────────────────────

def _brief(value, width: int = 400) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return s if len(s) <= width else s[:width] + f"… (+{len(s) - width} chars)"


def read(path: Path, width: int) -> None:
    """Print the calls and their results. Requests are matched to responses by JSON-RPC id.

    Only `tools/call` and its result are printed. The handshake and the list methods are noise
    once the server is up, and printing them buries the one line a reader came for.
    """
    pending: dict = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
            msg = rec["msg"]
        except Exception:
            continue
        stamp = time.strftime("%H:%M:%S", time.localtime(rec["t"]))

        if rec["dir"] == "c2s" and msg.get("method") == "tools/call":
            p = msg.get("params", {})
            pending[msg.get("id")] = (stamp, p.get("name"))
            print(f"\n[{stamp}] → {p.get('name')}")
            for k, v in (p.get("arguments") or {}).items():
                print(f"           {k} = {_brief(v)}")

        elif rec["dir"] == "s2c" and msg.get("id") in pending:
            start, name = pending.pop(msg["id"])
            if "error" in msg:
                print(f"[{stamp}] ← {name} ERROR {_brief(msg['error'], width)}")
                continue
            content = (msg.get("result") or {}).get("content") or []
            body = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            print(f"[{stamp}] ← {name}\n{_brief(body, width)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="trace file to append to")
    ap.add_argument("--read", type=Path, metavar="TRACE", help="print a trace instead of running")
    ap.add_argument("--width", type=int, default=2000, help="max chars per result body")
    ap.add_argument("server", nargs="*", help="the real server command, after --")
    args = ap.parse_args()

    if args.read:
        read(args.read, args.width)
        return 0
    if not args.out or not args.server:
        ap.error("--out and a server command are both required to trace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return trace(args.out, args.server)


if __name__ == "__main__":
    sys.exit(main())
