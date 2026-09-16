#!/usr/bin/env python3
"""
scripts/trial_keys.py — what the starter allowance costs, and which keys to switch off.

Typed by an operator on the gateway box. It is a script rather than a `__main__` on
`gateway/trial.py` for the reason `no-hand-run-cli-in-these-library-modules` gives: that module
is library code a running server imports, and a front door on it would be reachable by nobody in
production. This one is reachable by exactly the person it is for.

  python3 scripts/trial_keys.py                 # report: usage per key, and the cost summary
  python3 scripts/trial_keys.py --sweep         # ALSO switch off keys that are spent or expired
  python3 scripts/trial_keys.py --sweep --dry-run

THE REPORT IS THE POINT, and `--sweep` is housekeeping. `readiness.COST_NOTE` tells users their
reading costs "a few pennies"; until a real corpus has been built against a real key that is an
intention, not a measurement. `mean_usd` here is the measurement. If it comes back in dollars,
that string is wrong — and it is wrong at the moment a user is deciding whether to trust OPYT
with a card, which is the worst moment to be wrong.

Reads `OPYT_TRIAL_MANAGEMENT_KEY` and `OPYT_TRIAL_LEDGER` from the environment, same as the
gateway. Run it with the gateway's own EnvironmentFile:

  set -a; . /etc/opyt-gateway.env; set +a; python3 scripts/trial_keys.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx                                                       # noqa: E402

from gateway import trial                                          # noqa: E402

# Keys in these states can no longer spend anything, so switching them off changes nothing for
# their holder and keeps the provider's key listing readable. `live` is never swept — cutting
# somebody off mid-use is not housekeeping.
_SWEEPABLE = ("spent", "expired")


def _ledger() -> trial.Ledger:
    root = os.environ.get("OPYT_HOMES_ROOT") or str(Path.home() / ".opyt-homes")
    return trial.Ledger(Path(os.environ.get("OPYT_TRIAL_LEDGER") or Path(root) / ".trial-ledger.json"))


def _print_report(rows: list[dict]) -> None:
    print(f"{'subject':<24} {'state':<12} {'usage':>9} {'limit':>8}  expires")
    print("-" * 72)
    for row in rows:
        usage = f"${row['usage']:.4f}" if isinstance(row.get("usage"), (int, float)) else "-"
        limit = f"${row['limit']:.2f}" if isinstance(row.get("limit"), (int, float)) else "-"
        expires = (row.get("expires_at") or "")[:10] or "-"
        print(f"{row['subject'][:24]:<24} {row['state']:<12} {usage:>9} {limit:>8}  {expires}")


def _print_summary(summary: dict) -> None:
    print()
    print(f"keys minted        {summary['keys']}")
    print(f"  used             {summary['used']}")
    print(f"  never used       {summary['unused']}   (cost $0 — `limit` is a ceiling, not a "
          f"prepayment)")
    print(f"total spent        ${summary['total_usd']:.4f}")
    if summary["mean_usd"] is None:
        print("\nNo usage yet, so COST_NOTE is still unmeasured.")
        return
    print("per user who used it:")
    print(f"  mean             ${summary['mean_usd']:.4f}")
    print(f"  median           ${summary['median_usd']:.4f}")
    print(f"  most             ${summary['max_usd']:.4f}")
    print()
    # The threshold tracks what COST_NOTE actually CLAIMS, and moved with it on 2026-09-16: the
    # copy said "a few pennies" and this checked 0.10; it now says "well under a dollar" and this
    # checks 0.50. Half a dollar is where "well under" stops being honest — a check set at the
    # claim's own breaking point (1.00) would pass right up to the moment the sentence became a
    # lie, which is the one moment it needs to have already failed.
    verdict = ("holds — `readiness.COST_NOTE` claims well under a dollar, and this is"
               if summary["mean_usd"] < 0.50 else
               "BROKEN — `readiness.COST_NOTE` claims well under a dollar and this is not, and "
               "it is shown to users at the moment they are asked for a card")
    print(f"COST_NOTE check:   {verdict}")


async def _run(sweep: bool, dry_run: bool) -> int:
    if not trial.enabled():
        print(f"{trial.MANAGEMENT_KEY_ENV} is not set — this gateway mints nothing.",
              file=sys.stderr)
        return 1
    ledger = _ledger()
    if not ledger.path.exists():
        print(f"no ledger at {ledger.path} — nothing has been minted here.", file=sys.stderr)
        return 1

    async with httpx.AsyncClient() as http:
        rows = await trial.audit(http, ledger)
        _print_report(rows)
        if rows and all(r["state"] == "unauthorized" for r in rows):
            # Every row failing the same way is one fact, not N. Said once, plainly, because the
            # alternative reading — "OpenRouter is down" — sends the operator to the wrong place.
            print(f"\n{trial.MANAGEMENT_KEY_ENV} was rejected by OpenRouter. It must be a "
                  f"MANAGEMENT key (openrouter.ai -> Account -> Management Keys), not an "
                  f"ordinary API key.", file=sys.stderr)
            return 1
        _print_summary(trial.cost_summary(rows))

        if not sweep:
            return 0
        targets = [r for r in rows if r["state"] in _SWEEPABLE and r.get("hash")]
        print()
        if not targets:
            print("sweep: nothing to switch off.")
            return 0
        if dry_run:
            print(f"sweep (dry run): would switch off {len(targets)} key(s).")
            return 0
        done = 0
        for row in targets:
            if await trial.disable(http, row["hash"]):
                done += 1
            else:
                print(f"  could not switch off {row['subject']}", file=sys.stderr)
        print(f"sweep: switched off {done} of {len(targets)} key(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trial_keys",
        description="Report what the starter allowance costs; optionally switch off keys that "
                    "are already spent or expired.")
    ap.add_argument("--sweep", action="store_true",
                    help="also disable keys that are spent or expired (never a live one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --sweep, say what would be switched off and change nothing")
    args = ap.parse_args(argv)
    return asyncio.run(_run(args.sweep, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
