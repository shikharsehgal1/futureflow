"""
refresh.py — One refresh cycle for the World Cup model. Designed to be run on a
schedule (cron, or the Claude Code /loop skill) so the dashboard always reflects the
latest sharp lines before each kickoff.

Credit-frugal by design (free tier = 500/month):
  * Always pulls FEATURED markets (h2h/totals/spreads) for all events = 3 credits.
  * Pulls NICHE markets (corners/cards/players/1H) only for matches kicking off within
    NICHE_HOURS that aren't already cached — and only if remaining credits > FLOOR.
  * Recomputes the question table, and settles any newly-finished matches (Brier).

Usage:
  python3 refresh.py                 # normal cycle
  python3 refresh.py --niche-hours 6 # widen the niche window
  python3 refresh.py --no-niche      # featured + recompute only (3 credits)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, datetime as dt
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENT_DIR = os.path.join(DATA, "events")
API_KEY = os.environ.get("ODDS_API_KEY", "0ce377e63ef8333b1794c5f26e08b384")
SPORT = "soccer_fifa_world_cup"
CREDIT_FLOOR = 160          # never spend niche credits below this
NICHE_HOURS_DEFAULT = 5.0


def _run(*cmd):
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=HERE, timeout=300).returncode


def remaining_credits():
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds",
                         params={"apiKey": API_KEY, "regions": "eu", "markets": "h2h",
                                 "oddsFormat": "decimal"}, timeout=30)
        json.dump(r.json(), open(os.path.join(DATA, "wc_featured_raw.json"), "w"))
        return int(r.headers.get("x-requests-remaining", 0)), r.json()
    except Exception as e:
        print(f"  credit check failed: {e}", file=sys.stderr)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niche-hours", type=float, default=NICHE_HOURS_DEFAULT)
    ap.add_argument("--no-niche", action="store_true")
    args = ap.parse_args()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[refresh {stamp}]")

    # 1. featured pull (also returns remaining credits + all events) — overwrites raw cache
    rem, events = remaining_credits()
    if rem is not None:
        print(f"  featured pulled · {rem} credits remaining")

    # 2. niche markets for imminent, uncached matches (budget-guarded)
    if events and not args.no_niche and rem and rem > CREDIT_FLOOR:
        now = dt.datetime.now(dt.timezone.utc)
        soon = []
        for e in events:
            ko = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            hrs = (ko - now).total_seconds() / 3600
            cached = os.path.exists(os.path.join(EVENT_DIR, f"{e['id']}.json"))
            if 0 <= hrs <= args.niche_hours and not cached:
                soon.append(e)
        if soon:
            print(f"  {len(soon)} imminent uncached match(es) -> niche pull")
            _run("python3", os.path.join(HERE, "fetch_wc_odds.py"),
                 "--events", "--hours", str(args.niche_hours))
        else:
            print("  no imminent uncached matches needing niche markets")
    elif rem and rem <= CREDIT_FLOOR:
        print(f"  SKIP niche: only {rem} credits left (floor {CREDIT_FLOOR})")

    # 3. refresh Benz's published ratings (free; updates on his refit dates), recompute
    _run("python3", os.path.join(HERE, "fetch_lbenz.py"))
    _run("python3", os.path.join(HERE, "predict_wc.py"))
    _run("python3", os.path.join(HERE, "apply_adjustments.py"))
    _run("python3", os.path.join(HERE, "simulate_wc.py"))

    # 4. snapshot predictions (freezes closing values so finished matches stay scoreable)
    _run("python3", os.path.join(HERE, "snapshot.py"))

    # 5. settle any newly finished matches (Brier tracker, scored off the snapshot)
    _run("python3", os.path.join(HERE, "track_brier.py"), "--days", "3")

    print(f"[refresh done {stamp}]")


if __name__ == "__main__":
    main()
