"""
fetch_wc_odds.py — Pull 2026 World Cup odds from The Odds API.

Two tiers, to respect the free 500-credit/month quota:

  FEATURED (cheap): one call returns h2h + totals + spreads for ALL ~71 events.
                    Cost = (#markets) credits per region, regardless of #events.
  PER-EVENT (niche): btts, double_chance, 1st-half, corners, cards, player props.
                     Cost = (#markets actually returned) per event. Only pulled for
                     matches you ask for (default: those closing within --hours).

Usage:
  python3 fetch_wc_odds.py                  # featured only (3 credits)
  python3 fetch_wc_odds.py --events --hours 30   # + niche markets for soon matches
  python3 fetch_wc_odds.py --events --all        # + niche markets for ALL matches (pricey)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime as dt
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENT_DIR = os.path.join(DATA, "events")
os.makedirs(EVENT_DIR, exist_ok=True)

API_KEY = os.environ.get("ODDS_API_KEY", "0ce377e63ef8333b1794c5f26e08b384")
BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"
REGION = "eu"  # includes Pinnacle (sharp) + Betfair exchange

FEATURED_MARKETS = "h2h,totals,spreads"
# niche markets available on the per-event endpoint (eu region, mostly Pinnacle)
NICHE_MARKETS = [
    "btts", "double_chance", "draw_no_bet", "team_totals",
    "h2h_h1", "totals_h1", "btts_h1",
    "alternate_totals_corners", "alternate_totals_cards",
    "player_goal_scorer_anytime", "player_shots_on_target",
    "player_to_receive_card",
]

FEATURED_RAW = os.path.join(DATA, "wc_featured_raw.json")


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-last")
    if r.status_code != 200:
        print(f"  ! {r.status_code}: {r.text[:160]}", file=sys.stderr)
        return None, rem, used
    return r.json(), rem, used


def fetch_featured():
    print(f"[featured] {FEATURED_MARKETS} for all events ({REGION})…")
    data, rem, used = _get(f"{BASE}/sports/{SPORT}/odds",
                           dict(apiKey=API_KEY, regions=REGION, markets=FEATURED_MARKETS,
                                oddsFormat="decimal"))
    if data is None:
        sys.exit("featured fetch failed")
    json.dump(data, open(FEATURED_RAW, "w"))
    print(f"  {len(data)} events · cost {used} · {rem} credits left -> {FEATURED_RAW}")
    return data


def fetch_event(event_id, label):
    """Per-event niche markets. The API bills only markets actually returned."""
    data, rem, used = _get(f"{BASE}/sports/{SPORT}/events/{event_id}/odds",
                           dict(apiKey=API_KEY, regions=REGION,
                                markets=",".join(NICHE_MARKETS), oddsFormat="decimal"))
    if data is None:
        return
    json.dump(data, open(os.path.join(EVENT_DIR, f"{event_id}.json"), "w"))
    nbk = len(data.get("bookmakers", []))
    print(f"  [event] {label}: {nbk} books · cost {used} · {rem} left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", action="store_true", help="also pull per-event niche markets")
    ap.add_argument("--all", action="store_true", help="niche markets for ALL events (pricey)")
    ap.add_argument("--hours", type=float, default=30.0,
                    help="pull niche markets for events kicking off within N hours")
    args = ap.parse_args()

    events = fetch_featured()

    if args.events:
        now = dt.datetime.now(dt.timezone.utc)
        targets = []
        for e in events:
            ko = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            hrs = (ko - now).total_seconds() / 3600
            if args.all or (0 <= hrs <= args.hours):
                targets.append(e)
        print(f"[events] pulling niche markets for {len(targets)} event(s)…")
        for e in targets:
            fetch_event(e["id"], f"{e['home_team']} vs {e['away_team']}")


if __name__ == "__main__":
    main()
