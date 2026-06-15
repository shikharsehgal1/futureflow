"""
fetch_sp.py — Pull the LIVE SportsPredict Probability Cup questions via the official API.

Gives the exact question wording per match (no more guessing the catalog) so the model
can map each real question to our probability + to-win entry, and (later) read/submit
our predictions.

SECURITY: the API key is read from the SP_API_KEY environment variable and never
written to disk. Set it before running:
    export SP_API_KEY="sp_live_..."     # rotate this key — it has been exposed in chat
    python3 fetch_sp.py

Endpoints confirmed working with the key (NestJS API, query-param style):
    GET /api/v1/markets?event_id=<cup>     -> all questions across the Cup
    GET /api/v1/markets?match_id=<id>      -> 10 questions for one match
    GET /api/v1/predictions?event_id=<cup> -> OUR submitted predictions (read)
NOT exposed: crowd/consensus probability (hidden by design), leaderboard, /me.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import urllib.request

BASE = "https://api.sportspredict.com/api"
CUP_EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"   # Jump Trading Probability Cup
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(DATA, "sp_questions.csv")


def _key():
    k = os.environ.get("SP_API_KEY")
    if not k:
        sys.exit("Set SP_API_KEY env var (export SP_API_KEY='sp_live_...'). Not hardcoded for security.")
    return k


def _get(path):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": f"Bearer {_key()}",
                                          "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def main():
    markets = _get(f"/v1/markets?event_id={CUP_EVENT}")
    rows = []
    for m in markets:
        match = m.get("match", {})
        rows.append(dict(match=match.get("name", ""), match_id=match.get("id", ""),
                         question=m.get("question", ""), market_id=m.get("id", ""),
                         lobby_id=m.get("lobby_id", ""), status=m.get("status", ""),
                         closing_time=match.get("closing_time", "")))
    rows.sort(key=lambda r: (r["closing_time"] or "", r["match"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["match", "match_id", "question", "market_id",
                                          "lobby_id", "status", "closing_time"])
        w.writeheader(); w.writerows(rows)
    nmatch = len({r["match_id"] for r in rows})
    print(f"Pulled {len(rows)} live questions across {nmatch} matches -> {OUT}")
    # our current predictions (read-only)
    preds = _get(f"/v1/predictions?event_id={CUP_EVENT}")
    print(f"Our submitted predictions so far: {len(preds)}")


if __name__ == "__main__":
    main()
