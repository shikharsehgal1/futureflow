"""
fetch_apifootball.py — Sharpen per-team match-stat rates for 18 "gap" national
teams using API-Football (api-sports.io / RapidAPI) free-tier data.

The World Cup model's data/team_stats_extra.csv currently holds *real goals* for
18 national teams but PRIOR (placeholder) values for every other market — shots,
shots on target, corners, offsides, fouls and cards are all just the tournament
base-rate prior. API-Football publishes per-fixture team statistics on its free
tier (shots, shots on target, corners, fouls, offsides, yellow/red cards), which
lets us replace those priors with real, shrunk per-match rates.

Approach (mirrors fetch_statsbomb.py exactly):
  * For each gap team, find recent national-team fixtures (2024-2026, last ~18).
  * Pull per-fixture statistics for both sides; aggregate to per-match "for" and
    "against" sums.
  * Shrink each raw rate toward the SAME tournament prior with the SAME weight
    (PRIOR_W = 3 equivalent matches), so the output is consistent with the rest
    of the model:
        rate = (raw_sum + PRIOR[stat] * PRIOR_W) / (n_matches + PRIOR_W)
  * Goals are KEPT from the existing file (already real / better-sourced) unless
    we genuinely recompute more of them; we only ever *improve* the 18 gap teams
    and never touch other teams or the CSV schema.

Authentication — two supported transports, auto-detected:
  1. api-sports.io direct (default):
       base   https://v3.football.api-sports.io
       header x-apisports-key: <KEY>
  2. RapidAPI:
       set API_FOOTBALL_HOST to the RapidAPI host
       (e.g. api-football-v1.p.rapidapi.com), and we send
       header x-rapidapi-key:  <KEY>
       header x-rapidapi-host: <API_FOOTBALL_HOST>
     The base URL becomes https://<API_FOOTBALL_HOST>/v3

Environment variables:
  API_FOOTBALL_KEY    (required to actually fetch; if unset we print setup
                       instructions and exit 0)
  API_FOOTBALL_HOST   (optional; if it contains "rapidapi" we use the RapidAPI
                       transport, otherwise it's treated as a direct base host)

Usage:
  python3 fetch_apifootball.py                 # all 18 gap teams
  python3 fetch_apifootball.py --teams "Norway,Scotland"
  python3 fetch_apifootball.py --dry-run       # show plan/budget, never call API
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

try:
    import requests
except ImportError:  # pragma: no cover - requests is expected to be present
    print("ERROR: this script needs the 'requests' package (pip install requests).",
          file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "team_stats_extra.csv")

# --- prior / shrinkage: identical to fetch_statsbomb.py so output is consistent
PRIOR = dict(goals=1.35, shots=11.5, sot=4.0, corners=4.8, offsides=1.8,
             fouls=11.5, fouls_drawn=11.5, cards=2.0, pens=0.10)
PRIOR_W = 3.0   # equivalent matches of prior weight

# stats written to the CSV (must match the 18-column schema, in this order)
STATS = ["goals", "shots", "sot", "corners", "offsides", "fouls", "cards", "pens"]
FIELDS = ["team", "n_matches"] + [f"{s}_{d}" for s in STATS for d in ("for", "against")]

# the 18 "gap" national teams we are allowed to improve
GAP_TEAMS = [
    "Algeria", "Austria", "Bosnia & Herzegovina", "Cape Verde", "Curaçao",
    "Czech Republic", "DR Congo", "Haiti", "Iraq", "Ivory Coast", "Jordan",
    "New Zealand", "Norway", "Paraguay", "Scotland", "South Africa", "Turkey",
    "Uzbekistan",
]

# Map our CSV team names -> API-Football search term (the API uses its own names).
# A search term that reliably matches the national side avoids ambiguous hits.
API_SEARCH = {
    "Algeria": "Algeria",
    "Austria": "Austria",
    "Bosnia & Herzegovina": "Bosnia",
    "Cape Verde": "Cape Verde Islands",
    "Curaçao": "Curacao",
    "Czech Republic": "Czech Republic",
    "DR Congo": "Congo DR",
    "Haiti": "Haiti",
    "Iraq": "Iraq",
    "Ivory Coast": "Ivory Coast",
    "Jordan": "Jordan",
    "New Zealand": "New Zealand",
    "Norway": "Norway",
    "Paraguay": "Paraguay",
    "Scotland": "Scotland",
    "South Africa": "South Africa",
    "Turkey": "Turkey",
    "Uzbekistan": "Uzbekistan",
}

SEASONS = [2024, 2023, 2022]        # free plan only allows seasons 2022-2024
_MIN_INTERVAL = 6.7                  # free plan = 10 requests/minute -> pace proactively
_PACE = [0.0]                        # last-request timestamp (mutable for closure)
TARGET_FIXTURES = 18                # aim for ~last 18 fixtures per team
DIRECT_BASE = "https://v3.football.api-sports.io"


# --------------------------------------------------------------------------- #
# request budget estimate (for the no-key instructions and the run plan)
# --------------------------------------------------------------------------- #
def budget_for(n_teams: int) -> dict:
    """Estimate API request count for n_teams on the free tier.

    Per team:
      1   /teams?search=...                 (resolve the national-team id)
      |SEASONS| /fixtures?team=&season=     (list recent fixtures, 1 call/season)
      TARGET_FIXTURES /fixtures/statistics  (one call per fixture we keep)
    """
    per_team = 1 + len(SEASONS) + TARGET_FIXTURES
    total = per_team * n_teams
    return {
        "per_team": per_team,
        "teams_resolve": n_teams,
        "fixtures_list": len(SEASONS) * n_teams,
        "fixture_statistics": TARGET_FIXTURES * n_teams,
        "total": total,
    }


def print_no_key_instructions(n_teams: int) -> None:
    b = budget_for(n_teams)
    free_daily = 100
    days = (b["total"] + free_daily - 1) // free_daily
    print(
        f"""
=============================================================================
 API-Football integration — NO API KEY DETECTED (env var API_FOOTBALL_KEY)
=============================================================================

This script enriches data/team_stats_extra.csv for {n_teams} national teams with
REAL per-match rates (shots, shots on target, corners, fouls, offsides, cards)
from API-Football. Right now those columns are placeholder PRIORS; only goals
are real. No key is set, so nothing was fetched and no data was fabricated.

HOW TO GET A FREE KEY
---------------------
  1. Sign up at https://dashboard.api-football.com (free account).
     (API-Football is also available via RapidAPI:
      https://rapidapi.com/api-sports/api/api-football )
  2. The FREE tier allows ~100 requests/day and exposes the endpoints we need,
     including per-fixture team statistics.
  3. Copy your API key from the dashboard.

THEN SET YOUR ENVIRONMENT AND RE-RUN
------------------------------------
  # api-sports.io direct (default transport):
  export API_FOOTBALL_KEY=your_key_here
  python3 {os.path.basename(__file__)}

  # OR via RapidAPI:
  export API_FOOTBALL_KEY=your_rapidapi_key
  export API_FOOTBALL_HOST=api-football-v1.p.rapidapi.com
  python3 {os.path.basename(__file__)}

EXACTLY WHICH ENDPOINTS THIS SCRIPT WOULD CALL (base {DIRECT_BASE})
-------------------------------------------------------------------
  1. GET /teams?search=<name>
        -> resolve each national team's numeric id (filter to national=true).
  2. GET /fixtures?team=<id>&season=<year>            (seasons {SEASONS})
        -> list that team's fixtures; we keep the most recent ~{TARGET_FIXTURES}
           finished matches in 2024-2026.
  3. GET /fixtures/statistics?fixture=<fixture_id>
        -> per-team stats for each kept fixture: Total Shots, Shots on Goal,
           Corner Kicks, Fouls, Offsides, Yellow Cards, Red Cards.

REQUEST-BUDGET ESTIMATE (all {n_teams} gap teams)
-------------------------------------------------
  per team : 1 (teams) + {len(SEASONS)} (fixtures lists) + {TARGET_FIXTURES} (statistics) = {b['per_team']}
  totals   : {b['teams_resolve']} teams-resolve  +  {b['fixtures_list']} fixtures-list  +  {b['fixture_statistics']} fixture-statistics
  GRAND TOTAL : ~{b['total']} requests
  On the free tier (~{free_daily} req/day) that is roughly {days} day(s) of quota.
  Tip: run a few teams at a time with --teams "Norway,Scotland" to stay under
  the daily cap, or spread the 18 teams across {days} days.

(Use --dry-run at any time to print this plan without calling the API.)
=============================================================================
""".rstrip()
    )


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class APIFootball:
    """Thin client with rate-limit awareness for both transports."""

    def __init__(self, key: str, host: str | None):
        self.key = key
        if host and "rapidapi" in host.lower():
            self.base = f"https://{host}/v3"
            self.headers = {"x-rapidapi-key": key, "x-rapidapi-host": host}
            self.transport = f"RapidAPI ({host})"
        elif host:
            # treat a non-rapidapi host as a direct api-sports.io base host.
            # accept forms like "v3.football.api-sports.io" or ".../v3" or a
            # full "https://..." and normalise to a "https://<host>/v3" base.
            h = host.strip().rstrip("/")
            if h.startswith("http://") or h.startswith("https://"):
                h = h.split("://", 1)[1]
            h = h.rstrip("/")
            if not h.endswith("/v3"):
                h = h + "/v3"
            self.base = f"https://{h}"
            self.headers = {"x-apisports-key": key}
            self.transport = f"api-sports.io direct ({host})"
        else:
            self.base = DIRECT_BASE
            self.headers = {"x-apisports-key": key}
            self.transport = "api-sports.io direct"
        self.session = requests.Session()
        self.requests_made = 0

    def get(self, path: str, params: dict, retries: int = 3) -> dict | None:
        """GET <base>/<path>; return the parsed JSON 'response' list-wrapper or None.

        Robust to rate limits (HTTP 429 / daily-cap message) and transient errors.
        """
        url = f"{self.base}/{path.lstrip('/')}"
        for attempt in range(retries):
            # proactive pacing to respect the 10 requests/minute free-plan cap
            gap = time.time() - _PACE[0]
            if gap < _MIN_INTERVAL:
                time.sleep(_MIN_INTERVAL - gap)
            _PACE[0] = time.time()
            try:
                r = self.session.get(url, headers=self.headers, params=params, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"    network error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            self.requests_made += 1

            if r.status_code == 429:
                wait = 20 * (attempt + 1)   # 10/min cap — wait out the window
                print(f"    HTTP 429 rate-limited; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"    HTTP {r.status_code} on {path}: {r.text[:200]}",
                      file=sys.stderr)
                return None

            try:
                data = r.json()
            except ValueError:
                print(f"    non-JSON response on {path}", file=sys.stderr)
                return None

            # API-Football reports daily-cap / auth issues inside "errors"
            errors = data.get("errors")
            if errors:
                # errors may be a dict or a list depending on transport
                etext = errors if isinstance(errors, (list, str)) else list(errors.values())
                msg = str(etext).lower()
                if "rate" in msg or "limit" in msg or "requests" in msg:
                    print(f"    API rate/quota limit reached: {etext}", file=sys.stderr)
                    return "RATELIMIT"          # signal caller to stop cleanly
                print(f"    API error on {path}: {etext}", file=sys.stderr)
                return None
            return data
        print(f"    giving up on {path} after {retries} attempts", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# fetch + parse helpers
# --------------------------------------------------------------------------- #
# how API-Football labels the statistics we want -> our internal stat keys
STAT_LABEL_MAP = {
    "total shots": "shots",
    "shots on goal": "sot",
    "corner kicks": "corners",
    "fouls": "fouls",
    "offsides": "offsides",
    # cards handled specially (yellow + red summed)
}


def resolve_team_id(api: APIFootball, search: str):
    data = api.get("teams", {"search": search})
    if data in (None, "RATELIMIT"):
        return data
    resp = data.get("response", [])
    # prefer an explicit national team
    nationals = [x for x in resp if x.get("team", {}).get("national")]
    pick = (nationals or resp)
    if not pick:
        return None
    return pick[0]["team"]["id"]


def recent_fixtures(api: APIFootball, team_id: int):
    """Return a list of finished fixtures (most recent first) across SEASONS."""
    fixtures = []
    for season in SEASONS:
        data = api.get("fixtures", {"team": team_id, "season": season})
        if data == "RATELIMIT":
            return "RATELIMIT"
        if data is None:
            continue
        for fx in data.get("response", []):
            status = fx.get("fixture", {}).get("status", {}).get("short", "")
            if status in ("FT", "AET", "PEN"):          # finished only
                fixtures.append(fx)
    # sort most-recent first by timestamp
    fixtures.sort(key=lambda f: f.get("fixture", {}).get("timestamp", 0), reverse=True)
    return fixtures[:TARGET_FIXTURES]


def parse_fixture_stats(stats_response: list) -> dict:
    """From /fixtures/statistics response, return {team_id: {stat: value}}."""
    out = {}
    for team_block in stats_response:
        tid = team_block.get("team", {}).get("id")
        if tid is None:
            continue
        s = defaultdict(float)
        cards = 0.0
        for item in team_block.get("statistics", []):
            label = (item.get("type") or "").strip().lower()
            val = item.get("value")
            if val is None:
                continue
            # values may be "55%" or ints; coerce to a number, drop %s
            if isinstance(val, str):
                val = val.replace("%", "").strip()
                try:
                    val = float(val)
                except ValueError:
                    continue
            if label in STAT_LABEL_MAP:
                s[STAT_LABEL_MAP[label]] += float(val)
            elif label in ("yellow cards", "red cards"):
                cards += float(val)
        s["cards"] = cards
        out[tid] = s
    return out


# --------------------------------------------------------------------------- #
# CSV upsert
# --------------------------------------------------------------------------- #
def load_existing() -> dict:
    rows = {}
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["team"]] = r
    return rows


def shrink(raw_sum: float, n: int, stat: str) -> float:
    denom = n + PRIOR_W
    if denom <= 0:
        return PRIOR[stat]
    return (raw_sum + PRIOR[stat] * PRIOR_W) / denom


def write_csv(rows: dict) -> None:
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for team in sorted(rows):
            # only write the canonical fields, in order
            row = rows[team]
            w.writerow({k: row.get(k, "") for k in FIELDS})


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teams", default="",
                    help="comma-separated subset of the gap teams (default: all 18)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan/budget and exit without calling the API")
    args = ap.parse_args()

    # which teams
    if args.teams.strip():
        requested = {t.strip() for t in args.teams.split(",") if t.strip()}
        teams = [t for t in GAP_TEAMS if t in requested]
        unknown = requested - set(GAP_TEAMS)
        if unknown:
            print(f"WARNING: ignoring non-gap team(s): {sorted(unknown)}",
                  file=sys.stderr)
    else:
        teams = list(GAP_TEAMS)
    if not teams:
        print("No valid gap teams selected; nothing to do.", file=sys.stderr)
        return 0

    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    host = os.environ.get("API_FOOTBALL_HOST", "").strip() or None

    # ---- no-key path (the verified path) ---------------------------------- #
    if not key:
        print_no_key_instructions(len(teams))
        return 0

    if args.dry_run:
        b = budget_for(len(teams))
        print(f"DRY RUN — would process {len(teams)} team(s); "
              f"~{b['total']} requests (per team {b['per_team']}). No API calls made.")
        return 0

    # ---- keyed path ------------------------------------------------------- #
    api = APIFootball(key, host)
    print(f"Using transport: {api.transport}", file=sys.stderr)
    existing = load_existing()

    for team in teams:
        search = API_SEARCH.get(team, team)
        print(f"[{team}] resolving id (search='{search}')…", file=sys.stderr)
        tid = resolve_team_id(api, search)
        if tid == "RATELIMIT":
            print("  rate/quota limit hit — stopping cleanly; partial results saved.",
                  file=sys.stderr)
            break
        if not tid:
            print(f"  could not resolve team id for '{team}'; leaving priors.",
                  file=sys.stderr)
            continue

        fixtures = recent_fixtures(api, tid)
        if fixtures == "RATELIMIT":
            print("  rate/quota limit hit — stopping cleanly; partial results saved.",
                  file=sys.stderr)
            break
        if not fixtures:
            print(f"  no finished fixtures found for '{team}'; leaving priors.",
                  file=sys.stderr)
            continue

        agg = defaultdict(float)      # this team's "for" sums
        opp_agg = defaultdict(float)  # this team's "against" sums
        goals_for = goals_against = 0.0
        n = 0
        rate_limited = False

        for fx in fixtures:
            fid = fx.get("fixture", {}).get("id")
            if fid is None:
                continue
            data = api.get("fixtures/statistics", {"fixture": fid})
            if data == "RATELIMIT":
                rate_limited = True
                break
            if data is None:
                continue
            per_team = parse_fixture_stats(data.get("response", []))
            if tid not in per_team:
                continue  # no stats published for this fixture (common; skip)
            opp_ids = [k for k in per_team if k != tid]
            n += 1
            for k, v in per_team[tid].items():
                agg[k] += v
            for oid in opp_ids:
                for k, v in per_team[oid].items():
                    opp_agg[k] += v
                opp_agg["fouls_drawn"] += per_team[oid].get("fouls", 0)
            # goals from the fixture score
            goals = fx.get("goals", {})
            home_id = fx.get("teams", {}).get("home", {}).get("id")
            hg = goals.get("home") or 0
            ag = goals.get("away") or 0
            if home_id == tid:
                goals_for += hg
                goals_against += ag
            else:
                goals_for += ag
                goals_against += hg

        print(f"  aggregated {n} fixture(s) with statistics", file=sys.stderr)

        if n == 0:
            print(f"  no fixtures with usable statistics; leaving priors.",
                  file=sys.stderr)
            if rate_limited:
                break
            continue

        # build/overwrite this team's row — keep goals from existing file if present
        prev = existing.get(team, {})
        row = {"team": team, "n_matches": n}
        for s in STATS:
            if s == "goals":
                # keep the already-real goals from the existing file when available
                if prev.get("goals_for") not in (None, ""):
                    row["goals_for"] = prev["goals_for"]
                    row["goals_against"] = prev["goals_against"]
                else:
                    row["goals_for"] = round(shrink(goals_for, n, "goals"), 3)
                    row["goals_against"] = round(shrink(goals_against, n, "goals"), 3)
            elif s == "pens":
                # API-Football fixture stats don't expose penalties → keep prior
                row["pens_for"] = prev.get("pens_for", PRIOR["pens"])
                row["pens_against"] = prev.get("pens_against", PRIOR["pens"])
            else:
                row[f"{s}_for"] = round(shrink(agg.get(s, 0.0), n, s), 3)
                row[f"{s}_against"] = round(shrink(opp_agg.get(s, 0.0), n, s), 3)
        existing[team] = row
        # persist after each team so a mid-run rate-limit still saves progress
        write_csv(existing)

        if rate_limited:
            print("  rate/quota limit hit — stopping cleanly; partial results saved.",
                  file=sys.stderr)
            break

    print(f"Done. {api.requests_made} API request(s) made. Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
