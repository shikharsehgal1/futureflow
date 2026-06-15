"""
expand_statsbomb.py — Expand team_stats.csv with more senior tournaments.

Reuses the EXACT parsing + shrinkage logic from fetch_statsbomb.py, but aggregates
each team's raw match counts across ALL of:
    World Cup 2022, World Cup 2018, UEFA Euro 2024, UEFA Euro 2020,
    Copa America 2024, AFCON 2023
then shrinks each rate toward the same priors with the same weight.

Because team_stats.csv stores only shrunk rates (not raw counts), we re-derive
everything from scratch from the (cached) event files, so WC18/22 teams gain
extra matches from the new comps where applicable.

Writes data/team_stats.csv in place (same 18-column schema).
"""
from __future__ import annotations
import json, os, sys, urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENT_CACHE = os.path.join(DATA, "sb_events")
OUT = os.path.join(DATA, "team_stats.csv")
os.makedirs(EVENT_CACHE, exist_ok=True)

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# WC22, WC18  + Euro2024, Euro2020, Copa2024, AFCON2023
SEASONS = [(43, 106), (43, 3), (55, 282), (55, 43), (223, 282), (1267, 107)]

# StatsBomb spells a few teams differently from the WC schedule/extra-stats files
# the model keys on. Normalise so the real rates land under the name the model
# looks up (otherwise the team silently falls back to its team_stats_extra prior).
TEAM_RENAME = {
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
}


def _norm(name):
    return TEAM_RENAME.get(name, name)

# identical priors / weight to fetch_statsbomb.py
PRIOR = dict(goals=1.35, shots=11.5, sot=4.0, corners=4.8, offsides=1.8,
             fouls=11.5, fouls_drawn=11.5, cards=2.0, pens=0.10)
PRIOR_W = 3.0


def _get_json(url, cache=None):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    if cache:
        json.dump(data, open(cache, "w"))
    return data


def load_matches():
    matches = []
    for comp, season in SEASONS:
        c = os.path.join(EVENT_CACHE, f"matches_{comp}_{season}.json")
        matches += _get_json(f"{RAW}/matches/{comp}/{season}.json", c)
    return matches


def parse_event_file(events):
    """Return per-team counts for one match: {team_name: {stat: count}}.
    Logic copied verbatim from fetch_statsbomb.py."""
    st = defaultdict(lambda: defaultdict(float))
    for e in events:
        t = e.get("team", {}).get("name")
        if not t:
            continue
        t = _norm(t)
        typ = e.get("type", {}).get("name")
        if typ == "Shot":
            st[t]["shots"] += 1
            sh = e.get("shot", {})
            out = sh.get("outcome", {}).get("name", "")
            if out in ("Goal", "Saved", "Saved To Post"):
                st[t]["sot"] += 1
            if out == "Goal":
                st[t]["goals"] += 1
            if sh.get("type", {}).get("name") == "Penalty":
                st[t]["pens"] += 1
        elif typ == "Pass":
            p = e.get("pass", {})
            if p.get("type", {}).get("name") == "Corner":
                st[t]["corners"] += 1
            if p.get("outcome", {}).get("name") == "Pass Offside":
                st[t]["offsides"] += 1
        elif typ == "Offside":
            st[t]["offsides"] += 1
        elif typ == "Foul Committed":
            st[t]["fouls"] += 1
            card = e.get("foul_committed", {}).get("card", {}).get("name", "")
            if "Yellow" in card or "Red" in card:
                st[t]["cards"] += 1
        elif typ == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name", "")
            if "Yellow" in card or "Red" in card:
                st[t]["cards"] += 1
    return st


def main():
    matches = load_matches()
    print(f"Processing {len(matches)} matches across {len(SEASONS)} tournaments…",
          file=sys.stderr)

    agg = defaultdict(lambda: defaultdict(float))
    n_match = defaultdict(int)
    opp_agg = defaultdict(lambda: defaultdict(float))

    for i, m in enumerate(matches):
        mid = m["match_id"]
        h, a = _norm(m["home_team"]["home_team_name"]), _norm(m["away_team"]["away_team_name"])
        try:
            ev = _get_json(f"{RAW}/events/{mid}.json",
                           os.path.join(EVENT_CACHE, f"{mid}.json"))
        except Exception as e:
            print(f"  skip {mid}: {e}", file=sys.stderr); continue
        st = parse_event_file(ev)
        hg, ag = m.get("home_score", 0), m.get("away_score", 0)
        st[h]["goals"] = max(st[h]["goals"], hg)
        st[a]["goals"] = max(st[a]["goals"], ag)
        for team, opp in ((h, a), (a, h)):
            n_match[team] += 1
            for k, v in st[team].items():
                agg[team][k] += v
            for k, v in st[opp].items():
                opp_agg[team][k] += v
            opp_agg[team]["fouls_drawn"] += st[opp].get("fouls", 0)
        if (i + 1) % 25 == 0 or i + 1 == len(matches):
            print(f"  [{i+1}/{len(matches)}]", file=sys.stderr)

    import csv
    stats = ["goals", "shots", "sot", "corners", "offsides", "fouls", "cards", "pens"]
    rows = []
    teams = sorted(agg)
    for team in teams:
        n = n_match[team]
        row = {"team": team, "n_matches": n}
        for s in stats:
            raw = agg[team].get(s, 0.0)
            rate = (raw + PRIOR[s] * PRIOR_W) / (n + PRIOR_W) if (n + PRIOR_W) > 0 else PRIOR[s]
            row[s + "_for"] = round(rate, 3)
            craw = opp_agg[team].get(s, 0.0)
            crate = (craw + PRIOR[s] * PRIOR_W) / (n + PRIOR_W) if (n + PRIOR_W) > 0 else PRIOR[s]
            row[s + "_against"] = round(crate, 3)
        rows.append(row)

    # merge with existing (preserve any teams not seen in these comps)
    existing = {}
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            existing[r["team"]] = r
    for r in rows:
        existing[r["team"]] = {k: str(v) for k, v in r.items()}
    fields = ["team", "n_matches"] + [f"{s}_{d}" for s in stats for d in ("for", "against")]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for t in sorted(existing):
            w.writerow(existing[t])
    print(f"Wrote {len(existing)} teams -> {OUT}")


if __name__ == "__main__":
    main()
