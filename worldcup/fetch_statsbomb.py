"""
fetch_statsbomb.py — Exact per-team match-stat rates from StatsBomb open data.

StatsBomb publishes full event data for the men's World Cup 2018 & 2022 (free, no
auth). From the event stream we compute, per team, per-match rates for every market
the model cares about — for AND against:

    goals, shots, shots on target, corners, offsides (caught), fouls committed,
    fouls drawn, yellow/red cards, penalties won/conceded.

These rates become team "ratings" for the unpriced markets (offsides, fouls) and a
cross-check/blend for the priced ones. Each team's raw rate is shrunk toward a
tournament base-rate prior (few matches per team -> regularise), exactly like the
preseason-prior logic in the club model.

Usage:
  python3 fetch_statsbomb.py --teams "South Korea,Portugal"   # specific teams
  python3 fetch_statsbomb.py --all                            # every WC18+22 team
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENT_CACHE = os.path.join(DATA, "sb_events")
OUT = os.path.join(DATA, "team_stats.csv")
os.makedirs(EVENT_CACHE, exist_ok=True)

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SEASONS = [(43, 106), (43, 3)]   # World Cup 2022, 2018

# tournament base-rate priors (per team per match) for shrinkage of thin samples
PRIOR = dict(goals=1.35, shots=11.5, sot=4.0, corners=4.8, offsides=1.8,
             fouls=11.5, fouls_drawn=11.5, cards=2.0, pens=0.10)
PRIOR_W = 3.0   # equivalent matches of prior weight


def _get_json(url, cache=None):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
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
    """Return per-team counts for one match: {team_name: {stat: count}}."""
    from collections import defaultdict
    st = defaultdict(lambda: defaultdict(float))
    for e in events:
        t = e.get("team", {}).get("name")
        if not t:
            continue
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
            # most offsides are encoded as the pass outcome, not a standalone event
            if p.get("outcome", {}).get("name") == "Pass Offside":
                st[t]["offsides"] += 1       # passing team caught offside
        elif typ == "Offside":
            st[t]["offsides"] += 1           # standalone offside (rarer encoding)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default="", help="comma-separated team names")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    targets = {t.strip() for t in args.teams.split(",") if t.strip()}

    matches = load_matches()
    # which matches to process
    use = []
    for m in matches:
        h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        if args.all or not targets or (h in targets or a in targets):
            use.append(m)
    print(f"Processing {len(use)} matches…", file=sys.stderr)

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(float))   # team -> stat -> sum
    n_match = defaultdict(int)
    opp_agg = defaultdict(lambda: defaultdict(float))  # team -> stat conceded (opponent's for)

    for i, m in enumerate(use):
        mid = m["match_id"]
        h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        try:
            ev = _get_json(f"{RAW}/events/{mid}.json", os.path.join(EVENT_CACHE, f"{mid}.json"))
        except Exception as e:
            print(f"  skip {mid}: {e}", file=sys.stderr); continue
        st = parse_event_file(ev)
        # goals from final score (more reliable than counting)
        hg, ag = m.get("home_score", 0), m.get("away_score", 0)
        st[h]["goals"] = max(st[h]["goals"], hg); st[a]["goals"] = max(st[a]["goals"], ag)
        for team, opp in ((h, a), (a, h)):
            n_match[team] += 1
            for k, v in st[team].items():
                agg[team][k] += v
            # opponent's "for" stats become this team's "against"
            for k, v in st[opp].items():
                opp_agg[team][k] += v
            opp_agg[team]["fouls_drawn"] += st[opp].get("fouls", 0)
        print(f"  [{i+1}/{len(use)}] {h} {hg}-{ag} {a}", file=sys.stderr)

    # shrink each rate toward prior, write CSV
    import csv
    stats = ["goals", "shots", "sot", "corners", "offsides", "fouls", "cards", "pens"]
    rows = []
    teams = sorted(set(list(agg) + list(targets)))
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

    # merge with any existing file (so incremental --teams runs accumulate)
    existing = {}
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            existing[r["team"]] = r
    for r in rows:
        existing[r["team"]] = {k: str(v) for k, v in r.items()}
    fields = ["team", "n_matches"] + [f"{s}_{d}" for s in stats for d in ("for", "against")]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for t in sorted(existing): w.writerow(existing[t])
    print(f"Wrote {len(existing)} teams -> {OUT}")


if __name__ == "__main__":
    main()
