"""
analyze_players.py — Player-level statistics from CACHED StatsBomb event data.

Pure local parsing of data/sb_events/*.json (World Cup 2018 + 2022, 128 match
files). NO network. Produces:

  1. data/player_stats.csv     — one row per (player, team) with raw + per90 stats.
  2. data/player_rankings.csv  — per team top-5 attackers & top-5 defenders.

NOTE: this only covers players who featured at WC2018/2022, so current squads
are partially stale — that is acceptable for this model.
"""
from __future__ import annotations
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENT_CACHE = os.path.join(DATA, "sb_events")
OUT_STATS = os.path.join(DATA, "player_stats.csv")
OUT_RANK = os.path.join(DATA, "player_rankings.csv")

SOT_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
DEF_POS_KEYS = ("Back", "Defender", "Defensive Midfield")


def _name(d, *path):
    """Safe nested .get() chain returning '' on any missing/None link."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
        if cur is None:
            return ""
    return cur if cur is not None else ""


def new_player():
    return defaultdict(float)


def parse_match(events, agg, match_minutes, match_positions):
    """Accumulate per-(player,team) stats from one match's event list.

    agg: dict[(player,team)] -> defaultdict(float) of running totals (cross-match)
    match_minutes / match_positions are reset per match by the caller.
    """
    for e in events:
        if not isinstance(e, dict):
            continue
        team = _name(e, "team", "name")
        player = _name(e, "player", "name")
        if not team or not player:
            continue
        key = (player, team)
        st = agg[key]

        # minutes tracking: first/last event minute for this player this match
        minute = e.get("minute")
        if isinstance(minute, (int, float)):
            mm = match_minutes[key]
            if mm[0] is None or minute < mm[0]:
                mm[0] = minute
            if mm[1] is None or minute > mm[1]:
                mm[1] = minute

        pos = _name(e, "position", "name")
        if pos:
            match_positions[key][pos] = match_positions[key].get(pos, 0) + 1

        typ = _name(e, "type", "name")

        if typ == "Shot":
            sh = e.get("shot", {}) or {}
            st["shots"] += 1
            xg = sh.get("statsbomb_xg")
            if isinstance(xg, (int, float)):
                st["xg"] += xg
            out = _name(sh, "outcome", "name")
            if out in SOT_OUTCOMES:
                st["sot"] += 1
            if out == "Goal":
                st["goals"] += 1
        elif typ == "Pass":
            p = e.get("pass", {}) or {}
            if p.get("shot_assist"):
                st["key_passes"] += 1
            if p.get("goal_assist"):
                st["assists"] += 1
                st["key_passes"] += 1  # a goal assist is also a key pass
        elif typ == "Foul Committed":
            st["fouls_committed"] += 1
            card = _name(e, "foul_committed", "card", "name")
            if "Yellow" in card or "Red" in card:
                st["cards"] += 1
        elif typ == "Foul Won":
            st["fouls_won"] += 1
        elif typ == "Bad Behaviour":
            card = _name(e, "bad_behaviour", "card", "name")
            if "Yellow" in card or "Red" in card:
                st["cards"] += 1
        elif typ == "Interception":
            st["interceptions"] += 1
        elif typ == "Block":
            st["blocks"] += 1
        elif typ == "Clearance":
            st["clearances"] += 1
        elif typ == "Ball Recovery":
            st["recoveries"] += 1
        elif typ == "Duel":
            if _name(e, "duel", "type", "name") == "Tackle":
                st["tackles"] += 1


def main():
    files = sorted(
        f for f in glob.glob(os.path.join(EVENT_CACHE, "*.json"))
        if not os.path.basename(f).startswith("matches_")
    )
    print(f"Parsing {len(files)} event files...")

    agg = defaultdict(new_player)            # (player,team) -> stat totals
    minutes_total = defaultdict(float)       # (player,team) -> summed minutes
    matches_count = defaultdict(int)         # (player,team) -> matches appeared
    pos_total = defaultdict(lambda: defaultdict(int))  # (player,team) -> pos -> count

    for fp in files:
        try:
            events = json.load(open(fp))
        except Exception as ex:
            print(f"  skip {os.path.basename(fp)}: {ex}")
            continue
        if not isinstance(events, list):
            continue
        match_minutes = defaultdict(lambda: [None, None])
        match_positions = defaultdict(dict)
        parse_match(events, agg, match_minutes, match_positions)
        # roll up minutes & match appearances for this match
        for key, (lo, hi) in match_minutes.items():
            matches_count[key] += 1
            if lo is not None and hi is not None:
                # at least 1 minute so we never divide by zero downstream
                minutes_total[key] += max(hi - lo, 1)
        for key, posmap in match_positions.items():
            for pos, c in posmap.items():
                pos_total[key][pos] += c

    # ---- build player_stats rows ----
    stat_cols = [
        "goals", "xg", "shots", "sot", "assists", "key_passes",
        "fouls_committed", "fouls_won", "cards",
        "tackles", "interceptions", "blocks", "clearances", "recoveries",
    ]
    rows = []
    for (player, team), st in agg.items():
        minutes = minutes_total[(player, team)]
        matches = matches_count[(player, team)]
        per90 = (90.0 / minutes) if minutes > 0 else 0.0
        def_actions = (st["tackles"] + st["interceptions"]
                       + st["blocks"] + st["clearances"])
        row = {
            "player": player,
            "team": team,
            "matches": matches,
            "minutes": round(minutes, 1),
        }
        for c in stat_cols:
            v = st.get(c, 0.0)
            row[c] = round(v, 4) if c == "xg" else int(v) if float(v).is_integer() else round(v, 2)
        row["goals90"] = round(st["goals"] * per90, 3)
        row["xg90"] = round(st["xg"] * per90, 3)
        row["shots90"] = round(st["shots"] * per90, 3)
        row["sot90"] = round(st["sot"] * per90, 3)
        row["def_actions90"] = round(def_actions * per90, 3)
        # keep a private dominant-position for ranking (not strictly required col)
        dom_pos = ""
        pm = pos_total[(player, team)]
        if pm:
            dom_pos = max(pm.items(), key=lambda kv: kv[1])[0]
        row["_position"] = dom_pos
        rows.append(row)

    fields = (["player", "team", "matches", "minutes"] + stat_cols
              + ["goals90", "xg90", "shots90", "sot90", "def_actions90", "position"])
    with open(OUT_STATS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields if k != "position"}
            out["position"] = r["_position"]
            w.writerow(out)
    print(f"Wrote {len(rows)} (player,team) rows -> {OUT_STATS}")

    # ---- rankings ----
    by_team = defaultdict(list)
    for r in rows:
        by_team[r["team"]].append(r)

    rank_rows = []
    for team, players in by_team.items():
        # attackers: score = 0.5*xg90 + 0.3*goals90 + 0.2*sot90
        for r in players:
            r["_att_score"] = (0.5 * r["xg90"] + 0.3 * r["goals90"]
                               + 0.2 * r["sot90"])
        atts = sorted(players, key=lambda r: r["_att_score"], reverse=True)[:5]
        for i, r in enumerate(atts, 1):
            rank_rows.append({
                "team": team, "role": "attacker", "rank": i,
                "player": r["player"], "score": round(r["_att_score"], 4),
                "key_stat": f"xg90={r['xg90']}, goals90={r['goals90']}, sot90={r['sot90']}",
            })
        # defenders: score = def_actions90, restricted by position
        defs_pool = [r for r in players
                     if any(k in r["_position"] for k in DEF_POS_KEYS)]
        defs = sorted(defs_pool, key=lambda r: r["def_actions90"], reverse=True)[:5]
        for i, r in enumerate(defs, 1):
            rank_rows.append({
                "team": team, "role": "defender", "rank": i,
                "player": r["player"], "score": round(r["def_actions90"], 4),
                "key_stat": (f"pos={r['_position']}, def_actions90={r['def_actions90']}, "
                             f"tk={r['tackles']}, int={r['interceptions']}, "
                             f"blk={r['blocks']}, clr={r['clearances']}"),
            })

    with open(OUT_RANK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team", "role", "rank", "player", "score", "key_stat"])
        w.writeheader()
        for r in rank_rows:
            w.writerow(r)
    print(f"Wrote {len(rank_rows)} ranking rows -> {OUT_RANK}")

    # ---- summary for caller ----
    distinct_players = len({p for (p, t) in agg})
    distinct_teams = len({t for (p, t) in agg})
    top_att = sorted(rank_rows, key=lambda r: (r["role"] == "attacker", r["score"]),
                     reverse=True)
    return rows, rank_rows, distinct_players, distinct_teams


if __name__ == "__main__":
    main()
