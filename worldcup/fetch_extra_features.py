#!/usr/bin/env python3
"""Pull four small, free, no-auth data sources for the World Cup model.

Sources:
  1. FIFA world rankings (hidden inside.fifa.com API)
  2. Venue altitude (Open-Meteo elevation API)
  3. 2026 fixtures + derived rest/congestion/travel features (openfootball)
  4. Goalscorers + shootouts cache (martj42/international_results)

Outputs go to ./data/. Does not modify any existing .py files.
"""
import os
import re
import csv
import json
import time
import math
import io
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Odds API spellings observed in data/wc_weather.csv (match column).
ODDS_SPELLINGS = {
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "USA",
    "United States": "USA",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Cape Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia & Herzegovina": "Bosnia & Herzegovina",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
}


def to_odds(name):
    if name is None:
        return name
    return ODDS_SPELLINGS.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# 1. FIFA RANKINGS
# ---------------------------------------------------------------------------
def fetch_fifa_rankings(max_past=3):
    print("[1] FIFA rankings ...")
    date_ids = []
    try:
        r = requests.get(
            "https://inside.fifa.com/fifa-world-ranking/men", headers=UA, timeout=30
        )
        r.raise_for_status()
        # Each id<digits> token corresponds to a historical release dateId.
        toks = re.findall(r"id\d+", r.text)
        # Preserve order, dedupe, keep ones that look like ranking date ids.
        seen = set()
        for t in toks:
            if t not in seen:
                seen.add(t)
                date_ids.append(t)
    except Exception as e:
        print("   warn: could not scrape dateIds:", e)

    # Always ensure the known-good latest id is first.
    known_latest = "id14870"
    if known_latest in date_ids:
        date_ids.remove(known_latest)
    date_ids.insert(0, known_latest)

    rows = []
    used = []
    for did in date_ids:
        if len(used) >= 1 + max_past:
            break
        url = f"https://inside.fifa.com/api/ranking-overview?locale=en&dateId={did}"
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print(f"   skip {did}: {e}")
            continue
        teams = (j.get("rankings") or j.get("data") or j.get("items") or [])
        if not teams:
            continue
        date_str = None
        for r_ in teams:
            ds = r_.get("rankingItem", {}).get("lastUpdateDate") or r_.get("lastUpdateDate")
            if ds:
                date_str = ds[:10] if isinstance(ds, str) else ds
                break
        n_added = 0
        for r_ in teams:
            item = r_.get("rankingItem", r_) if isinstance(r_.get("rankingItem"), dict) else r_
            name = item.get("name") or item.get("teamName")
            rank = item.get("rank")
            pts = item.get("totalPoints") or item.get("points")
            if name is None or rank is None:
                continue
            rows.append(
                {
                    "team": to_odds(name),
                    "rank": int(rank),
                    "points": round(float(pts), 2) if pts is not None else None,
                    "date": date_str or did,
                }
            )
            n_added += 1
        if n_added:
            used.append(did)
            print(f"   {did} -> {n_added} teams (date={date_str})")
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "fifa_rankings.csv")
    df.to_csv(out, index=False)
    print(f"   wrote {out}: {len(df)} rows, {df['date'].nunique()} releases")
    return df


# ---------------------------------------------------------------------------
# 2. VENUE ALTITUDE
# ---------------------------------------------------------------------------
def fetch_venue_altitude():
    print("[2] Venue altitude ...")
    wx = pd.read_csv(os.path.join(DATA, "wc_weather.csv"))
    venues = wx[["venue", "city", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for _, v in venues.iterrows():
        lat, lon = round(float(v["lat"]), 4), round(float(v["lon"]), 4)
        elev = None
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/elevation",
                params={"latitude": lat, "longitude": lon},
                headers=UA,
                timeout=30,
            )
            r.raise_for_status()
            elev = r.json().get("elevation")
            if isinstance(elev, list):
                elev = elev[0]
        except Exception as e:
            print(f"   warn {v['venue']}: {e}")
        rows.append(
            {
                "venue": v["venue"],
                "city": v["city"],
                "lat": lat,
                "lon": lon,
                "elevation_m": round(float(elev), 1) if elev is not None else None,
            }
        )
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "venue_altitude.csv")
    df.to_csv(out, index=False)
    print(f"   wrote {out}: {len(df)} venues")
    return df


# ---------------------------------------------------------------------------
# 3. 2026 FIXTURES + rest/congestion/travel features
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _norm_city(s):
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", "", s)  # drop parenthetical
    s = re.sub(r"[^a-z ]", "", s)
    return s.strip()


def fetch_fixtures(altitude_df=None):
    print("[3] 2026 fixtures + features ...")
    url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()
    matches = j.get("matches", [])

    def team_name(x):
        if isinstance(x, dict):
            return x.get("name") or x.get("key")
        return x

    sched = []
    for m in matches:
        sched.append(
            {
                "date": m.get("date"),
                "team1": to_odds(team_name(m.get("team1"))),
                "team2": to_odds(team_name(m.get("team2"))),
                "group": m.get("group", ""),
                "ground": (m.get("ground", {}) or {}).get("name")
                if isinstance(m.get("ground"), dict)
                else m.get("ground"),
                "round": m.get("round", ""),
                "time": m.get("time", ""),
            }
        )
    sdf = pd.DataFrame(sched)
    # Filter to rows that have a real date & both teams (skip TBD knockouts).
    out_sched = sdf[["date", "team1", "team2", "group", "ground"]]
    out1 = os.path.join(DATA, "wc_schedule.csv")
    out_sched.to_csv(out1, index=False)
    print(f"   wrote {out1}: {len(out_sched)} matches")

    # ---- build city->lat/lon lookup from altitude file ----
    city_loc = {}
    if altitude_df is None:
        ap = os.path.join(DATA, "venue_altitude.csv")
        if os.path.exists(ap):
            altitude_df = pd.read_csv(ap)
    if altitude_df is not None:
        for _, a in altitude_df.iterrows():
            for key in (_norm_city(a["city"]), _norm_city(a["venue"])):
                if key:
                    city_loc.setdefault(key, (float(a["lat"]), float(a["lon"]), a["city"]))

    def loc_for(ground):
        nc = _norm_city(ground)
        if nc in city_loc:
            return city_loc[nc]
        # partial match
        for k, v in city_loc.items():
            if k and (k in nc or nc in k):
                return v
        return (None, None, ground)

    # ---- per-team long schedule, sorted by date ----
    long_rows = []
    valid = sdf.dropna(subset=["date"]).copy()
    valid["dt"] = pd.to_datetime(valid["date"], errors="coerce")
    valid = valid.dropna(subset=["dt"])
    for _, mt in valid.iterrows():
        lat, lon, city_clean = loc_for(mt["ground"])
        for team, opp in ((mt["team1"], mt["team2"]), (mt["team2"], mt["team1"])):
            if not team or str(team).strip() == "":
                continue
            long_rows.append(
                {
                    "match_date": mt["dt"].date().isoformat(),
                    "dt": mt["dt"],
                    "team": team,
                    "opponent": opp,
                    "ground": mt["ground"],
                    "city": city_clean,
                    "lat": lat,
                    "lon": lon,
                }
            )
    ldf = pd.DataFrame(long_rows).sort_values(["team", "dt"]).reset_index(drop=True)

    feats = []
    for team, grp in ldf.groupby("team"):
        prev = None
        for _, row in grp.iterrows():
            days_rest = None
            travel_km = None
            prev_city = None
            if prev is not None:
                days_rest = (row["dt"] - prev["dt"]).days
                prev_city = prev["city"]
                if all(
                    v is not None and not (isinstance(v, float) and math.isnan(v))
                    for v in (row["lat"], row["lon"], prev["lat"], prev["lon"])
                ):
                    travel_km = round(
                        _haversine(prev["lat"], prev["lon"], row["lat"], row["lon"]), 1
                    )
            feats.append(
                {
                    "match_date": row["match_date"],
                    "team": team,
                    "opponent": row["opponent"],
                    "days_rest": days_rest,
                    "is_back_to_back": (days_rest is not None and days_rest <= 4),
                    "prev_city": prev_city,
                    "city": row["city"],
                    "travel_km": travel_km,
                }
            )
            prev = row

    fdf = pd.DataFrame(feats)
    out2 = os.path.join(DATA, "wc_schedule_features.csv")
    fdf.to_csv(out2, index=False)
    matched = city_loc and fdf["travel_km"].notna().any()
    print(
        f"   wrote {out2}: {len(fdf)} team-match rows "
        f"(travel mapped: {fdf['travel_km'].notna().sum()})"
    )
    return out_sched, fdf


# ---------------------------------------------------------------------------
# 4. GOALSCORERS + SHOOTOUTS cache
# ---------------------------------------------------------------------------
def fetch_companion_files():
    print("[4] goalscorers + shootouts ...")
    base = "https://raw.githubusercontent.com/martj42/international_results/master/"
    results = {}
    for fname in ("goalscorers.csv", "shootouts.csv"):
        r = requests.get(base + fname, headers=UA, timeout=60)
        r.raise_for_status()
        out = os.path.join(DATA, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(r.text)
        df = pd.read_csv(out)
        results[fname] = df
        print(f"   wrote {out}: {len(df)} rows")
    return results


if __name__ == "__main__":
    fifa = fetch_fifa_rankings()
    alt = fetch_venue_altitude()
    sched, feats = fetch_fixtures(alt)
    comp = fetch_companion_files()
    print("\nDONE.")
