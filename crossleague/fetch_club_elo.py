"""
fetch_club_elo.py — Download Club Elo ratings for all Big 5 teams

Club Elo (clubelo.com) provides free daily Elo ratings for European clubs
going back to 1960. We use these as priors for teams with few CL appearances.

API endpoints:
  http://api.clubelo.com/{TeamName}    — full history for one team
  http://api.clubelo.com/{YYYY-MM-DD}  — all ratings on a specific date

Output:
  crossleague/data/club_elo.csv        — latest ratings for all Big 5 teams
  crossleague/data/club_elo_history.csv — full history (large file)

Usage:
    python3.10 crossleague/scrapers/fetch_club_elo.py
    python3.10 crossleague/scrapers/fetch_club_elo.py --latest-only
    python3.10 crossleague/scrapers/fetch_club_elo.py --date 2024-08-01
"""

import os, time, argparse
from datetime import date, datetime
import requests
import pandas as pd
from io import StringIO

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "crossleague/data"
SLEEP        = 1.0   # seconds between requests

# All teams across Big 5 leagues — Club Elo uses English names mostly
# Format: (club_elo_name, our_team_name, league_div)
# Club Elo names sometimes differ from football-data.co.uk names
BIG5_TEAMS = [
    # Premier League
    ("Arsenal",        "Arsenal",        "E0"),
    ("Chelsea",        "Chelsea",        "E0"),
    ("Liverpool",      "Liverpool",      "E0"),
    ("ManCity",        "Man City",       "E0"),
    ("ManUnited",      "Man United",     "E0"),
    ("Tottenham",      "Tottenham",      "E0"),
    ("Newcastle",      "Newcastle",      "E0"),
    ("AstonVilla",     "Aston Villa",    "E0"),
    ("Brighton",       "Brighton",       "E0"),
    ("WestHam",        "West Ham",       "E0"),
    ("Brentford",      "Brentford",      "E0"),
    ("Fulham",         "Fulham",         "E0"),
    ("CrystalPalace",  "Crystal Palace", "E0"),
    ("Everton",        "Everton",        "E0"),
    ("Wolves",         "Wolves",         "E0"),
    ("NottmForest",    "Nott'm Forest",  "E0"),
    ("Bournemouth",    "Bournemouth",    "E0"),
    ("Leeds",          "Leeds",          "E0"),
    ("Burnley",        "Burnley",        "E0"),
    ("Sunderland",     "Sunderland",     "E0"),
    # Bundesliga
    ("Bayern",         "Bayern Munich",  "D1"),
    ("Dortmund",       "Dortmund",       "D1"),
    ("Leverkusen",     "Leverkusen",     "D1"),
    ("Leipzig",        "RB Leipzig",     "D1"),
    ("Frankfurt",      "Ein Frankfurt",  "D1"),
    ("Stuttgart",      "Stuttgart",      "D1"),
    ("Freiburg",       "Freiburg",       "D1"),
    ("Gladbach",       "M'gladbach",     "D1"),
    ("Hoffenheim",     "Hoffenheim",     "D1"),
    ("Wolfsburg",      "Wolfsburg",      "D1"),
    ("Mainz",          "Mainz",          "D1"),
    ("Augsburg",       "Augsburg",       "D1"),
    ("Union Berlin",   "Union Berlin",   "D1"),
    ("Werder",         "Werder Bremen",  "D1"),
    ("Koeln",          "FC Koln",        "D1"),
    ("Heidenheim",     "Heidenheim",     "D1"),
    # La Liga
    ("Barcelona",      "Barcelona",      "SP1"),
    ("RealMadrid",     "Real Madrid",    "SP1"),
    ("Atletico",       "Ath Madrid",     "SP1"),
    ("Sevilla",        "Sevilla",        "SP1"),
    ("Villarreal",     "Villarreal",     "SP1"),
    ("RealSociedad",   "Sociedad",       "SP1"),
    ("AthleticClub",   "Ath Bilbao",     "SP1"),
    ("Betis",          "Betis",          "SP1"),
    ("Osasuna",        "Osasuna",        "SP1"),
    ("Valencia",       "Valencia",       "SP1"),
    ("Getafe",         "Getafe",         "SP1"),
    ("Girona",         "Girona",         "SP1"),
    # Serie A
    ("Inter",          "Inter",          "I1"),
    ("Juventus",       "Juventus",       "I1"),
    ("ACMilan",        "Milan",          "I1"),
    ("Napoli",         "Napoli",         "I1"),
    ("Roma",           "Roma",           "I1"),
    ("Atalanta",       "Atalanta",       "I1"),
    ("Lazio",          "Lazio",          "I1"),
    ("Fiorentina",     "Fiorentina",     "I1"),
    ("Bologna",        "Bologna",        "I1"),
    ("Torino",         "Torino",         "I1"),
    # Ligue 1
    ("PSG",            "Paris SG",       "F1"),
    ("Monaco",         "Monaco",         "F1"),
    ("Marseille",      "Marseille",      "F1"),
    ("Lyon",           "Lyon",           "F1"),
    ("Lille",          "Lille",          "F1"),
    ("Rennes",         "Rennes",         "F1"),
    ("Nice",           "Nice",           "F1"),
    ("Lens",           "Lens",           "F1"),
    ("Strasbourg",     "Strasbourg",     "F1"),
    ("Nantes",         "Nantes",         "F1"),
]

API_BASE = "http://api.clubelo.com"


def fetch_team_history(club_elo_name, session):
    """Fetch full Elo history for one team."""
    url = f"{API_BASE}/{club_elo_name}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"    Error fetching {club_elo_name}: {e}")
        return None


def fetch_date_snapshot(date_str, session):
    """Fetch all Elo ratings on a specific date."""
    url = f"{API_BASE}/{date_str}"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        return df
    except Exception as e:
        print(f"    Error fetching snapshot {date_str}: {e}")
        return None


def main(latest_only=False, date_str=None, history=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research project)"})

    # ── Option 1: fetch latest snapshot (fastest) ─────────────────────────────
    today = date_str or date.today().strftime("%Y-%m-%d")
    print(f"Fetching Club Elo snapshot for {today}...")
    snapshot = fetch_date_snapshot(today, session)
    if snapshot is not None:
        snapshot.to_csv(f"{OUTPUT_DIR}/club_elo_snapshot_{today}.csv", index=False)
        print(f"  Snapshot: {len(snapshot)} clubs → "
              f"{OUTPUT_DIR}/club_elo_snapshot_{today}.csv")

        # Filter to Big 5 teams and add our name mapping
        elo_names = {row[0]: row[1] for row in BIG5_TEAMS}
        div_map   = {row[0]: row[2] for row in BIG5_TEAMS}

        if "Club" in snapshot.columns and "Elo" in snapshot.columns:
            big5 = snapshot[snapshot["Club"].isin(elo_names.keys())].copy()
            big5["our_name"] = big5["Club"].map(elo_names)
            big5["Div"]      = big5["Club"].map(div_map)
            big5.to_csv(f"{OUTPUT_DIR}/club_elo_latest.csv", index=False)
            print(f"  Big 5 filtered: {len(big5)} teams → "
                  f"{OUTPUT_DIR}/club_elo_latest.csv")
            print(big5[["Club","our_name","Div","Elo"]]
                  .sort_values("Elo", ascending=False)
                  .head(20).to_string(index=False))

    if latest_only:
        return

    # ── Option 2: fetch season-start snapshots (for historical priors) ────────
    # Get ratings at start of each season (Aug 1) — used as preseason priors
    print("\nFetching season-start snapshots...")
    season_starts = [
        ("2015-16", "2015-08-01"),
        ("2016-17", "2016-08-01"),
        ("2017-18", "2017-08-01"),
        ("2018-19", "2018-08-01"),
        ("2019-20", "2019-08-01"),
        ("2020-21", "2020-09-01"),   # COVID — season started Sep
        ("2021-22", "2021-08-01"),
        ("2022-23", "2022-08-01"),
        ("2023-24", "2023-08-01"),
        ("2024-25", "2024-08-01"),
        ("2025-26", "2025-08-01"),
    ]

    season_frames = []
    for season, date_s in season_starts:
        print(f"  {season} ({date_s})...")
        snap = fetch_date_snapshot(date_s, session)
        if snap is not None:
            snap["season"] = season
            season_frames.append(snap)
            print(f"    {len(snap)} clubs")
        time.sleep(SLEEP)

    if season_frames:
        hist = pd.concat(season_frames, ignore_index=True)
        hist.to_csv(f"{OUTPUT_DIR}/club_elo_season_starts.csv", index=False)
        print(f"\nSeason starts: {len(hist)} rows → "
              f"{OUTPUT_DIR}/club_elo_season_starts.csv")

    # ── Option 3: full per-team history (slow) ────────────────────────────────
    if history:
        print("\nFetching full history per team (this will take ~5 minutes)...")
        all_frames = []
        for elo_name, our_name, div in BIG5_TEAMS:
            print(f"  {our_name} ({elo_name})...")
            df = fetch_team_history(elo_name, session)
            if df is not None:
                df["our_name"] = our_name
                df["Div"]      = div
                all_frames.append(df)
                print(f"    {len(df)} rows")
            time.sleep(SLEEP)

        if all_frames:
            hist_df = pd.concat(all_frames, ignore_index=True)
            hist_df.to_csv(f"{OUTPUT_DIR}/club_elo_history.csv", index=False)
            print(f"\nFull history: {len(hist_df)} rows → "
                  f"{OUTPUT_DIR}/club_elo_history.csv")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-only", action="store_true",
                        help="Only fetch today's snapshot (fastest)")
    parser.add_argument("--date", default=None,
                        help="Specific date YYYY-MM-DD for snapshot")
    parser.add_argument("--history", action="store_true",
                        help="Also fetch full per-team history (slow)")
    args = parser.parse_args()
    main(args.latest_only, args.date, args.history)