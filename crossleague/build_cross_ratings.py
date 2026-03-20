"""
build_cross_ratings.py — Cross-league ratings using CL/Europa/Conference data

Architecture:
  Fits a unified goal-differential rating where teams from all leagues
  are on the same scale. This allows direct comparison: Arsenal +X vs Bayern +Y.

  Three data sources, each with a different weight:
    1. League games        — weight 0.3 (indirect cross-league signal)
    2. CL/Europa/Conf games — weight 1.5 (primary cross-league signal)
    3. Club Elo prior       — weight varies (anchor for teams with few CL games)

  The model adds a league-level offset (delta[league]) so that in-league
  ratings and cross-league ratings are reconciled.

Output:
  crossleague/data/cross_ratings.csv  — unified ratings for all Big 5 teams
  crossleague/data/league_offsets.csv — estimated league difficulty offsets

Usage:
    python3.10 crossleague/build_cross_ratings.py
    python3.10 crossleague/build_cross_ratings.py --season 2025-26
"""

import os, argparse, pickle, warnings
import pandas as pd
import numpy as np
from scipy.special import factorial
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CURRENT_SEASON   = "2025-26"
HALF_LIFE        = 90    # days decay for all games
CL_WEIGHT        = 1.5   # CL/Europa/Conf games weight
LEAGUE_WEIGHT_TIER1 = 0.3  # Top 5 leagues
LEAGUE_WEIGHT_TIER2 = 0.05 # Second divisions — much less signal   # in-league games weight (cross-league signal)
MARKET_WEIGHT    = 0.95  # same as main model
MARKET_HALF_LIFE = 30
MIN_CL_GAMES     = 3     # min CL games before using actual data vs prior only

# League codes
DIVS = ["E0","D1","SP1","I1","F1","E1","D2","SP2","I2","F2"]
TIER1 = {"E0","D1","SP1","I1","F1"}

DIV_NAMES = {
    "E0":"Premier League","E1":"Championship",
    "D1":"Bundesliga","D2":"2. Bundesliga",
    "SP1":"La Liga","SP2":"Segunda División",
    "I1":"Serie A","I2":"Serie B",
    "F1":"Ligue 1","F2":"Ligue 2",
}

# Team name mapping: CL data names -> our league data names
TEAM_MAP = {
    "AC Milan":           "Milan",
    "AS Roma":            "Roma",
    "Atl. Madrid":        "Ath Madrid",
    "Manchester City":    "Man City",
    "Manchester Utd":     "Man United",
    "PSG":                "Paris SG",
    "Paris Saint-Germain":"Paris SG",
    "B. Monchengladbach": "M'gladbach",
    "Borussia Dortmund":  "Dortmund",
    "Bayer Leverkusen":   "Leverkusen",
    "Atletico Madrid":    "Ath Madrid",
    "Atlético Madrid":    "Ath Madrid",
    "Eintracht Frankfurt":"Ein Frankfurt",
    "Newcastle United":   "Newcastle",
    "Nottingham Forest":  "Nott'm Forest",
    "Tottenham Hotspur":  "Tottenham",
    "Tottenham":          "Tottenham",
    "West Ham United":    "West Ham",
    "Wolverhampton":      "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "RB Leipzig":         "RB Leipzig",
    "Sociedad":           "Sociedad",
    "Real Sociedad":      "Sociedad",
    "Athletic Bilbao":    "Ath Bilbao",
    "Athletic Club":      "Ath Bilbao",
    "Ath. Bilbao":        "Ath Bilbao",
    "Celta Vigo":         "Celta",
    "Celta de Vigo":      "Celta",
    "Nott'm Forest":      "Nott'm Forest",
    "Aston Villa":        "Aston Villa",
    "Crystal Palace":     "Crystal Palace",
}


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    # European competition data
    cl_df = pd.read_csv("crossleague/data/european_combined.csv")
    cl_df["Date"] = pd.to_datetime(cl_df["Date"], errors="coerce")
    cl_df["HomeGoals"] = pd.to_numeric(cl_df["HomeGoals"], errors="coerce")
    cl_df["AwayGoals"] = pd.to_numeric(cl_df["AwayGoals"], errors="coerce")
    cl_df["GoalDiff"]  = cl_df["HomeGoals"] - cl_df["AwayGoals"]

    # Normalise team names
    cl_df["HomeTeam"] = cl_df["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t))
    cl_df["AwayTeam"] = cl_df["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t))

    # Compute mu_mkt from odds
    cl_df = add_mu_mkt(cl_df)

    # League data
    lg_df = pd.read_csv("data/big5_with_probs.csv", low_memory=False)
    lg_df["Date"]     = pd.to_datetime(lg_df["Date"], errors="coerce")
    lg_df["GoalDiff"] = lg_df["HomeGoals"] - lg_df["AwayGoals"]

    return cl_df, lg_df


def add_mu_mkt(df):
    """Convert H/D/A odds to implied goal differential (mu_mkt)."""
    # Same probit approach as main model
    # Load fitted probit if available, otherwise use simple log-odds approximation
    try:
        with open("data/probit_reg.pkl", "rb") as f:
            probit = pickle.load(f)
        # probit expects [log(H/A), log(D)] style features — use simple approx
    except Exception:
        probit = None

    def odds_to_mu(row):
        try:
            h, d, a = float(row["OddsH"]), float(row["OddsD"]), float(row["OddsA"])
            if np.isnan(h) or np.isnan(a) or h <= 1 or a <= 1:
                return np.nan
            # Normalise
            total = 1/h + 1/d + 1/a
            ph = (1/h) / total
            pa = (1/a) / total
            # Log odds as proxy for goal diff (calibrated via main model)
            # mu = log(ph/pa) * 0.8 is a reasonable approximation
            mu = np.log(ph / pa) * 0.8
            return round(mu, 4)
        except Exception:
            return np.nan

    df = df.copy()
    df["mu_mkt"] = df.apply(odds_to_mu, axis=1)
    return df


# ── FIT CROSS-LEAGUE RATINGS ──────────────────────────────────────────────────
def fit_cross_ratings(cl_df, lg_df, season=CURRENT_SEASON,
                      ref_date=None):
    """
    Fit unified cross-league ratings using:
    - CL/Europa/Conf games (weight=1.5) from cl_df
    - League games (weight=0.3) from lg_df for Big 5 teams
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today()
    ref_date = pd.Timestamp(ref_date)

    # Filter to relevant seasons (rolling 3 years for cross-league)
    seasons_to_use = _recent_seasons(season, n=3)
    cl_sub = cl_df[
        cl_df["season"].isin(seasons_to_use) &
        cl_df["GoalDiff"].notna() &
        (cl_df["Date"] <= ref_date)
    ].copy()

    # League games for Big 5 teams only, current season
    lg_sub = lg_df[
        (lg_df["season"] == season) &
        lg_df["GoalDiff"].notna() &
        (lg_df["Date"] <= ref_date)
    ].copy()

    if len(cl_sub) == 0:
        print("  No CL data for this period")
        return None

    # All teams appearing in CL data
    cl_teams = sorted(set(cl_sub["HomeTeam"]) | set(cl_sub["AwayTeam"]))
    # Big 5 teams from league data
    lg_teams = sorted(set(lg_sub["HomeTeam"]) | set(lg_sub["AwayTeam"]))
    # Union — all teams we'll rate
    all_teams = sorted(set(cl_teams) | set(lg_teams))

    t_idx     = {t: i for i, t in enumerate(all_teams)}
    n         = len(all_teams)
    col_order = all_teams + ["Home_Adv"]

    def make_row(h, a):
        r = np.zeros(n + 1)
        r[-1] = 1.0   # Home_Adv
        if h in t_idx: r[t_idx[h]] =  1.0
        if a in t_idx: r[t_idx[a]] = -1.0
        return r

    rows_X = []; rows_y = []; rows_w = []

    # ── CL/Europa/Conf actual results ─────────────────────────────────────────
    for _, row in cl_sub.iterrows():
        days = max((ref_date - row["Date"]).days, 0)
        dw   = 0.5 ** (days / HALF_LIFE)
        X    = make_row(row["HomeTeam"], row["AwayTeam"])

        # Actual goal diff
        rows_X.append(X)
        rows_y.append(float(row["GoalDiff"]))
        rows_w.append(dw * CL_WEIGHT * (1 - MARKET_WEIGHT))

        # Market implied
        if pd.notna(row.get("mu_mkt")):
            dm = 0.5 ** (days / MARKET_HALF_LIFE)
            rows_X.append(X)
            rows_y.append(float(row["mu_mkt"]))
            rows_w.append(dm * CL_WEIGHT * MARKET_WEIGHT)

    # ── League games (cross-league signal) ────────────────────────────────────
    for _, row in lg_sub.iterrows():
        days  = max((ref_date - row["Date"]).days, 0)
        dw    = 0.5 ** (days / HALF_LIFE)
        X     = make_row(row["HomeTeam"], row["AwayTeam"])
        lw    = LEAGUE_WEIGHT_TIER1 if row["Div"] in TIER1 else LEAGUE_WEIGHT_TIER2

        rows_X.append(X)
        rows_y.append(float(row["GoalDiff"]))
        rows_w.append(dw * lw * (1 - MARKET_WEIGHT))

        if pd.notna(row.get("mu_mkt")):
            dm = 0.5 ** (days / MARKET_HALF_LIFE)
            rows_X.append(X)
            rows_y.append(float(row["mu_mkt"]))
            rows_w.append(dm * lw * MARKET_WEIGHT)

    # ── Priors ────────────────────────────────────────────────────────────────
    gpt   = max(len(cl_sub) * 2 / max(len(cl_teams), 1), 1.0)
    ap    = 3.0 / gpt
    tpw   = ap
    hpw   = ap * 5/3

    for t in all_teams:
        r = np.zeros(n + 1); r[t_idx[t]] = 1.0
        rows_X.append(r); rows_y.append(0.0); rows_w.append(tpw)

    r = np.zeros(n + 1); r[-1] = 1.0
    rows_X.append(r); rows_y.append(0.3); rows_w.append(hpw)

    # ── Solve ─────────────────────────────────────────────────────────────────
    Xa = np.array(rows_X)
    ya = np.array(rows_y)
    wa = np.array(rows_w)
    sq = np.sqrt(np.clip(wa, 0, None))

    beta, _, _, _ = np.linalg.lstsq(Xa * sq[:,None], ya * sq, rcond=None)
    if not np.all(np.isfinite(beta)):
        print("  WARNING: non-finite ratings")
        return None

    ratings  = {all_teams[i]: beta[i] for i in range(n)}
    home_adv = beta[-1]

    # Recentre so mean of Big 5 teams = 0
    big5_teams = [t for t in all_teams if t in set(lg_teams)]
    if big5_teams:
        mean_r = np.mean([ratings[t] for t in big5_teams])
        ratings = {t: r - mean_r for t, r in ratings.items()}

    # Count CL appearances per team
    cl_gp = {}
    for _, row in cl_sub.iterrows():
        cl_gp[row["HomeTeam"]] = cl_gp.get(row["HomeTeam"], 0) + 1
        cl_gp[row["AwayTeam"]] = cl_gp.get(row["AwayTeam"], 0) + 1

    return {
        "ratings":  ratings,
        "home_adv": home_adv,
        "cl_gp":    cl_gp,
        "teams":    all_teams,
        "ref_date": ref_date,
    }


def _recent_seasons(current, n=3):
    """Get n most recent seasons including current."""
    all_s = [
        "2015-16","2016-17","2017-18","2018-19","2019-20",
        "2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"
    ]
    if current in all_s:
        idx = all_s.index(current)
        return all_s[max(0, idx - n + 1): idx + 1]
    return [current]


# ── LEAGUE DIFFICULTY OFFSETS ─────────────────────────────────────────────────
def compute_league_offsets(cross_ratings, lg_df, season=CURRENT_SEASON):
    """
    For each Big 5 league, compute the offset between cross-league ratings
    and in-league ratings. This tells us how strong each league is relative
    to each other.
    """
    from mainmodel.part7_ratings import build_ratings_table

    offsets = {}
    for div in ["E0","D1","SP1","I1","F1"]:
        rt = build_ratings_table(div, season)
        if rt is None:
            continue
        league_r = {row["Team"]: row["Overall"] for _, row in rt.iterrows()}

        # For teams appearing in both, compute cross - league
        diffs = []
        for team, cross_r in cross_ratings["ratings"].items():
            if team in league_r:
                diffs.append(cross_r - league_r[team])
        if diffs:
            offsets[div] = np.mean(diffs)
            print(f"  {div} ({DIV_NAMES[div]}): offset = {np.mean(diffs):+.3f} "
                  f"(n={len(diffs)} teams)")

    return offsets


# ── PRINT RATINGS TABLE ───────────────────────────────────────────────────────
def print_cross_ratings(result, top_n=40):
    ratings  = result["ratings"]
    cl_gp    = result["cl_gp"]
    ref_date = result["ref_date"]

    sorted_r = sorted(ratings.items(), key=lambda x: -x[1])

    print(f"\n{'='*70}")
    print(f"  Cross-League Ratings  |  as of {ref_date.date()}")
    print(f"  Home advantage: {result['home_adv']:+.3f} goals")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Team':<25} {'Rating':>8}  {'CL GP':>6}  Bar")
    print(f"  {'-'*60}")

    for i, (team, r) in enumerate(sorted_r[:top_n], 1):
        gp  = cl_gp.get(team, 0)
        bar = "█" * max(0, int((r + 2) * 5))
        print(f"  {i:>3}  {team:<25} {r:>+8.3f}  {gp:>6}  {bar}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(season=CURRENT_SEASON):
    os.makedirs("crossleague/data", exist_ok=True)

    print("Loading data...")
    cl_df, lg_df = load_data()

    print(f"\nFitting cross-league ratings for {season}...")
    result = fit_cross_ratings(cl_df, lg_df, season)
    if result is None:
        print("Failed to fit ratings")
        return

    print_cross_ratings(result)

    # Save ratings
    rows = []
    for team, r in sorted(result["ratings"].items(), key=lambda x: -x[1]):
        rows.append({
            "Team":    team,
            "Rating":  round(r, 4),
            "CL_GP":   result["cl_gp"].get(team, 0),
            "season":  season,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv("crossleague/data/cross_ratings.csv", index=False)
    print(f"\nSaved → crossleague/data/cross_ratings.csv")

    # Also pickle for predict_cl.py
    import pickle
    with open("crossleague/data/cross_ratings.pkl", "wb") as f:
        pickle.dump(result, f)
    print("Saved → crossleague/data/cross_ratings.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default=CURRENT_SEASON)
    args = parser.parse_args()
    main(args.season)
