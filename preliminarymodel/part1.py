import pandas as pd
import numpy as np
from io import StringIO

# ── 1. DATA ───────────────────────────────────────────────────────────────────
# Toy data — will be replaced by real CSV in Part 3/4
raw = """AwayTeam,HomeTeam,AwayGoals,HomeGoals,Date,Div,season
DET,TOR,5,1,2024-10-01,E0,2024-25
BOS,NYR,3,4,2024-10-05,E0,2024-25
MON,CHI,5,0,2024-10-08,E0,2024-25
DET,MON,2,4,2024-10-12,E0,2024-25
NYR,TOR,3,4,2024-10-15,E0,2024-25
CHI,BOS,0,1,2024-10-19,E0,2024-25
TOR,BOS,0,3,2024-10-22,E0,2024-25
MON,NYR,1,2,2024-10-26,E0,2024-25
CHI,DET,1,6,2024-10-29,E0,2024-25
BOS,DET,2,3,2024-11-02,E0,2024-25
NYR,CHI,4,3,2024-11-06,E0,2024-25
TOR,MON,0,5,2024-11-10,E0,2024-25
DET,CHI,2,3,2024-11-14,E0,2024-25
BOS,NYR,1,3,2024-11-18,E0,2024-25"""

df = pd.read_csv(StringIO(raw))
df["Date"] = pd.to_datetime(df["Date"])

# ── 2. PARAMETERS ─────────────────────────────────────────────────────────────
HALF_LIFE_DAYS  = 60
PRIOR_WEIGHT    = 0.5
PRIOR_GOAL_DIFF = 0.0
HOME_ADV_PRIOR  = 0.3

# ── 3. CORE FITTING FUNCTION ──────────────────────────────────────────────────
def fit_ratings(subset, half_life=HALF_LIFE_DAYS, prior_weight=PRIOR_WEIGHT,
                prior_goal_diff=PRIOR_GOAL_DIFF, home_adv_prior=HOME_ADV_PRIOR):
    """
    Fit a goal-differential power rating model on a single league-season subset.
    Teams, anchor, and all indicators are derived exclusively from this subset —
    no information leaks in from other leagues or seasons.

    Returns:
        ratings_df  : DataFrame with Team, Rating columns (sorted best to worst)
        home_adv    : float, fitted home advantage in goals
        diagnostics : dict with R², RMSE, MAE, N
        df_out      : input df with Predicted and Residual columns added
    """
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    # Time decay — relative to most recent game in THIS subset only
    ref_date   = subset["Date"].max()
    days_ago   = (ref_date - subset["Date"]).dt.days
    subset["w"] = 0.5 ** (days_ago / half_life)

    # Teams defined exclusively from this subset
    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = teams[0]
    free_teams = [t for t in teams if t != anchor]
    col_order  = free_teams + ["Home_Adv"]

    def make_row(home, away):
        r = {t: 0 for t in col_order}
        r["Home_Adv"] = 1
        if home in free_teams: r[home] =  1
        if away in free_teams: r[away] = -1
        return [r[c] for c in col_order]

    # Real game rows
    X_real = np.array([make_row(r.HomeTeam, r.AwayTeam) for _, r in subset.iterrows()])
    y_real = subset["GoalDiff"].values.astype(float)
    w_real = subset["w"].values.astype(float)

    # Prior dummy rows — one per free team + one for home advantage
    X_prior, y_prior, w_prior = [], [], []
    for team in free_teams:
        r = {t: 0 for t in col_order}
        r["Home_Adv"] = 0
        X_prior.append([r[c] for c in col_order])
        y_prior.append(prior_goal_diff)
        w_prior.append(prior_weight)

    r = {t: 0 for t in col_order}
    r["Home_Adv"] = 1
    X_prior.append([r[c] for c in col_order])
    y_prior.append(home_adv_prior)
    w_prior.append(prior_weight)

    X_prior = np.array(X_prior)
    y_prior = np.array(y_prior)
    w_prior = np.array(w_prior)

    # Weighted OLS via sqrt(w) scaling
    X_all = np.vstack([X_real, X_prior])
    y_all = np.concatenate([y_real, y_prior])
    w_all = np.concatenate([w_real, w_prior])

    sqrt_w = np.sqrt(w_all)
    beta, _, _, _ = np.linalg.lstsq(X_all * sqrt_w[:, None], y_all * sqrt_w, rcond=None)
    coefs = dict(zip(col_order, beta))

    # Ratings table
    ratings = {anchor: 0.0, **{t: coefs[t] for t in free_teams}}
    ratings_df = (
        pd.DataFrame({"Team": list(ratings.keys()), "Rating": list(ratings.values())})
        .sort_values("Rating", ascending=False)
        .reset_index(drop=True)
    )
    ratings_df.index += 1
    home_adv = coefs["Home_Adv"]

    # Diagnostics on real games only
    y_hat     = X_real @ beta
    residuals = y_real - y_hat
    ss_res    = np.sum(w_real * residuals**2)
    ss_tot    = np.sum(w_real * (y_real - np.average(y_real, weights=w_real))**2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    diagnostics = {
        "R2":   round(r2, 4),
        "RMSE": round(float(np.sqrt(np.mean(residuals**2))), 4),
        "MAE":  round(float(np.mean(np.abs(residuals))), 4),
        "N":    len(y_real),
    }

    subset["Predicted"] = y_hat.round(3)
    subset["Residual"]  = residuals.round(3)

    return ratings_df, home_adv, diagnostics, subset


# ── 4. RUN ACROSS ALL LEAGUE-SEASONS ─────────────────────────────────────────
def run_all(df, div_col="Div", season_col="season"):
    """
    Iterate over every (Div, season) combination and fit an independent model.
    Each fit sees only its own teams — no cross-contamination.
    """
    results = {}
    groups  = df.groupby([div_col, season_col])
    print(f"Fitting {len(groups)} league-season(s) independently...\n")

    for (div, season), subset in groups:
        ratings_df, home_adv, diag, fitted = fit_ratings(subset)
        results[(div, season)] = {
            "ratings":     ratings_df,
            "home_adv":    home_adv,
            "diagnostics": diag,
            "fitted":      fitted,
        }
        print(f"{'='*55}")
        print(f"  {div}  |  {season}  ({diag['N']} games)")
        print(f"{'='*55}")
        print(ratings_df.to_string())
        print(f"\n  Home Advantage : {home_adv:+.3f} goals")
        print(f"  R²={diag['R2']}  RMSE={diag['RMSE']}  MAE={diag['MAE']}")

    return results


# ── 5. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all(df)

    # Example: pull ratings for a specific league-season
    key = ("E0", "2024-25")
    if key in results:
        print(f"\n\nExample lookup — {key}:")
        print(results[key]["ratings"])