import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_football.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df             = df.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})

# ── 2. PARAMETERS ─────────────────────────────────────────────────────────────
# Half-life tuning showed ∞ (no decay) is optimal for full-season ratings.
# Use a finite value (e.g. 90) only when you want recent-form emphasis.
HALF_LIFE_DAYS  = None   # None = no decay
PRIOR_WEIGHT    = 0.5
PRIOR_GOAL_DIFF = 0.0
HOME_ADV_PRIOR  = 0.3

# COVID seasons had no fans → home advantage suppressed.
# We keep these games for team rating estimation but re-estimate
# home advantage using only non-COVID games within the same season.
COVID_SEASONS = ["2019-20", "2020-21"]

# ── 3. FIT RATINGS ────────────────────────────────────────────────────────────
def fit_ratings(subset, half_life=HALF_LIFE_DAYS, prior_weight=PRIOR_WEIGHT,
                prior_goal_diff=PRIOR_GOAL_DIFF, home_adv_prior=HOME_ADV_PRIOR):
    """
    Fit a goal-differential power rating model on a single league-season subset.
    All teams, anchor, and indicators are derived exclusively from this subset.

    Returns (ratings_df, home_adv, diagnostics, fitted_df) or
            (None, None, None, subset) if the solution is degenerate.
    """
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    # Time decay — None = equal weight across full season
    if half_life is None:
        subset["w"] = 1.0
    else:
        days_ago    = (subset["Date"].max() - subset["Date"]).dt.days
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

    X_real = np.array([make_row(r.HomeTeam, r.AwayTeam) for _, r in subset.iterrows()])
    y_real = subset["GoalDiff"].values.astype(float)
    w_real = subset["w"].values.astype(float)

    # Bayesian priors as dummy rows
    # One per free team (pulls rating toward league average)
    # One for home advantage (anchors toward HOME_ADV_PRIOR)
    X_prior, y_prior, w_prior = [], [], []
    for team in free_teams:
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 0
        X_prior.append([r[c] for c in col_order])
        y_prior.append(prior_goal_diff)
        w_prior.append(prior_weight)
    r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
    X_prior.append([r[c] for c in col_order])
    y_prior.append(home_adv_prior)
    w_prior.append(prior_weight)

    # Weighted OLS via sqrt(w) scaling
    X_all  = np.vstack([X_real, np.array(X_prior)])
    y_all  = np.concatenate([y_real, np.array(y_prior)])
    w_all  = np.concatenate([w_real, np.array(w_prior)])
    sqrt_w = np.sqrt(w_all)

    beta, _, _, _ = np.linalg.lstsq(
        X_all * sqrt_w[:, None], y_all * sqrt_w, rcond=None)

    # Guard against degenerate solutions
    if not np.all(np.isfinite(beta)):
        return None, None, None, subset

    coefs = dict(zip(col_order, beta))

    # COVID correction — re-estimate home advantage from non-COVID games only.
    # Team ratings are kept as-is (goal differentials remain valid).
    season_val = subset.get("season", pd.Series(dtype=str))
    is_covid   = season_val.isin(COVID_SEASONS)
    if is_covid.any() and not is_covid.all():
        X_nc      = X_real[~is_covid.values]
        y_nc      = y_real[~is_covid.values]
        w_nc      = w_real[~is_covid.values]
        team_pred = X_nc[:, :-1] @ beta[:-1]
        resid_nc  = y_nc - team_pred
        coefs["Home_Adv"] = float(np.average(resid_nc, weights=w_nc))

    # Ratings table
    ratings    = {anchor: 0.0, **{t: coefs[t] for t in free_teams}}
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
        "R2":   round(float(r2), 4),
        "RMSE": round(float(np.sqrt(np.mean(residuals**2))), 4),
        "MAE":  round(float(np.mean(np.abs(residuals))), 4),
        "N":    len(y_real),
    }

    subset["Predicted"] = y_hat.round(3)
    subset["Residual"]  = residuals.round(3)

    return ratings_df, home_adv, diagnostics, subset


# ── 4. RUN ACROSS ALL LEAGUE-SEASONS ──────────────────────────────────────────
def run_all(df):
    """
    Fit an independent model for every (Div, season) combination.
    Each fit sees only its own teams — no cross-contamination.
    """
    results = {}
    groups  = df.groupby(["Div", "season"])
    print(f"Fitting {len(groups)} league-seasons independently...\n")

    summary_rows = []
    for (div, season), subset in groups:
        ratings_df, home_adv, diag, fitted = fit_ratings(subset)
        if ratings_df is None:
            print(f"  SKIPPED {div} {season} — degenerate solution")
            continue
        results[(div, season)] = {
            "ratings":     ratings_df,
            "home_adv":    home_adv,
            "diagnostics": diag,
            "fitted":      fitted,
        }
        summary_rows.append({
            "Div":     div,
            "Season":  season,
            "N":       diag["N"],
            "R2":      diag["R2"],
            "RMSE":    diag["RMSE"],
            "MAE":     diag["MAE"],
            "HomeAdv": round(home_adv, 3),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["Div", "Season"])
    return results, summary


# ── 5. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, summary = run_all(df)

    print("=" * 75)
    print("  DIAGNOSTICS ACROSS ALL LEAGUE-SEASONS")
    print("=" * 75)
    print(summary.to_string(index=False))

    # Spot check — EPL 2023-24
    key = ("E0", "2023-24")
    if key in results:
        r = results[key]
        anchor = r["ratings"]["Team"].iloc[-1]
        print(f"\n{'='*55}")
        print(f"  RATINGS: EPL 2023-24  (anchor = {anchor})")
        print(f"{'='*55}")
        print(r["ratings"].to_string())
        print(f"\n  Home advantage : {r['home_adv']:+.3f} goals")
        print(f"  R²={r['diagnostics']['R2']}  "
              f"RMSE={r['diagnostics']['RMSE']}  "
              f"MAE={r['diagnostics']['MAE']}")

# ── 6. HELPER — LOOK UP ANY LEAGUE-SEASON ─────────────────────────────────────
def show_ratings(div, season):
    key = (div, season)
    if key not in results:
        print(f"No data for {key}.")
        return
    r = results[key]
    print(f"\n{div} | {season}  (anchor = {r['ratings']['Team'].iloc[-1]})")
    print(r["ratings"].to_string())
    print(f"Home advantage : {r['home_adv']:+.3f} goals")
    print(f"R²={r['diagnostics']['R2']}  RMSE={r['diagnostics']['RMSE']}")


def predict_match(div, season, home, away, neutral=False):
    key = (div, season)
    if key not in results:
        print(f"No data for {key}.")
        return
    r        = results[key]
    ratings  = r["ratings"].set_index("Team")["Rating"].to_dict()
    home_adv = 0 if neutral else r["home_adv"]
    h_rat    = ratings.get(home, 0.0)
    a_rat    = ratings.get(away, 0.0)
    pred     = h_rat - a_rat + home_adv
    venue    = "neutral" if neutral else "home"
    print(f"\n{home} vs {away} ({venue})  [{div} {season}]")
    print(f"  Predicted goal diff : {pred:+.3f}  (positive = {home} win)")