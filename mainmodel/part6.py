import pandas as pd
import numpy as np
from scipy.special import factorial
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_with_probs.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

with open("data/mean_goals.pkl", "rb") as f:
    mg = pickle.load(f)
MEAN_HOME     = mg["home"]
MEAN_AWAY     = mg["away"]
COVID_SEASONS = ["2019-20", "2020-21"]
MIN_GAMES     = 10

# ── 2. FIT RATINGS ────────────────────────────────────────────────────────────
def fit_ratings(subset, ref_date, half_life=None, market_weight=0.0,
                prior_weight=0.5, ha_prior=0.3):
    """
    Fit goal-differential power ratings with two hyperparameters:

    half_life     : time decay — games half_life days ago get weight 0.5
                    None = no decay (equal weight)
    market_weight : how much to pull each game's target toward mu_mkt
                    0.0 = pure goal differential
                    1.0 = pure market-implied goal differential

    prior_weight scales with data sparsity — heavier priors early in the
    season prevent overfitting when there are few games per team.
    With N games and T teams, effective games per team ≈ N*2/T.
    Prior weight = base_prior / sqrt(games_per_team) so it diminishes
    naturally as more data accumulates.
    """
    orig_index = subset.index.copy()
    subset     = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    subset["y_target"] = ((1 - market_weight) * subset["GoalDiff"] +
                                market_weight  * subset["mu_mkt"])

    subset["w"] = 1.0 if half_life is None else \
        0.5 ** ((pd.Timestamp(ref_date) - subset["Date"]).dt.days / half_life)

    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = teams[0]
    free_teams = [t for t in teams if t != anchor]
    col_order  = free_teams + ["Home_Adv"]

    # Two independent adaptive priors — separate degrees of freedom:
    #   team_prior_w : shrinks each team rating toward 0 (league average)
    #   hfa_prior_w  : anchors HFA toward ha_prior (0.3)
    # Previously coupled (same weight, same row). Now decoupled so you can
    # e.g. strongly anchor HFA while letting team ratings float more freely.
    #
    # BASE_TEAM_PRIOR = 3.0 : fades naturally as games_per_team grows
    # BASE_HFA_PRIOR  = 5.0 : HFA is stable across seasons, anchor harder
    n_games        = len(subset)
    n_teams        = len(teams)
    games_per_team = max(n_games * 2 / n_teams, 0.1)
    team_prior_w   = 3.0 / games_per_team   # strong early, fades with data
    hfa_prior_w    = 5.0 / games_per_team   # stronger anchor on HFA

    def make_row(h, a):
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
        if h in free_teams: r[h] =  1
        if a in free_teams: r[a] = -1
        return [r[c] for c in col_order]

    X_real = np.array([make_row(r.HomeTeam, r.AwayTeam)
                       for _, r in subset.iterrows()])
    y_real = subset["y_target"].values.astype(float)
    w_real = subset["w"].values.astype(float)

    # ── Prior rows — now decoupled ────────────────────────────────────────────
    # Team prior:  [team=+1, HFA=0] = 0
    #   → rating_team = 0  → shrinks team toward league average
    #   → HFA NOT included so team shrinkage does not drag HFA estimate
    #
    # HFA prior:   [HFA=+1] = ha_prior
    #   → HFA = ha_prior  → pure HFA anchor, independent of team ratings
    #   → weighted by hfa_prior_w (stronger than team_prior_w)
    Xp, yp, wp = [], [], []

    # Team shrinkage — one row per team, HFA column = 0
    for t in free_teams:
        r = {c: 0 for c in col_order}   # HFA stays 0
        r[t] = 1
        Xp.append([r[c] for c in col_order]); yp.append(0.0); wp.append(team_prior_w)

    # HFA anchor — pure HFA signal, no team indicators
    r = {c: 0 for c in col_order}; r["Home_Adv"] = 1
    Xp.append([r[c] for c in col_order]); yp.append(ha_prior); wp.append(hfa_prior_w)

    Xa = np.vstack([X_real, np.array(Xp)])
    ya = np.concatenate([y_real, np.array(yp)])
    wa = np.concatenate([w_real, np.array(wp)])
    sq = np.sqrt(wa)

    beta, _, _, _ = np.linalg.lstsq(Xa * sq[:,None], ya * sq, rcond=None)
    if not np.all(np.isfinite(beta)): return None

    coefs = dict(zip(col_order, beta))

    is_covid = subset["season"].isin(COVID_SEASONS)
    if is_covid.any() and not is_covid.all():
        Xnc = X_real[~is_covid.values]
        ync = subset["GoalDiff"].values[~is_covid.values]
        wnc = w_real[~is_covid.values]
        coefs["Home_Adv"] = float(np.average(
            ync - Xnc[:,:-1] @ beta[:-1], weights=wnc))

    y_hat     = X_real @ beta
    residuals = subset["GoalDiff"].values - y_hat
    rmse      = float(np.sqrt(np.mean(residuals**2)))
    ss_res    = np.sum(w_real * residuals**2)
    ss_tot    = np.sum(w_real * (subset["GoalDiff"].values -
                np.average(subset["GoalDiff"].values, weights=w_real))**2)
    r2        = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    return {
        "y_hat":         y_hat,
        "residuals":     residuals,
        "rmse":          rmse,
        "r2":            r2,
        "home_adv":      coefs["Home_Adv"],
        "ratings":       {anchor: 0.0, **{t: coefs[t] for t in free_teams}},
        "orig_index":    orig_index,
        "team_prior_w": team_prior_w, "hfa_prior_w": hfa_prior_w,
    }


# ── 3. EVALUATE HYPERPARAMETERS ───────────────────────────────────────────────
def evaluate_params_walkforward(df, half_life, market_weight, min_games=MIN_GAMES):
    """
    Walk-forward evaluation: for each gameday D, fit on Date < D,
    predict games on date D. Measures true out-of-sample RMSE and R².
    This is the correct evaluation for operational hyperparameter tuning.
    """
    all_actual = []
    all_pred   = []

    for (div, season), group in df.groupby(["Div","season"]):
        group    = group.sort_values("Date").reset_index(drop=True)
        gamedays = group["Date"].unique()

        for gd in gamedays:
            train   = group[group["Date"] < gd]
            predict = group[group["Date"] == gd]
            if len(train) < min_games: continue

            res = fit_ratings(train,
                              ref_date=gd,
                              half_life=half_life,
                              market_weight=market_weight)
            if res is None: continue

            ratings  = res["ratings"]
            home_adv = res["home_adv"]

            for _, row in predict.iterrows():
                mu = (ratings.get(row["HomeTeam"], 0.0) -
                      ratings.get(row["AwayTeam"], 0.0) + home_adv)
                all_actual.append(row["GoalDiff"])
                all_pred.append(mu)

    if not all_actual:
        return np.nan, np.nan

    actual = np.array(all_actual)
    pred   = np.array(all_pred)
    rmse   = float(np.sqrt(np.mean((actual - pred)**2)))
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2     = float(1 - ss_res / ss_tot)
    return rmse, r2


# ── 4. GRID SEARCH ────────────────────────────────────────────────────────────
# Use completed seasons only for tuning
tune_df = df[~df["season"].isin(["2024-25","2025-26"])].copy()

half_lives     = [None, 90, 180, 365]
market_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

print("Grid search: half_life × market_weight")
print("  (Walk-forward OOS: fit on Date < D, predict Date = D)")
print(f"\n  {'HalfLife':>10} {'MktWeight':>10} {'RMSE':>8} {'R²':>8}")

best_rmse   = np.inf
best_params = {}
grid_rows   = []

for hl in half_lives:
    for mw in market_weights:
        rmse, r2 = evaluate_params_walkforward(tune_df, hl, mw)
        hl_label = str(hl) if hl else "∞"
        print(f"  {hl_label:>10}  {mw:>10.1f}  {rmse:>8.4f}  {r2:>8.4f}")
        grid_rows.append({"half_life": hl, "market_weight": mw,
                          "rmse": rmse, "r2": r2})
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = {"half_life": hl, "market_weight": mw}

print(f"\n  Best: half_life={best_params['half_life']}  "
      f"market_weight={best_params['market_weight']}  "
      f"(RMSE={best_rmse:.4f})")


# ── 5. FULL RATINGS WITH BEST PARAMS ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FULL RATINGS — BEST PARAMS")
print(f"{'='*65}")

results = {}
summary_rows = []

for (div, season), subset in tune_df.groupby(["Div","season"]):
    res = fit_ratings(subset,
                      ref_date=subset["Date"].max(),
                      half_life=best_params["half_life"],
                      market_weight=best_params["market_weight"])
    if res is None: continue
    results[(div, season)] = res
    summary_rows.append({
        "Div": div, "Season": season,
        "N": len(subset),
        "RMSE": round(res["rmse"], 4),
        "R2":   round(res["r2"],   4),
        "HomeAdv": round(res["home_adv"], 3),
    })

summary = pd.DataFrame(summary_rows).sort_values(["Div","Season"])
print(summary.to_string(index=False))


# ── 6. EARLY SEASON ANALYSIS ──────────────────────────────────────────────────
# The doc specifically flags early season as sensitive to dummy row weight.
# Here we test how goodness of fit evolves as more games are added.
print(f"\n{'='*65}")
print(f"  EARLY SEASON GOF — EPL 2023-24")
print(f"  (adaptive prior = base_prior / sqrt(games_per_team))")
print(f"{'='*65}")
print(f"  {'Games used':>12} {'Prior_w':>8} {'RMSE':>8} {'R²':>8} {'HomeAdv':>9}")

epl = df[(df["Div"]=="E0") & (df["season"]=="2023-24")].copy()
epl = epl.sort_values("Date").reset_index(drop=True)

for n_games in [5, 10, 15, 20, 30, 38, len(epl)//2, len(epl)]:
    subset = epl.iloc[:n_games]
    if len(subset) < 4: continue
    res = fit_ratings(subset,
                      ref_date=subset["Date"].max(),
                      half_life=best_params["half_life"],
                      market_weight=best_params["market_weight"])
    if res is None: continue
    print(f"  {n_games:>12}  {res['team_prior_w']:>8.3f}  {res['rmse']:>8.4f}  "
          f"{res['r2']:>8.4f}  {res['home_adv']:>9.3f}")


# ── 7. SPOT CHECK — EPL 2023-24 FULL SEASON ───────────────────────────────────
key = ("E0", "2023-24")
if key in results:
    r = results[key]
    ratings_df = (pd.DataFrame({"Team": list(r["ratings"].keys()),
                                 "Rating": list(r["ratings"].values())})
                  .sort_values("Rating", ascending=False)
                  .reset_index(drop=True))
    ratings_df.index += 1
    anchor = ratings_df["Team"].iloc[-1]
    print(f"\n{'='*65}")
    print(f"  EPL 2023-24 RATINGS  (anchor={anchor})")
    print(f"{'='*65}")
    print(ratings_df.to_string())
    print(f"\n  Home advantage: {r['home_adv']:+.3f} goals")
    print(f"  RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")


# ── 8. SAVE ───────────────────────────────────────────────────────────────────
pd.DataFrame(grid_rows).to_csv("data/part6_grid.csv", index=False)
summary.to_csv("data/part6_summary.csv", index=False)
pd.DataFrame([best_params]).to_csv("data/part6_best_params.csv", index=False)
print(f"\nSaved grid search  → data/part6_grid.csv")
print(f"Saved summary      → data/part6_summary.csv")
print(f"Saved best params  → data/part6_best_params.csv")