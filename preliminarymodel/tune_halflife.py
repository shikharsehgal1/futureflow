import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_football.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df             = df.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]
df["Result"]   = np.where(df["GoalDiff"] > 0, "H",
                 np.where(df["GoalDiff"] < 0, "A", "D"))

# Closing odds with opening fallback
for s, cs in [("H","CH"), ("D","CD"), ("A","CA")]:
    col_c, col_o = f"B365{cs}", f"B365{s}"
    df[f"odds_{s}"] = df[col_c].fillna(df[col_o]) if col_c in df.columns else df[col_o]

# Drop missing or zero odds
df = df.dropna(subset=["odds_H","odds_D","odds_A"])
df = df[(df["odds_H"] > 1) & (df["odds_D"] > 1) & (df["odds_A"] > 1)].copy()
df = df.reset_index(drop=True)

# Market implied probs (normalised to remove overround)
raw = np.column_stack([1/df["odds_H"], 1/df["odds_D"], 1/df["odds_A"]])
raw = raw / raw.sum(axis=1, keepdims=True)
df["p_H_mkt"] = raw[:, 0]
df["p_D_mkt"] = raw[:, 1]
df["p_A_mkt"] = raw[:, 2]

SIGMA = df["GoalDiff"].std()

# ── FIT RATINGS ───────────────────────────────────────────────────────────────
def fit_ratings(subset, half_life=None, prior_weight=0.5,
                prior_goal_diff=0.0, home_adv_prior=0.3):
    orig_index = subset.index.copy()
    subset     = subset.copy().reset_index(drop=True)

    if half_life is None:
        subset["w"] = 1.0
    else:
        days_ago     = (subset["Date"].max() - subset["Date"]).dt.days
        subset["w"]  = 0.5 ** (days_ago / half_life)

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

    X_prior, y_prior, w_prior = [], [], []
    for team in free_teams:
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 0
        X_prior.append([r[c] for c in col_order])
        y_prior.append(prior_goal_diff); w_prior.append(prior_weight)
    r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
    X_prior.append([r[c] for c in col_order])
    y_prior.append(home_adv_prior); w_prior.append(prior_weight)

    X_all  = np.vstack([X_real, np.array(X_prior)])
    y_all  = np.concatenate([y_real, np.array(y_prior)])
    w_all  = np.concatenate([w_real, np.array(w_prior)])
    sqrt_w = np.sqrt(w_all)

    beta, _, _, _ = np.linalg.lstsq(
        X_all * sqrt_w[:, None], y_all * sqrt_w, rcond=None)
    if not np.all(np.isfinite(beta)):
        return None

    return {"y_hat": X_real @ beta, "orig_index": orig_index}

# ── PROBABILITY CONVERSION ────────────────────────────────────────────────────
def mu_to_probs(mu, sigma=SIGMA):
    p_H = 1 - norm.cdf(0.5 / sigma - mu / sigma)
    p_A = norm.cdf(-0.5 / sigma - mu / sigma)
    p_D = 1 - p_H - p_A
    return np.clip([p_H, p_D, p_A], 1e-6, 1 - 1e-6)

def log_loss_col(df, mu_col):
    losses = []
    for _, r in df.iterrows():
        pH, pD, pA = mu_to_probs(r[mu_col])
        p = {"H": pH, "D": pD, "A": pA}[r["Result"]]
        losses.append(-np.log(max(p, 1e-6)))
    return np.mean(losses)

def mkt_log_loss(df):
    losses = []
    for _, r in df.iterrows():
        p = {"H": r["p_H_mkt"], "D": r["p_D_mkt"], "A": r["p_A_mkt"]}[r["Result"]]
        losses.append(-np.log(max(p, 1e-6)))
    return np.mean(losses)

# ── TUNE HALF-LIFE ────────────────────────────────────────────────────────────
half_lives = [30, 60, 90, 120, 180, 270, 365, None]
hl_labels  = [str(h) if h else "∞" for h in half_lives]

completed_seasons = sorted([s for s in df["season"].unique()
                             if s not in ("2024-25", "2025-26")])

print(f"Evaluating {len(half_lives)} half-lives across "
      f"{len(completed_seasons)} seasons...\n")

tune_results = []
for hl, label in zip(half_lives, hl_labels):
    season_losses = []
    for season in completed_seasons:
        season_df = df[df["season"] == season].copy()
        season_df["mu_model"] = np.nan

        for div, subset in season_df.groupby("Div"):
            res = fit_ratings(subset, half_life=hl)
            if res is None:
                continue
            season_df.loc[res["orig_index"], "mu_model"] = res["y_hat"]

        season_df = season_df.dropna(subset=["mu_model"])
        if len(season_df) == 0:
            continue
        season_losses.append(log_loss_col(season_df, "mu_model"))

    mean_ll = np.mean(season_losses)
    tune_results.append({"HalfLife": label, "MeanLogLoss": mean_ll})
    print(f"  Half-life={label:>4} days  |  Mean log-loss={mean_ll:.4f}")

best = min(tune_results, key=lambda x: x["MeanLogLoss"])
print(f"\n  Best half-life : {best['HalfLife']} days")
print(f"  Best log-loss  : {best['MeanLogLoss']:.4f}")

# ── OUT-OF-SAMPLE TEST ────────────────────────────────────────────────────────
# Fit on season N-1, predict season N — true out-of-sample evaluation
print("\n" + "=" * 55)
print("  OUT-OF-SAMPLE EVALUATION (train N-1, predict N)")
print("=" * 55)

oos_results = []
for i in range(1, len(completed_seasons)):
    train_season = completed_seasons[i-1]
    test_season  = completed_seasons[i]

    train_df = df[df["season"] == train_season].copy()
    test_df  = df[df["season"] == test_season].copy()
    test_df["mu_model"] = np.nan

    # Fit ratings on train season, apply to test season teams
    for div in test_df["Div"].unique():
        train_sub = train_df[train_df["Div"] == div]
        test_sub  = test_df[test_df["Div"] == div]
        if len(train_sub) == 0 or len(test_sub) == 0:
            continue

        # Get ratings from train season
        res = fit_ratings(train_sub, half_life=None)
        if res is None:
            continue

        # Build ratings dict from training beta
        orig = train_sub.copy().reset_index(drop=True)
        teams      = sorted(set(orig["HomeTeam"]) | set(orig["AwayTeam"]))
        anchor     = teams[0]
        free_teams = [t for t in teams if t != anchor]
        col_order  = free_teams + ["Home_Adv"]

        sqrt_w = np.ones(len(orig) + len(free_teams) + 1)
        X_prior_rows = []
        for team in free_teams:
            r = {t: 0 for t in col_order}; r["Home_Adv"] = 0
            X_prior_rows.append([r[c] for c in col_order])
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
        X_prior_rows.append([r[c] for c in col_order])

        def make_row(home, away, free_teams=free_teams, col_order=col_order):
            r = {t: 0 for t in col_order}
            r["Home_Adv"] = 1
            if home in free_teams: r[home] =  1
            if away in free_teams: r[away] = -1
            return [r[c] for c in col_order]

        X_train = np.array([make_row(r.HomeTeam, r.AwayTeam)
                            for _, r in orig.iterrows()])
        y_train = orig["GoalDiff"].values.astype(float)
        X_all   = np.vstack([X_train, np.array(X_prior_rows)])
        y_all   = np.concatenate([y_train,
                    [0.0]*len(free_teams) + [0.3]])
        w_all   = np.concatenate([np.ones(len(y_train)),
                    [0.5]*(len(free_teams)+1)])
        sqrt_w  = np.sqrt(w_all)
        beta, _, _, _ = np.linalg.lstsq(
            X_all * sqrt_w[:, None], y_all * sqrt_w, rcond=None)
        if not np.all(np.isfinite(beta)):
            continue

        ratings  = {anchor: 0.0, **{t: beta[i] for i, t in enumerate(free_teams)}}
        home_adv = beta[-1]

        # Apply to test season games
        test_orig = test_sub.copy().reset_index(drop=True)
        preds = []
        for _, r in test_orig.iterrows():
            h_rat = ratings.get(r.HomeTeam, 0.0)
            a_rat = ratings.get(r.AwayTeam, 0.0)
            preds.append(h_rat - a_rat + home_adv)

        test_df.loc[test_sub.index, "mu_model"] = preds

    test_df = test_df.dropna(subset=["mu_model"])
    if len(test_df) == 0:
        continue

    ll_model = log_loss_col(test_df, "mu_model")
    ll_mkt   = mkt_log_loss(test_df)
    oos_results.append({
        "TrainSeason": train_season,
        "TestSeason":  test_season,
        "N":           len(test_df),
        "LL_model":    ll_model,
        "LL_mkt":      ll_mkt,
        "Gap":         ll_mkt - ll_model,
    })
    print(f"  Train={train_season} → Test={test_season}  "
          f"N={len(test_df):4d}  "
          f"model={ll_model:.4f}  mkt={ll_mkt:.4f}  "
          f"gap={ll_mkt-ll_model:+.4f}")

oos_df = pd.DataFrame(oos_results)
print(f"\n  Mean OOS model log-loss  : {oos_df['LL_model'].mean():.4f}")
print(f"  Mean OOS market log-loss : {oos_df['LL_mkt'].mean():.4f}")
print(f"  Mean OOS gap             : {oos_df['Gap'].mean():+.4f}")
pos = (oos_df['Gap'] > 0).sum()
print(f"  Seasons model beat mkt   : {pos}/{len(oos_df)}")